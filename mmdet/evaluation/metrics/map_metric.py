# Copyright (c) OpenMMLab. All rights reserved.
import json
from tqdm import tqdm
import os
import math
import wandb
import shutil
from typing import Dict, Optional, Sequence
import pickle
import numpy as np
import pandas as pd
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results

class WandbLogger:
    def __init__(self, project: str, name: str, id: str):
        self.project = project
        self.name = name
        self.id = id

    def log(self, data: Dict[str, float], commit: bool = True):
        with wandb.init(project=self.project, name=self.name, id=self.id, resume="allow") as run:
            run.log(data, commit=commit)
            run.finish()


@METRICS.register_module()
class MAPMetric(BaseMetric):
    """Custom evaluation metric.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        working_dir (str): The directory to store temporary files.
        confidence_threshold (float): The confidence threshold for filtering
            predictions. Defaults to 0.01
        iou_threshold (float): The IoU threshold for filtering predictions.
            Defaults to 0.5
    """
    default_prefix: Optional[str] = 'custom'

    def __init__(self,working_dir: str, collect_device: str = 'cpu',
                 prefix: Optional[str] = None, confidence_threshold: float = 0.01,
                iou_threshold: float = 0.5, run_id: Optional[str] = None, 
                run_project: Optional[str] = None, run_name: Optional[str] = None,) -> None:
        if not os.path.exists(working_dir):
            raise FileNotFoundError(
                f"Working directory {working_dir} does not exist.")
        
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.working_dir = working_dir
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.run_id =run_id
        self.run_project = run_project
        self.run_name = run_name
        if self.run_id is not None:
            self.wandb_logger = WandbLogger(project=run_project, name=run_name, id=run_id)


    def results2dataframe(self, results: Sequence[dict]) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.
        Args:
            results (Sequence[dict]): The results of the dataset.
        """
        prediction_results = []
        pickle.dump(results, open(f"/kaggle/working/results.pkl", "wb"))
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['Image_ID'] = image_id
                xmin, ymin, xmax, ymax = bboxes[i]
                data['xmin'] = float(xmin)
                data['ymin'] = float(ymin)
                data['xmax'] = float(xmax)
                data['ymax'] = float(ymax)
                data['confidence'] = float(scores[i])
                data['class'] = self.dataset_meta['classes'][int(label)]
                prediction_results.append(data)

        return pd.DataFrame(prediction_results)

    def gt_to_dataframe(self, gt_dicts: Sequence[dict]) -> pd.DataFrame:
        """Convert ground truth annotations to a pandas DataFrame.
        Args:
            gt_dicts (Sequence[dict]): The ground truth annotations.
        """
        annotations = []
        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            bboxes = gt_dict['anns']['bboxes'].numpy()   
            labels = gt_dict['anns']['labels'].numpy()   
            for label, bbox in zip(labels, bboxes):
                data = dict()
                data['Image_ID'] = img_id
                xmin, ymin, xmax, ymax = bbox
                data['xmin'] = float(xmin)
                data['ymin'] = float(ymin)
                data['xmax'] = float(xmax)
                data['ymax'] = float(ymax)
                data['class'] = self.dataset_meta['classes'][int(label)]
                annotations.append(data)

        return pd.DataFrame(annotations)

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        # print(data_samples[0].keys())
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            gt['anns'] = data_sample['gt_instances']
            # add converted result to the results list
            self.results.append((gt, result))

    def error(self, msg: str) -> None:
        """Print error message and exit the program."""
        print("error: %s" % msg)
        quit()

    def log_average_miss_rate(self, prec, rec, num_images):
        """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
            State of the Art." Pattern Analysis and Machine Intelligence, IEEE
            Transactions on 34.4 (2012): 743 - 761.
        """
        # if there were no detections of that class
        if prec.size == 0:
            lamr = 0
            mr   = 1
            fppi = 0
            return lamr, mr, fppi
        fppi = 1 - prec
        mr   = 1 - rec
        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp   = np.insert(mr, 0, 1.0)
        ref      = np.logspace(-2.0, 0.0, num=9) # Use 9 evenly spaced reference points in log-space
        for i, ref_i in enumerate(ref):
            # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
            j      = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i] = mr_tmp[j]
        # log(0) is undefined, so we use the np.maximum(1e-10, ref)
        lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
        return lamr, mr, fppi

    def voc_ap(self, rec: list, prec: list) -> tuple:
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #     range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #     range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
        This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap, mrec, mpre

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """

        # split gt and prediction list
        gts, preds = zip(*results)
        df_gt = self.gt_to_dataframe(gts)
        df_pred = self.results2dataframe(preds)
        df_pred = df_pred[df_pred['confidence'] >= self.confidence_threshold]

        try:
            ap_dict = self.calculate_map(df_gt, df_pred)
        except ValueError as e:
            print(f"Error calculating mAP: {e}\nGround truth nunique: {df_gt['Image_ID'].nunique()}\tPrediction nunique: {df_pred['Image_ID'].nunique()}")
            shutil.rmtree(f"{self.working_dir}/temp_files/")
            df_gt = df_gt[df_gt['Image_ID'].isin(df_pred['Image_ID'])].copy()
            ap_dict = self.calculate_map(df_gt, df_pred)        
        finally:
            if self.run_id is not None:
                self.wandb_logger.log(ap_dict, commit=False)
            shutil.rmtree(f"{self.working_dir}/temp_files/")
        return ap_dict
        
    def calculate_map(self, df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> Dict[str, float]:
        temp_files_path = f"{self.working_dir}/temp_files"
        MINOVERLAP = self.iou_threshold
        # make sure that the cwd() is the location of the python script (so that every path makes sense)
        # os.chdir(os.path.dirname(os.path.abspath(__file__)))
        # print(os.getcwd())
        os.makedirs(temp_files_path) # Make a folder under temp_files for working data
        """
        ground-truth
            Load each of the ground-truth files into a temporary ".json" file.
            Create a list of all the class names present in the ground-truth (gt_classes).
        """
        gt_counter_per_class     = {}
        counter_images_per_class = {}
        ####__******* PANDAS
        df_gt_check = df_gt["Image_ID"].to_list()
        df_gt["boxes"] = df_gt[["class", "ymin", "xmin", "ymax", "xmax"]].values.tolist()
        df_gt_dict     = df_gt.groupby("Image_ID")["boxes"].apply(list).to_dict()
        #### read sample submission
        submission_list   = list(df_pred.columns)
        expected_list     = ["Image_ID", "class", "confidence", "ymin", "xmin", "ymax", "xmax"]
        not_in_list       = set(expected_list) - set(submission_list)
        for column in not_in_list:
            error_msg = "Missing columns {} on the submission file".format(column)
            self.error(error_msg)
        df_pred_check = df_pred["Image_ID"].to_list()
        current_submission      = []
        # check the image id boxes in submission are the same with those in df_gt
        for image_id in df_gt_check:
            if image_id in df_pred_check:
                current_submission.append(image_id)
            else:
                error_msg = "Error. Image ID {} not found in submission file:".format(image_id)
                raise ValueError(error_msg)
        df_pred          = df_pred[df_pred['Image_ID'].isin(current_submission)]
        df_pred["boxes"] = df_pred[["class", "confidence", "ymin", "xmin", "ymax", "xmax"]].values.tolist()
        df_pred_dict     = df_pred.groupby("Image_ID")["boxes"].apply(list).to_dict()
        gt_files                   = []
        for k, v in tqdm(df_gt_dict.items(), desc="Loading ground-truth files"):
            file_id    = k
            lines_list = [" ".join([str(item) for item in box]) for box in v]
            ### pandas
            # create ground-truth dictionary
            bounding_boxes       = []
            is_difficult         = False
            already_seen_classes = []
            for line in lines_list:
                try:
                    if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult                                     = True
                    else:
                        class_name, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: Image  " + k + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                    error_msg += " Received: " + line
                    error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                    self.error(error_msg)
                # check if class is in the ignore list, if yes skip
                bbox = left + " " + top + " " + right + " " + bottom
                if is_difficult:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                    is_difficult = False
                else:
                    bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                    # count that object
                    if class_name in gt_counter_per_class:
                        gt_counter_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        gt_counter_per_class[class_name] = 1
                    if class_name not in already_seen_classes:
                        if class_name in counter_images_per_class:
                            counter_images_per_class[class_name] += 1
                        else:
                            # if class didn't exist yet
                            counter_images_per_class[class_name] = 1
                        already_seen_classes.append(class_name)
            # dump bounding_boxes into a ".json" file
            new_temp_file = temp_files_path + "/" + str(file_id) + "_ground_truth.json"
            gt_files.append(new_temp_file)
            with open(new_temp_file, "w") as outfile:
                json.dump(bounding_boxes, outfile)
        gt_classes = list(gt_counter_per_class.keys())
        gt_classes = sorted(gt_classes) # let's sort the classes alphabetically
        n_classes  = len(gt_classes)
        """
        detection-results
            Load each of the detection-results files into a temporary ".json" file.
        """
        for class_index, class_name in enumerate(gt_classes):
            bounding_boxes = []
            for k, v in df_pred_dict.items():
                file_id = k
                if class_index == 0:
                    if not k in df_gt_dict.keys():
                        error_msg = "Error. Image ID {} not found in df_gt file:".format(k)
                        self.error(error_msg)
                lines = [" ".join([str(item) for item in box]) for box in v]
                for line in lines:
                    try:
                        tmp_class_name, confidence, left, top, right, bottom = line.split()
                    except ValueError:
                        error_msg = "Error: Image " + k + " in the wrong format.\n"
                        error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                        error_msg += " Received: " + line
                        self.error(error_msg)
                    if tmp_class_name == class_name:
                        # print("match")
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
            # sort detection-results by decreasing confidence
            bounding_boxes.sort(key=lambda x: float(x["confidence"]), reverse=True)
            with open(temp_files_path + "/" + class_name + "_dr.json", "w") as outfile:
                json.dump(bounding_boxes, outfile)
        """
        Calculate the AP for each class
        """
        sum_AP               = 0.0
        ap_dictionary        = {}
        lamr_dictionary      = {}
        count_true_positives = {}
        # open file to store the output
        # with open(output_files_path + "/output.txt", 'w') as output_file:
        #     output_file.write("# AP and precision/recall per class\n")
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
                Load detection-results of that class
            """
            dr_file = temp_files_path + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))
            """
                Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, detection in tqdm(enumerate(dr_data), desc=f"Matching {class_name} detections to ground-truth", total=nd):
                file_id           = detection["file_id"]
                gt_file           = temp_files_path + "/" + str(file_id) + "_ground_truth.json" # open ground-truth with that file_id
                ground_truth_data = json.load(open(gt_file))
                ovmax             = -1
                gt_match          = -1
                bb                = [float(x) for x in detection["bbox"].split()] # load detected object bounding-box
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi   = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw   = bi[2] - bi[0] + 1
                        ih   = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (
                                (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                                + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                                - iw * ih
                            )
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax    = ov
                                gt_match = obj
                # set minimum overlap
                min_overlap = MINOVERLAP
                # if specific_iou_flagged:
                #     if class_name in specific_iou_classes:
                #         index = specific_iou_classes.index(class_name)
                #         min_overlap = float(iou_list[index])
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            tp[idx]                           = 1 # true positive
                            gt_match["used"]                  = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, "w") as f:
                                f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"
            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum  += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum  += val
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            ap, mrec, mprec           = self.voc_ap(rec[:], prec[:])
            sum_AP                   += ap
            text                      = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
            rounded_prec              = ["%.2f" % elem for elem in prec]
            rounded_rec               = ["%.2f" % elem for elem in rec]
            ap_dictionary[class_name] = ap
            n_images                    = counter_images_per_class[class_name]
            lamr, mr, fppi              = self.log_average_miss_rate(np.array(prec), np.array(rec), n_images)
            lamr_dictionary[class_name] = lamr
            mAP  = sum_AP / n_classes
            text = "mAP = {0:.2f}%".format(mAP * 100)
        ap_dictionary = {f"{k} AP":v for k, v in ap_dictionary.items()}
        ap_dictionary["mAP"] = mAP
        return ap_dictionary