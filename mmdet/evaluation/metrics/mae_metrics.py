# %%writefile mae_metric.py
# Copyright (c) OpenMMLab. All rights reserved.
import json
from tqdm import tqdm
import os
import math
import wandb
import shutil
import pickle
import numpy as np
import pandas as pd
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from typing import Dict, Optional, Sequence

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
class MAEMetric(BaseMetric):
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
        confidence_thresholds (List[float]): The confidence thresholds for filtering
            predictions. Defaults to [0.01]
        iou_threshold (float): The IoU threshold for filtering predictions.
            Defaults to 0.5
    """
    default_prefix: Optional[str] = 'mae'

    def __init__(self, working_dir: str, collect_device: str = 'cpu',
                 prefix: Optional[str] = None, confidence_thresholds: list[float] = [0.01],
                iou_threshold: float = 0.5, run_id: Optional[str] = None, 
                run_project: Optional[str] = None, run_name: Optional[str] = None) -> None:
        
        if not os.path.exists(working_dir):
            raise FileNotFoundError(
                f"Working directory {working_dir} does not exist.")
        
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.working_dir = working_dir
        self.confidence_thresholds = confidence_thresholds
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
            scores = result['scores']
            
            for score in scores:
                data = dict()
                data['ID'] = image_id
                data['confidence'] = float(score)
                prediction_results.append(data)

        return pd.DataFrame(prediction_results)

    def gt2dataframe(self, gt_dicts: Sequence[dict]) -> pd.DataFrame:
        """Convert ground truth annotations to a pandas DataFrame.
        Args:
            gt_dicts (Sequence[dict]): The ground truth annotations.
        """
        annotations = []
        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            labels = gt_dict['anns']['labels'].numpy()   
            data = dict()
            data['ID'] = img_id
            data['Count'] = len(labels)
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
        df_gt = self.gt2dataframe(gts)
        df_pred = self.results2dataframe(preds)

        metrics = dict()
        metrics['length_ground_truth'] = len(df_gt['ID'].unique())
        metrics['length_predictions'] = len(df_pred['ID'].unique())
        for confidence in self.confidence_thresholds: 
            try:
                df_pred_confidence = df_pred[df_pred['confidence'] >= confidence].copy()
                df_pred_count = df_pred_confidence.groupby('ID').size().reset_index(name='Count')
                df_count = df_gt.merge(df_pred_count, on='ID', suffixes=('_true', '_predicted'), how='left')
                df_count['Count_predicted'] = df_count['Count_predicted'].fillna(0)
                mae = np.mean(abs(df_count['Count_predicted'] - df_count['Count_true']))
                metrics[f"mae@{confidence}"] = mae
            except ValueError as e:
                metrics[f"mae@{confidence}"] = np.nan
                print(f"Error calculating mae: {e}")
        if self.run_id is not None:
            self.wandb_logger.log(metrics, commit=False)
        temp_dir = f"{self.working_dir}/temp_files/"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return metrics