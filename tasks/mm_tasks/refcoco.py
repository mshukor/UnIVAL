# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace

import torch
from fairseq import metrics
from fairseq.tasks import register_task

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.refcoco_dataset import RefcocoDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


from mapcalc import calculate_map, calculate_map_range
from functools import partial

@dataclass
class RefcocoConfig(OFAConfig):
    eval_acc: bool = field(
        default=False, metadata={"help": "evaluation with accuracy"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )


    acc_thresh: Optional[str] = field(
        default=None, metadata={"help": "acc thresh for refcoco"}
    )
    metric: Optional[str] = field(
        default='acc',
        metadata={"help": "metric"}
    )

    max_area_size: Optional[float] = field(
        default=None, metadata={"help": "max_area_size"}
    )

    min_area_size: Optional[float] = field(
        default=None, metadata={"help": "min_area_size"}
    )

@register_task("refcoco", dataclass=RefcocoConfig)
class RefcocoTask(OFATask):
    def __init__(self, cfg: RefcocoConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        
        self.metric = cfg.metric
        self.min_area_size = cfg.min_area_size
        self.max_area_size = cfg.max_area_size
        try:
            self.acc_thresh = float(cfg.acc_thresh)
        except:
            self.acc_thresh = cfg.acc_thresh

        print(self.acc_thresh, self.metric, self.min_area_size, self.max_area_size)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = RefcocoDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            num_bins=self.cfg.num_bins,
            max_image_size=self.cfg.max_image_size
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_acc:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        if self.cfg.scst:
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator(
                [model], Namespace(**scst_args)
            )

        return model

    def _calculate_ap_score(self, hyps, refs, thresh=0.5, min_area_size=None, max_area_size=None):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)

        if max_area_size is not None and min_area_size is not None:
            ious =  ious * ((area_targets < max_area_size).float() + (area_targets > min_area_size).float())/2
        elif min_area_size is not None:
            ious =  ious * (area_targets > min_area_size).float()

        elif max_area_size is not None:
            ious =  ious * (area_targets < max_area_size).float()

        if thresh is None:
            return ious
        else:
            return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def _calculate_map_score(self, hyps, refs, thresh=0.5):
        
        
        ground_truth = {
            'boxes': refs.cpu().numpy().tolist(),

            'labels': [1 for i in range(refs.shape[0])]
            }

        result_dict = {
            'boxes': hyps.cpu().numpy().tolist(),

            'labels': [1 for i in range(hyps.shape[0])], 
            }

        score = calculate_map(ground_truth, result_dict, thresh)

        score = torch.tensor(score).unsqueeze(0).repeat(refs.shape[0]).to(hyps.device)
        return score

        
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        if self.cfg.eval_acc:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            hyps = hyps / (self.cfg.num_bins - 1) * self.cfg.max_image_size
            refs = refs / (self.cfg.num_bins - 1) * self.cfg.max_image_size
            hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
            hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)
            refs[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
            refs[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

            # scores = self._calculate_ap_score(hyps, refs)
            # scores = self._calculate_ap_score(hyps, sample['region_coords'].float())

            # scores = self._calculate_ap_score(hyps, refs)
            scores = self._calculate_ap_score(hyps, sample['region_coords'].float(), thresh=self.acc_thresh)
            if self.min_area_size is not None:
                large_scores = self._calculate_ap_score(hyps, sample['region_coords'].float(), thresh=self.acc_thresh, 
                                min_area_size=self.min_area_size)
                logging_output["_large_score_sum"] = large_scores.sum().item()
                logging_output["_large_score_cnt"] = large_scores.size(0)

            if self.max_area_size is not None:
                small_scores = self._calculate_ap_score(hyps, sample['region_coords'].float(), thresh=self.acc_thresh, 
                                max_area_size=self.max_area_size)
                logging_output["_small_score_sum"] = small_scores.sum().item()
                logging_output["_small_score_cnt"] = small_scores.size(0)

            if self.metric == 'map':
                map_scores = self._calculate_map_score(hyps, sample['region_coords'].float(), thresh=self.acc_thresh)
                logging_output["_map_score_sum"] = map_scores.sum().item()
                logging_output["_map_score_cnt"] = map_scores.size(0)

            logging_output["_score_sum"] = scores.sum().item()
            logging_output["_score_cnt"] = scores.size(0)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters, prefix='_score'):
            score = meters[prefix+"_sum"].sum / meters[prefix+"_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("score", compute_score)
            if self.metric == 'map':
                metrics.log_scalar("_map_score_sum", sum_logs("_map_score_sum"))
                metrics.log_scalar("_map_score_cnt", sum_logs("_map_score_cnt"))
                metrics.log_derived("score", partial(compute_score, prefix='_map_score'))

            if self.min_area_size is not None:
                metrics.log_scalar("_large_score_sum", sum_logs("_large_score_sum"))
                metrics.log_scalar("_large_score_cnt", sum_logs("_large_score_cnt"))
                metrics.log_derived("score", partial(compute_score, prefix='_large_score'))
            if self.max_area_size is not None:
                metrics.log_scalar("_small_score_sum", sum_logs("_small_score_sum"))
                metrics.log_scalar("_small_score_cnt", sum_logs("_small_score_cnt"))
                metrics.log_derived("score", partial(compute_score, prefix='_small_score'))

    def _inference(self, generator, sample, model):
        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(gen_out[i][0]["tokens"][:-1] - len(self.src_dict) + self.cfg.num_bins)
            refs.append(sample["target"][i][:-1] - len(self.src_dict) + self.cfg.num_bins)
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: ", hyps[0])
            logger.info("example reference: ", refs[0])

        return torch.stack(hyps, dim=0), torch.stack(refs, dim=0)
