# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


from mapcalc import calculate_map, calculate_map_range

@dataclass
class AdjustLabelSmoothedCrossEntropySCSTCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    ignore_eos: bool = field(
        default=False,
        metadata={"help": "Ignore eos token"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    drop_worst_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio for discarding bad samples"},
    )
    drop_worst_after: int = field(
        default=0,
        metadata={"help": "steps for discarding bad samples"},
    )
    use_rdrop: bool = field(
        default=False, metadata={"help": "use R-Drop"}
    )
    reg_alpha: float = field(
        default=1.0, metadata={"help": "weight for R-Drop"}
    )
    sample_patch_num: int = field(
        default=196, metadata={"help": "sample patches for v1"}
    )
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )
    acc_thresh: Optional[float] = field(
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
    logprob: Optional[bool] = field(
        default=False, metadata={"help": "maximise log prob"}
    )

    pos_reward: Optional[float] = field(
        default=None, metadata={"help": "pos_reward"}
    )

    neg_reward: Optional[float] = field(
        default=None, metadata={"help": "neg_reward"}
    )

    reinforce: Optional[bool] = field(
        default=False, metadata={"help": "reinforce"}
    )

    lambda_reinforce: Optional[float] = field(
        default=0, metadata={"help": "lambda_reinforce"}
    )



def construct_rdrop_sample(x):
    if isinstance(x, dict):
        for key in x:
            x[key] = construct_rdrop_sample(x[key])
        return x
    elif isinstance(x, torch.Tensor):
        return x.repeat(2, *([1] * (x.dim()-1)))
    elif isinstance(x, int):
        return x * 2
    elif isinstance(x, np.ndarray):
        return x.repeat(2)
    else:
        raise NotImplementedError


def kl_loss(p, q):
    p_loss = F.kl_div(p, torch.exp(q), reduction='sum')
    q_loss = F.kl_div(q, torch.exp(p), reduction='sum')
    loss = (p_loss + q_loss) / 2
    return loss


def label_smoothed_nll_loss(
        lprobs, target, epsilon, update_num, reduce=True,
        drop_worst_ratio=0.0, drop_worst_after=0, use_rdrop=False, reg_alpha=1.0,
        constraint_masks=None, constraint_start=None, constraint_end=None
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    if constraint_masks is not None:
        smooth_loss = -lprobs.masked_fill(~constraint_masks, 0).sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (constraint_masks.sum(1) - 1 + 1e-6)
    elif constraint_start is not None and constraint_end is not None:
        constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
        smooth_loss = -lprobs[:, constraint_range].sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (len(constraint_range) - 1 + 1e-6)
    else:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    if drop_worst_ratio > 0 and update_num > drop_worst_after:
        if use_rdrop:
            true_batch_size = loss.size(0) // 2
            _, indices = torch.topk(loss[:true_batch_size], k=int(true_batch_size * (1 - drop_worst_ratio)), largest=False)
            loss = torch.cat([loss[indices], loss[indices+true_batch_size]])
            nll_loss = torch.cat([nll_loss[indices], nll_loss[indices+true_batch_size]])
            lprobs = torch.cat([lprobs[indices], lprobs[indices+true_batch_size]])
        else:
            loss, indices = torch.topk(loss, k=int(loss.shape[0] * (1 - drop_worst_ratio)), largest=False)
            nll_loss = nll_loss[indices]
            lprobs = lprobs[indices]

    ntokens = loss.numel()
    nll_loss = nll_loss.sum()
    # loss = loss.sum()
    if use_rdrop:
        true_batch_size = lprobs.size(0) // 2
        p = lprobs[:true_batch_size]
        q = lprobs[true_batch_size:]
        if constraint_start is not None and constraint_end is not None:
            constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
            p = p[:, constraint_range]
            q = q[:, constraint_range]
        loss = loss + ((kl_loss(p, q) * reg_alpha)/loss.shape[0])

    return loss, nll_loss, ntokens


@register_criterion(
    "adjust_label_smoothed_cross_entropy_scst", dataclass=AdjustLabelSmoothedCrossEntropySCSTCriterionConfig
)
class AdjustLabelSmoothedCrossEntropySCSTCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        ignore_eos=False,
        report_accuracy=False,
        drop_worst_ratio=0,
        drop_worst_after=0,
        use_rdrop=False,
        reg_alpha=1.0,
        sample_patch_num=196,
        constraint_range=None,
        acc_thresh=None,
        metric='acc',
        max_area_size=None,
        min_area_size=None,
        logprob=False,
        pos_reward=None,
        neg_reward=None,
        reinforce=False,
        lambda_reinforce=0,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.ignore_eos = ignore_eos
        self.report_accuracy = report_accuracy
        self.drop_worst_ratio = drop_worst_ratio
        self.drop_worst_after = drop_worst_after
        self.use_rdrop = use_rdrop
        self.reg_alpha = reg_alpha
        self.sample_patch_num = sample_patch_num

        

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

        self.acc_thresh = acc_thresh
        self.metric = metric
        self.min_area_size = min_area_size
        self.max_area_size = max_area_size
        self.logprob = logprob

        self.pos_reward = pos_reward
        self.neg_reward = neg_reward

        self.reinforce = reinforce
        self.lambda_reinforce = lambda_reinforce

    def get_generator_out(self, model, sample):

        model.eval()
        with torch.no_grad():
            self.task.scst_generator.model.eval()
            gen_out = self.task.scst_generator.generate([model], sample)

        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(gen_out[i][0]["tokens"][:-1] - len(self.task.src_dict) + self.task.cfg.num_bins)
            refs.append(sample["target"][i][:-1] - len(self.task.src_dict) + self.task.cfg.num_bins)

        return torch.stack(hyps, dim=0), torch.stack(refs, dim=0)
    
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
        
    def _calculate_ap_score(self, hyps, refs, thresh=0.5, min_area_size=None, max_area_size=None):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1]) ## x1, y1, x2, y2, x1 < x2
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        

        if max_area_size is not None and min_area_size is not None:
            ious =  ious * (torch.logical_or(area_targets < max_area_size, area_targets > min_area_size).float())

        elif min_area_size is not None:
            ious =  ious * (area_targets > min_area_size).float()

        elif max_area_size is not None:
            ious =  ious * (area_targets < max_area_size).float()

        if thresh is None:
            return ious
        else:
            return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def reward_step(self, sample, model):
        
        hyps, refs = self.get_generator_out(model, sample)
        hyps = hyps / (self.task.cfg.num_bins - 1) * self.task.cfg.max_image_size
        refs = refs / (self.task.cfg.num_bins - 1) * self.task.cfg.max_image_size
        hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
        hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)
        refs[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
        refs[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

        # scores = self._calculate_ap_score(hyps, refs)
        if self.metric == 'acc':
            scores = self._calculate_ap_score(hyps, sample['region_coords'].float(), thresh=self.acc_thresh, 
                                              min_area_size=self.min_area_size, max_area_size=self.max_area_size)
        elif self.metric == 'map':
            scores = self._calculate_map_score(hyps, sample['region_coords'].float(), thresh=self.acc_thresh)
        else:
            raise NotImplemented
        
        # logging_output["_score_sum"] = scores.sum().item()
        # logging_output["_score_cnt"] = scores.size(0)

        if self.pos_reward:
            scores = torch.where(scores > 0, self.pos_reward, scores)
        if self.neg_reward:
            scores = torch.where(scores == 0, self.neg_reward, scores)


        return scores
    
    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if isinstance(sample, list):
            if self.sample_patch_num > 0:
                sample[0]['net_input']['sample_patch_num'] = self.sample_patch_num
            # change to support len(samples) > 2
            loss_v1, sample_size_v1, logging_output_v1 = self.forward(model, sample[0], update_num, reduce)
            loss_v2, sample_size_v2, logging_output_v2 = self.forward(model, sample[1], update_num, reduce)
            loss = loss_v1 / sample_size_v1 + loss_v2 / sample_size_v2
            sample_size = 1
            logging_output = {
                "loss": loss.data,
                "loss_v1": loss_v1.data,
                "loss_v2": loss_v2.data,
                "nll_loss": logging_output_v1["nll_loss"].data / sample_size_v1 + logging_output_v2["nll_loss"].data / sample_size_v2,
                "ntokens": logging_output_v1["ntokens"] + logging_output_v2["ntokens"],
                "nsentences": logging_output_v1["nsentences"] + logging_output_v2["nsentences"],
                "sample_size": 1,
                "sample_size_v1": sample_size_v1,
                "sample_size_v2": sample_size_v2,
                "reward": (logging_output_v1["reward"] + logging_output_v2["reward"])/2,
            }
            return loss, sample_size, logging_output

        if self.use_rdrop:
            construct_rdrop_sample(sample)

        ### scst
        reward = self.reward_step(sample, model) # shape = bs
        model.train()
        net_output = model(**sample["net_input"])
        loss, nll_loss, ntokens = self.compute_loss(model, net_output, sample, update_num, reduce=reduce, reward=reward)


        
        
        # loss = loss*reward
        
        loss = loss.sum()
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else ntokens
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "reward": reward.mean(),
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, reward=None):
        conf = sample['conf'][:, None, None] if 'conf' in sample and sample['conf'] is not None else 1
        constraint_masks = None
        if "constraint_masks" in sample and sample["constraint_masks"] is not None:
            constraint_masks = sample["constraint_masks"]
            net_output[0].masked_fill_(~constraint_masks, -math.inf)
        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4:self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end:] = -math.inf
        lprobs = model.get_normalized_probs(net_output, log_probs=True) * conf
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[:, self.ignore_prefix_size :, :].contiguous()
        if self.ignore_eos:
            bsz, seq_len, embed_dim = lprobs.size()
            eos_indices = target.eq(self.task.tgt_dict.eos())
            lprobs = lprobs[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
            target = target[~eos_indices].reshape(bsz, seq_len-1)
            if constraint_masks is not None:
                constraint_masks = constraint_masks[~eos_indices].reshape(bsz, seq_len-1, embed_dim)
        if constraint_masks is not None:
            constraint_masks = constraint_masks.view(-1, constraint_masks.size(-1))
        
        if reward is not None:
            reward = reward.unsqueeze(1).unsqueeze(1)
            lprobs = lprobs*reward
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), constraint_masks

    def compute_loss(self, model, net_output, sample, update_num, reduce=True, reward=None):
        lprobs, target, constraint_masks = self.get_lprobs_and_target(model, net_output, sample, reward=reward)

        if constraint_masks is not None:
            constraint_masks = constraint_masks[target != self.padding_idx]
        # print(target.shape, self.padding_idx, lprobs.shape, target, lprobs)
        lprobs = lprobs[target != self.padding_idx]
        target = target[target != self.padding_idx]

        
        loss, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            update_num,
            reduce=reduce,
            drop_worst_ratio=self.drop_worst_ratio,
            drop_worst_after=self.drop_worst_after,
            use_rdrop=self.use_rdrop,
            reg_alpha=self.reg_alpha,
            constraint_masks=constraint_masks,
            constraint_start=self.constraint_start,
            constraint_end=self.constraint_end
        )
        
        if self.logprob and self.reinforce:
            # print(-lprobs.max(dim=-1)[0].squeeze(-1).sum(), loss)
            if self.lambda_reinforce > 0:
                lprobs_, target_, constraint_masks_ = self.get_lprobs_and_target(model, net_output, sample, reward=None)
                
                loss_, _, ntokens = label_smoothed_nll_loss(
                    lprobs_,
                    target_,
                    self.eps,
                    update_num,
                    reduce=reduce,
                    drop_worst_ratio=self.drop_worst_ratio,
                    drop_worst_after=self.drop_worst_after,
                    use_rdrop=self.use_rdrop,
                    reg_alpha=self.reg_alpha,
                    constraint_masks=constraint_masks_,
                    constraint_start=self.constraint_start,
                    constraint_end=self.constraint_end
                )
                # print(-lprobs.max(dim=-1)[0].squeeze(-1).sum(), loss_)
                # loss = -lprobs.max(dim=-1)[0].squeeze(-1).sum()*self.lambda_reinforce + loss_

                loss = loss*self.lambda_reinforce + loss_ # only supervised with more weights via reward

            else:
                loss = -lprobs.max(dim=-1)[0].squeeze(-1).sum()
            
        elif self.logprob:
            loss = nll_loss

        return loss, nll_loss, ntokens

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_sum_v1 = sum(log.get("loss_v1", 0) for log in logging_outputs)
        loss_sum_v2 = sum(log.get("loss_v2", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_v1 = sum(log.get("sample_size_v1", 0) for log in logging_outputs)
        sample_size_v2 = sum(log.get("sample_size_v2", 0) for log in logging_outputs)


        reward = sum(log.get("reward", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_v1", loss_sum_v1 / max(sample_size_v1, 1), max(sample_size_v1, 1), round=3
        )
        metrics.log_scalar(
            "loss_v2", loss_sum_v2 / max(sample_size_v2, 1), max(sample_size_v2, 1), round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v1", sample_size_v1, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v2", sample_size_v2, 1, round=3
        )

        metrics.log_scalar(
            "reward", reward / sample_size, sample_size, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
