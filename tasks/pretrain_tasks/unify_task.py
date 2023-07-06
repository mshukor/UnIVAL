# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import os
import math
from typing import Optional
from fairseq.tasks import register_task
from fairseq.data import FairseqDataset, iterators

from tasks.ofa_task import OFATask, OFAConfig
from data.pretrain_data.unify_dataset import UnifyDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)

 
@dataclass
class UnifyConfig(OFAConfig):
    max_image_size: int = field(
        default=512, metadata={"help": ""}
    )
    neg_sample_dir: Optional[str] = field(
        default=None,
        metadata={"help": "negative sample directory, which contains captions (taken from all image-text pairs), "
                          "answers (taken from VQA), "
                          "objects (taken form OpenImages) "},
    )
    code_image_size: int = field(
        default=128, metadata={"help": "the resolution of the generated image in the image infilling task"}
    )

    pretrain_seed: int = field(
        default=7,
        metadata={"help": "pretrain seed"},
    )

    mask_ratio: float = field(
        default=0.3,
        metadata={"help": "fraction of words/subwords that will be masked"},
    )
    random_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], use random token this often"},
    )
    keep_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], keep original token this often"},
    )
    mask_length: str = field(
        default="span-poisson",
        metadata={"help": "mask length to choose ['subword', 'word', 'span-poisson']"},
    )
    poisson_lambda: float = field(
        default=3.0,
        metadata={"help": "randomly shuffle sentences for this proportion of inputs"},
    )
    replace_length: int = field(
        default=1,
        metadata={"help": "when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)"},
    )


    neg_captions_video: Optional[str] = field(
        default=None,
        metadata={"help": "negative sample file .txt, which contains captions (taken from all video-text pairs), "},
    )

    neg_captions_audio: Optional[str] = field(
        default=None,
        metadata={"help": "negative sample file .txt, which contains captions (taken from all video-text pairs), "},
    )

    audio_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "detection data selected cols"},
    )

    audio_data: Optional[str] = field(
        default=None,
        metadata={"help": "detection data"},
    )
    audio_cnt: int = field(
        default=2,
        metadata={"help": "to control the ratio of video examples in the batch"},
    )

    video_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "video data selected cols"},
    )

    video_data: Optional[str] = field(
        default=None,
        metadata={"help": "video data"},
    )

    video_cnt: int = field(
        default=2,
        metadata={"help": "to control the ratio of video examples in the batch"},
    )

    image_text_data: Optional[str] = field(
        default=None,
        metadata={"help": "cc12m data"},
    )

    image_text_cnt: int = field(
        default=2,
        metadata={"help": "to control the ratio of image-text examples in the batch"},
    )

    other_data_cnt: int = field(
        default=8,
        metadata={"help": "to control the ratio of image-text examples in the batch"},
    )


    init_image_text_data: Optional[str] = field(
        default=None,
        metadata={"help": "init data"},
    )

    init_dataset_epoch: int = field(
        default=3,
        metadata={"help": "to witch from the init dataset"},
    )

    init_text_data: Optional[str] = field(
        default=None,
        metadata={"help": "init data"},
    )


    image_text_vqa_data: Optional[str] = field(
        default=None,
        metadata={"help": "vqa data"},
    )

    image_text_vqa_cnt: int = field(
        default=2,
        metadata={"help": "to control the ratio of image-text examples in the batch"},
    )


    image_text_ground_data: Optional[str] = field(
        default=None,
        metadata={"help": "cc12m data"},
    )

    image_text_ground_cnt: int = field(
        default=2,
        metadata={"help": "to control the ratio of image-text examples in the batch"},
    )

    only_video_data: Optional[str] = field(
        default=None,
        metadata={"help": "only video data"},
    )
    only_audio_data: Optional[str] = field(
        default=None,
        metadata={"help": "only video data"},
    )

    video_text_data: Optional[str] = field(
        default=None,
        metadata={"help": "cc12m data"},
    )

    video_text_cnt: int = field(
        default=2,
        metadata={"help": "to control the ratio of image-text examples in the batch"},
    )

    audio_text_data: Optional[str] = field(
        default=None,
        metadata={"help": "cc12m data"},
    )

    audio_text_cnt: int = field(
        default=2,
        metadata={"help": "to control the ratio of image-text examples in the batch"},
    )

    audio_with_video: bool = field(
        default=False,
        metadata={"help": "audio_with_video"},
    )


@register_task("unify_task", dataclass=UnifyConfig)
class UnifyTask(OFATask):
    def __init__(self, cfg: UnifyConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.type2ans_dict = json.load(open(os.path.join(self.cfg.neg_sample_dir, 'type2ans.json')))
        self.ans2type_dict = {}
        for type, answer_list in self.type2ans_dict.items():
            if type == 'other':
                continue
            for answer in answer_list:
                self.ans2type_dict[answer] = type

        self.all_object_list = [
            row.strip() for row in open(os.path.join(self.cfg.neg_sample_dir, 'object.txt')) if row.strip() != ''
        ]
        neg_captions_path = os.path.join(self.cfg.neg_sample_dir, 'all_captions.txt')
        self.all_caption_list = [
            row.strip() for row in open(os.path.join(neg_captions_path)) if row.strip() != ''
        ]

        if self.cfg.neg_captions_video is not None:
            neg_captions_video = self.cfg.neg_captions_video
        else:
            neg_captions_video = neg_captions_path

        if self.cfg.neg_captions_audio is not None:
            neg_captions_audio = self.cfg.neg_captions_audio
        else:
            neg_captions_audio = neg_captions_path

        print("Reading negative video captions:", neg_captions_video)
        self.all_caption_video_list = [
            row.strip() for row in open(os.path.join(neg_captions_video)) if row.strip() != ''
        ]
        print("Reading negative audio captions:", neg_captions_audio)
        self.all_caption_audio_list = [
            row.strip() for row in open(os.path.join(neg_captions_audio)) if row.strip() != ''
        ]



        # audio 
        self.audio_dataset = None 
        self.video_dataset = None
        self.image_text_dataset = None 

        self.video_text_dataset = None

        self.audio_text_dataset = None

        self.init_image_text_dataset = None

        self.init_text_dataset = None

        self.image_text_ground_dataset = None 
        self.image_text_vqa_dataset = None 
        


        if self.cfg.audio_data is not None:
            print("Load audio data")
            self.audio_dataset = FileDataset(self.cfg.audio_data, self.cfg.audio_selected_cols)

        if self.cfg.video_data is not None:
            print("Load video data")
            self.video_dataset = FileDataset(self.cfg.video_data, self.cfg.video_selected_cols)

        if self.cfg.image_text_data is not None:
            print("Load Image Text data")
            self.image_text_dataset = FileDataset(self.cfg.image_text_data, self.cfg.selected_cols)

        if self.cfg.init_image_text_data is not None:
            print("Load Init Image Text data")
            self.init_image_text_dataset = FileDataset(self.cfg.init_image_text_data, self.cfg.selected_cols)

        if self.cfg.init_text_data is not None:
            print("Load Init Text data")
            self.init_text_dataset = FileDataset(self.cfg.init_text_data, self.cfg.text_selected_cols)

        if self.cfg.image_text_vqa_data is not None:
            print("Load Image Text data")
            self.image_text_vqa_dataset = FileDataset(self.cfg.image_text_vqa_data, self.cfg.selected_cols)

        if self.cfg.image_text_ground_data is not None:
            print("Load Image Text data")
            self.image_text_ground_dataset = FileDataset(self.cfg.image_text_ground_data, self.cfg.selected_cols)

        
        if self.cfg.video_text_data is not None:
            print("Load Video Text data")
            self.video_text_dataset = FileDataset(self.cfg.video_text_data, self.cfg.selected_cols)

        if self.cfg.audio_text_data is not None:
            print("Load Video Text data")
            self.audio_text_dataset = FileDataset(self.cfg.audio_text_data, self.cfg.selected_cols)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        file_path = paths[(epoch - 1) % (len(paths))]
        dataset = FileDataset(file_path, self.cfg.selected_cols)



        self.datasets[split] = UnifyDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            seed=self.cfg.pretrain_seed,
            code_dict_size=self.cfg.code_dict_size,
            num_bins=self.cfg.num_bins,
            patch_image_size=self.cfg.patch_image_size,
            code_image_size=self.cfg.code_image_size,
            all_object_list=self.all_object_list,
            all_caption_list=self.all_caption_list,
            type2ans_dict=self.type2ans_dict,
            ans2type_dict=self.ans2type_dict,
            max_image_size=self.cfg.max_image_size,
            mask_ratio=self.cfg.mask_ratio,
            random_ratio=self.cfg.random_ratio,
            keep_ratio=self.cfg.keep_ratio,
            mask_length=self.cfg.mask_length,
            poisson_lambda=self.cfg.poisson_lambda,
            replace_length=self.cfg.replace_length,
            read_from_img_path=self.cfg.read_from_img_path,
            image_dir=self.cfg.image_dir,
            no_image_transform=self.cfg.no_image_transform,
            patch_frame_size=self.cfg.patch_frame_size,
            num_frames=self.cfg.num_frames,
            all_caption_video_list=self.all_caption_video_list,
            all_caption_audio_list=self.all_caption_audio_list,
            max_audio_len=self.cfg.max_audio_len,
            audio_dataset=self.audio_dataset,
            video_dataset=self.video_dataset,
            video_cnt=self.cfg.video_cnt,
            audio_cnt=self.cfg.audio_cnt,
            image_text_dataset=self.image_text_dataset,
            image_text_cnt=self.cfg.image_text_cnt,
            other_data_cnt=self.cfg.other_data_cnt,
            init_image_text_dataset=self.init_image_text_dataset,
            init_dataset_epoch=self.cfg.init_dataset_epoch,
            init_text_dataset=self.init_text_dataset,
            image_text_vqa_dataset=self.image_text_vqa_dataset,
            image_text_vqa_cnt=self.cfg.image_text_vqa_cnt,
            image_text_ground_dataset=self.image_text_ground_dataset,
            image_text_ground_cnt=self.cfg.image_text_ground_cnt,
            only_video_data=self.cfg.only_video_data,
            only_audio_data=self.cfg.only_audio_data,
            video_text_dataset=self.video_text_dataset,
            video_text_cnt=self.cfg.video_text_cnt,
            audio_text_dataset=self.audio_text_dataset,
            audio_text_cnt=self.cfg.audio_text_cnt,
            audio_with_video=self.cfg.audio_with_video,
            sample_rate=self.cfg.sample_rate,
        )

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # create mini-batches with given size constraints
        batch_sampler = [
            [j for j in range(i, min(i + max_sentences, len(dataset)))]
            for i in range(0, len(dataset), max_sentences)
        ]
        total_row_count = dataset.dataset.get_total_row_count()
        num_batches = math.ceil(math.ceil(total_row_count / num_shards) / max_sentences)
        if len(batch_sampler) < num_batches:
            batch_sampler.append([1])

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=1,
            shard_id=0,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size
        )

        return epoch_iter
