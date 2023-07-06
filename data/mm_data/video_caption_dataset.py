# Modified from OFA code.
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.


import logging
import warnings
import string

import numpy as np
import torch
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

from data.video_utils import VIDEO_READER_FUNCS

import os 
import random 

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    patch_videos = torch.stack([sample['patch_video'] for sample in samples], dim=0)
    patch_types = torch.cat([sample['patch_type'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens,
            "patch_videos": patch_videos,
            "patch_types": patch_types,
        },
        "target": target,
    }

    return batch


class CaptionDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        patch_image_size=224,
        imagenet_default_mean_and_std=False,
        scst=False,
        image_dir='/gpfsscratch/rech/dyf/ugz83ue/data', 
        patch_frame_size=224,
        num_frames=4,
        sample_type='rand',
        use_dataaug=False,
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.scst = scst

        self.image_dir = image_dir

        self.transtab = str.maketrans({key: None for key in string.punctuation})

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]


 
        self.split = split
        type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
        if self.split != 'train' or not use_dataaug:
            self.patch_video_resize_transform = transforms.Compose([
                    transforms.CenterCrop(patch_frame_size),
                    type_transform, 
                    transforms.Normalize(mean=mean, std=std),
                ])
            logger.info("val split, do not use random augmentation.")
        else:
            aug_transform = transforms.RandAugment()
            self.patch_video_resize_transform = transforms.Compose(
                [
                    aug_transform,
                    transforms.RandomResizedCrop(
                        patch_frame_size,
                        scale=(0.5, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    type_transform,
                    transforms.Normalize(mean=mean, std=std),
                ]
            )


            logger.info("train split, use random augmentation.")


        # video
        self.num_frames =  num_frames
        self.sample_type = sample_type
        self.video_reader = VIDEO_READER_FUNCS['decord'] 
        self.max_num_frames = num_frames
        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = " what does the video describe?"
        else:
            raise NotImplemented

        self.num_tries = 4

    def __getitem__(self, index, tries=0, other_dataset=None):
        uniq_id, image, caption = self.dataset[index]


        # video 
        image_path = os.path.join(self.image_dir, image)
        data_path = image_path

        max_num_frames = self.max_num_frames 


        try:

            frames, frame_indices, video_duration = self.video_reader(
                data_path, self.num_frames, self.sample_type, max_num_frames=max_num_frames
            )


        except Exception as e:
            new_index = random.randint(0, len(self) - 1)
            logger.warning(
                f"Caught exception {e} when loading video {data_path}, "
                f"randomly sample a new video as replacement"
            )
            if tries < self.num_tries:
                return self.__getitem__(new_index, tries=tries+1, other_dataset=other_dataset)
            else:
                print("Videos are too corrupted, try increase the num_tries")
                raise 



        

        patch_video = self.patch_video_resize_transform(frames)
        patch_video = patch_video.permute(1, 0, 2, 3) # -> (C, T, h, w)


        patch_image = torch.zeros((3, self.patch_image_size, self.patch_image_size))
        patch_type = torch.tensor([1])




        patch_mask = torch.tensor([True])

        if self.split == 'train' and not self.scst:
            caption = caption.translate(self.transtab).strip()
            caption_token_list = caption.strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        else:
            caption = ' '.join(caption.strip().split())
            caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
            tgt_caption = '&&'.join(caption_list)
        src_item = self.encode_text(self.prompt)
        tgt_item = self.encode_text(" {}".format(tgt_caption))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "patch_type": patch_type,
            "patch_video": patch_video,
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
