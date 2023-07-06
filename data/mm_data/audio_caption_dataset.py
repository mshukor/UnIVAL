# Modified from OFA code.
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import string

import numpy as np
import torch

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset


import os 
import random 

import soundfile as sf

import torchaudio


from data.audio_utils import get_audio_features, int16_to_float32, float32_to_int16, AUDIO_CFG

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
    patch_audios = torch.stack([sample['patch_audio'] for sample in samples], dim=0)

    

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
            "patch_audios": patch_audios,
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
        audio_cfg=AUDIO_CFG,
        max_audio_len = 480000,
        num_frames=4,
        sample_rate = 48000,
        audio_sample_rate=False,
        ast=False,
        mode='train',
        mel_bins=64,

    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.scst = scst

        self.image_dir = image_dir

        self.sample_rate = sample_rate

        self.transtab = str.maketrans({key: None for key in string.punctuation})


        # video
        self.num_frames = num_frames

        # audio 
        self.audio_cfg = audio_cfg
        self.max_audio_len = max_audio_len

        self.audio_sample_rate = audio_sample_rate



        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = " what does the video describe?"
        else:
            raise NotImplemented

        # for AST encoder
        self.ast = ast
        self.target_length = 1024 # 1024
        self.mode = split # train
        self.freqm_p = 24
        self.timem_p = 96
        self.skip_norm = False
        self.noise = False
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974
        self.freqm = torchaudio.transforms.FrequencyMasking(self.freqm_p)
        self.timem = torchaudio.transforms.TimeMasking(self.timem_p)
        self.mel_bins = mel_bins

    def __getitem__(self, index):
        uniq_id, image, caption = self.dataset[index]


        # audio 
        image_path = os.path.join(self.image_dir, image)
        data_path = image_path


        try:

            # load the waveform of the shape (T,), should resample to 48000
            if not self.audio_sample_rate:
                audio_data, orig_sr = sf.read(data_path) # no sample rate
                if audio_data.ndim>1:
                    audio_data = np.mean(audio_data,axis=1)
                audio_data = int16_to_float32(float32_to_int16(audio_data))
                audio_data = torch.tensor(audio_data).float() # (T)
            else:
                audio_data, orig_sr = torchaudio.load(data_path)
                audio_data = torchaudio.transforms.Resample(orig_sr, self.sample_rate)(audio_data[0])
 
            sample = {}

            sample = get_audio_features(
                sample, audio_data, self.max_audio_len, 
                data_truncating='rand_trunc', 
                data_filling='repeatpad',
                audio_cfg=self.audio_cfg
            )
        except Exception as e:
            new_index = random.randint(0, len(self) - 1)
            logger.warning(
                f"Caught exception {e} when loading video {data_path}, "
                f"randomly sample a new video as replacement"
            )
            return self.__getitem__(new_index)

        waveform = sample['waveform']
        patch_audio = waveform


        patch_type = torch.tensor([2])

        patch_image = torch.zeros((3, self.patch_image_size, self.patch_image_size))
        patch_video = torch.zeros((3, self.num_frames, self.patch_image_size, self.patch_image_size))


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
            "patch_audio": patch_audio,
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
