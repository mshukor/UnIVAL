# Datasets

Below you can download the datasets used during pretraining and finetunning. We only provide the tsv files for these datasets.
Note that each datasets has its specific LICENSE that you should refer to.

## Pretraining


 The pretraining datasets used in UnIVAL are all publicly available. Here we provide the public links to these data, it is recommended that you download the data from the links first, and then process the downloaded dataset into a similar format as the examples we provided.
-   _CC12M_:  https://github.com/google-research-datasets/conceptual-12m
-   _CC3M_: https://github.com/google-research-datasets/conceptual-captions
-   _SBU_: https://www.cs.virginia.edu/~vicente/sbucaptions
-   _COCO_: https://cocodataset.org/#home
-   _VG_: https://visualgenome.org/
-   _VQAv2_: https://visualqa.org/
- _GQA_: https://cs.stanford.edu/people/dorarad/gqa/about.html
- _RefCOCO_/_RefCOCO+_/RefCOCOg: https://github.com/lichengunc/refer

The following are the different tsv files used during pretraining:

* <a href="https://data.isir.upmc.fr/unival/data/pretrain/vision_language_ground.tsv"> Datasets for Grounding </a>
* <a href="https://data.isir.upmc.fr/unival/data/pretrain/vision_language_caption.tsv"> Datasets for Image Captioning </a>
* <a href="https://data.isir.upmc.fr/unival/data/pretrain/cc12m.tsv"> Datasets for CC12M </a>
* <a href="https://data.isir.upmc.fr/unival/data/pretrain/vision_language_qa.tsv"> Datasets for Image VQA </a>
* <a href="https://data.isir.upmc.fr/unival/data/pretrain/vision_language_mini_vqa_ground.tsv"> Dataset for Image Captioning + VQA + Grounding combined (used in stage 2) </a>
* <a href="https://data.isir.upmc.fr/unival/data/pretrain/video_mini_webvid2mccapqa.tsv"> Dataset for Video-Text </a>
* <a href="https://data.isir.upmc.fr/unival/data/pretrain/negative_sample/"> Negative samples </a>


## Image-Text Tasks
For image-text tasks, we use the same data as in OFA, please refer to their [repo](https://github.com/OFA-Sys/OFA/blob/main/datasets.md#vision--language-tasks) for download.

We provide the following additional datasets for zero-shot evaluation:
* <a href="https://data.isir.upmc.fr/unival/data/vqa_data/"> Dataset for VQA on VizWiz and OKVQA </a>
* <a href="https://data.isir.upmc.fr/unival/data/caption_data/"> Dataset for Caption on Nocaps </a>

## Video-Text Tasks

* <a href="https://data.isir.upmc.fr/unival/data/video_data/caption_data/"> Dataset for Video Caption on MSRVTT and Activitynet-Captions </a>
* <a href="https://data.isir.upmc.fr/unival/data/video_data/vqa_data/"> Dataset for VideoQA on MSRVTT-QA and MSVD-QA  </a>

## Audio-Text Tasks

* <a href="https://data.isir.upmc.fr/unival/data/audio_data/caption_data/"> Dataset for Audio Caption on Audiocaps and Clotho v1 </a>
