import sys

sys.path.append('../../')
import torch
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks import ImageGenTask
from models.ofa import OFAModel
from PIL import Image
from torchvision import transforms
import time

# Register caption task
tasks.register_task('image_gen', ImageGenTask)
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = True if use_cuda else False

base_log_dir='/work/NAT/gda2204/mshukor/logs'

clip_model_path=f"{base_log_dir}/ofa/pretrained_models/clip/ViT-B-16.pt"
vqgan_model_path=f"{base_log_dir}/ofa/pretrained_models/vqgan/last.ckpt"
vqgan_config_path=f"{base_log_dir}/ofa/pretrained_models/vqgan/model.yaml"

video_model_path = f'{base_log_dir}/pretrained_models/3dcnn/resnext-101-kinetics.pth'
resnet_model_path = f'{base_log_dir}/pretrained_models/resnet101-5d3b4d8f.pth'



model_name='image_gen_ofaplus_stage_1_base_s2_long'
checkpoint_path = '/work/NAT/gda2204/mshukor/logs/ofa/checkpoints/image_gen/image_gen_ofaplus_stage_1_base_s2_long/50000_2000_1e-3/checkpoint_best.pt'
gen_images_path=f'/lus/scratch/NAT/gda2204/SHARED/tmp/results/test_{model_name}'


# Load pretrained ckpt & config
overrides = {"bpe_dir": "../../utils/BPE",
             "eval_cider": False,
             "beam": 16,
             "max_len_b": 1024,
             "min_len": 1024,
             "sampling_topk": 256,
             "constraint_range": "50265,58457",
             "clip_model_path": clip_model_path,
             "vqgan_model_path": vqgan_model_path,
             "vqgan_config_path": vqgan_config_path,
             "seed": 7,
            "video_model_path": video_model_path, 
             "resnet_model_path": resnet_model_path,
             "gen_images_path":gen_images_path,
             }


models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths(checkpoint_path),
    arg_overrides=overrides
)
task.cfg.sampling_times = 2
# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


# Construct input for image generation task
def construct_sample(query: str):
    code_mask = torch.tensor([True])
    src_text = encode_text(" what is the complete image? caption: {}".format(query), append_bos=True,
                           append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "code_masks": code_mask
        }
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


# Function for image generation
def image_generation(caption):
    sample = construct_sample(caption)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    print('|Start|', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), caption)
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)

    # return top-4 results (ranked by clip)
    images = [result[i]['image'] for i in range(4)]
    pic_size = 256
    retImage = Image.new('RGB', (pic_size * 2, pic_size * 2))
    print('|FINISHED|', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), caption)
    for i in range(4):
        loc = ((i % 2) * pic_size, int(i / 2) * pic_size)
        retImage.paste(images[i], loc)
    return retImage


# Waiting for user input
print('Please input your query.')
while True:
    query = input()
    retImage = image_generation(query)
    retImage.save(f'{query}.png')

