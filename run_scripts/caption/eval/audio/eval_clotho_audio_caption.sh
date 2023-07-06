#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
# Number of GPUs per GPU worker
export GPUS_PER_NODE=8
# Number of GPU workers, for single-worker training, please set to 1
export NUM_NODES=$SLURM_NNODES
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

# The port for communication
export MASTER_PORT=12350
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=$SLURM_NODEID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "RANK :$RANK"
echo "NUM_NODES :$NUM_NODES"
echo "GPUS_PER_NODE :$GPUS_PER_NODE"

export MIOPEN_USER_DB_PATH=/lus/home/NAT/gda2204/mshukor/.config/miopen_${MASTER_ADDR}_${SLURM_PROCID}/

echo "MIOPEN_USER_DB_PATH :$MIOPEN_USER_DB_PATH"

num_workers=0


exp_name=eval_clotho_audio_caption



ofa_dir=/lus/home/NAT/gda2204/mshukor/code/unival
base_data_dir=/lus/scratch/NAT/gda2204/SHARED/data
base_log_dir=/work/NAT/gda2204/mshukor/logs




bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module



task=video_caption
arch=unival_base
pretrained_model=  


data_dir=${base_data_dir}/ofa/audio_data/caption_data
split=test
data=${data_dir}/clotho_evaluation.tsv
image_dir=${base_data_dir}


eval_cider_cached=${data_dir}/cider_cached_tokens/clotho-evaluation-words.p


new_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs
model_name=unival_audio_caption_clotho
path=${new_base_log_dir}/ofa/checkpoints/caption/${model_name}/checkpoint_best.pt

result_path=${new_base_log_dir}/ofa/results/caption/${model_name}_${split}
mkdir ${result_path}

selected_cols=1,4,2



###
image_encoder_name=timm_resnet #vit_base_patch16_224
patch_image_size=480
resnet_type=resnet101

resnet_model_path=${base_log_dir}/pretrained_models/resnet101-5d3b4d8f.pth

# video
video_encoder_name=all_resnext101
patch_frame_size=384
video_model_path=${base_log_dir}/pretrained_models/3dcnn/resnext-101-kinetics.pth #${base_log_dir}/pretrained_models/TimeSformer_divST_8x32_224_K600.pyth
num_frames=16



python3 -m torch.distributed.launch \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --use_env ${ofa_dir}/evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=${task} \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}" 


python ${ofa_dir}/run_scripts/caption/coco_eval.py ${result_path}/${split}_predict.json ${data_dir}/clotho_evaluation_caption_coco_format.json
