#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
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





ofa_dir=/lus/home/NAT/gda2204/mshukor/code/unival
base_data_dir=/lus/scratch/NAT/gda2204/SHARED/data
base_log_dir=/work/NAT/gda2204/mshukor/logs




bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module


data_dir=${base_data_dir}/ofa/video_data/vqa_data

# val or test or fullval
split=test
read_from_img_path=True
image_dir=${base_data_dir}

data=${data_dir}/msvd_qa_1k_test.tsv

ans2label_file=${base_data_dir}/ofa/video_data/vqa_data/msvd_trainval_1k_ans2label.pkl


selected_cols=0,5,2,3,4
valid_batch_size=40

eval_ema='--ema-eval'
zero_shot=''
new_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs




# model_name=unival_s2_hs
# path=/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/pretrain/unival_s2_hs/checkpoint1.pt
# eval_ema=''
# zero_shot='--zero-shot'


model_name=unival_video_vqa_msvd
path='/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/vqa/unival_video_vqa_msvd/checkpoint_best.pt'


echo ${path}
result_path=${new_base_log_dir}/ofa/results/vqa/eval_msvd_${exp_name}_${split}
mkdir ${result_path}

num_frames=8
patch_frame_size=384

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
    --task=video_vqa_gen \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --beam-search-vqa-eval \
    --beam=5 \
    --temperature=1.0 \
    --unnormalized \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\",\"valid_batch_size\":\"${valid_batch_size}\"}" \
    --image-dir=${image_dir} \
    --read-from-img-path \
    ${zero_shot} \
    --prompt-type='none' \
    --patch-frame-size=${patch_frame_size} \
    --num-frames=${num_frames} \
    ${eval_ema}


    # --ema-eval \


