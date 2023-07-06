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






ofa_dir=/lus/home/NAT/gda2204/mshukor/code/unival
base_data_dir=/lus/scratch/NAT/gda2204/SHARED/data
base_log_dir=/work/NAT/gda2204/mshukor/logs




bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module




selected_cols=0,4,2,3


image_encoder_name=resnet #vit_base_patch16_224





zero_shot=''
new_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs

patch_image_size=512
# sample_patch_num='--sample-patch-num=784' # ''
sample_patch_num=''

# exp_name=avg_rata_l0_7refcapsnlivqa
# new_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs
# path=/lus/scratch/NAT/gda2204/SHARED/logs/ofa/pretrained_models/average_models/avg_rata_l0_7refcapsnlivqa.pt


model_name=avg_postratafuse
path=/lus/scratch/NAT/gda2204/SHARED/logs/ofa/pretrained_models/average_models/avg_postratafuse.pt
zero_shot='--zero-shot'





acc_thresh='0.4,0.5,0.6,0.7,0.8,0.9'
metric=map
min_area_size=100000 # max 1000000
max_area_size=30000

echo ${path}
result_path=${new_base_log_dir}/ofa/results/refcocoplus/${exp_name}
mkdir ${result_path}




data=${base_data_dir}/ofa/refcocoplus_data/refcocoplus_val.tsv
split='refcocoplus_val'

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
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}" \
    --acc-thresh=${acc_thresh} \
    --metric=${metric} \
    ${zero_shot} \
    --min-area-size=${min_area_size} \
    --max-area-size=${max_area_size} \
    --patch-image-size=${patch_image_size} \
    ${sample_patch_num}

data=${base_data_dir}/ofa/refcocoplus_data/refcocoplus_testA.tsv
split='refcocoplus_testA'
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
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}" \
    --acc-thresh=${acc_thresh} \
    --metric=${metric} \
    ${zero_shot} \
    --min-area-size=${min_area_size} \
    --max-area-size=${max_area_size} \
    --patch-image-size=${patch_image_size} \
    ${sample_patch_num}


data=${base_data_dir}/ofa/refcocoplus_data/refcocoplus_testB.tsv
split='refcocoplus_testB'

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
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}" \
    --acc-thresh=${acc_thresh} \
    --metric=${metric} \
    ${zero_shot} \
    --min-area-size=${min_area_size} \
    --max-area-size=${max_area_size} \
    --patch-image-size=${patch_image_size} \
    ${sample_patch_num}
