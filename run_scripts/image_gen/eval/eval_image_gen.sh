#!/usr/bin/env

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


image_dir=${base_data_dir}
data_dir=${base_data_dir}/ofa/image_gen_data

data=${data_dir}/coco_vqgan_full_test.tsv

# data=${data_dir}/coco_vqgan_dev.tsv


model_name=unival_image_gen_ofa_stage_2
path=/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/image_gen/unival_image_gen_ofa_stage_2/checkpoint_best1.pt


selected_cols=0,2,1
split='test'


VQGAN_MODEL_PATH=${base_log_dir}/ofa/pretrained_models/vqgan/last.ckpt
VQGAN_CONFIG_PATH=${base_log_dir}/ofa/pretrained_models/vqgan/model.yaml
CLIP_MODEL_PATH=${base_log_dir}/ofa/pretrained_models/clip/ViT-B-16.pt
GEN_IMAGES_PATH=/lus/scratch/NAT/gda2204/SHARED/tmp/results/image_gen/${model_name}
mkdir -p $GEN_IMAGES_PATH

echo "torch_home_path"
echo $TORCH_HOME
export TORCH_HOME=/home/mshukor/.cache/torch

python3 -m torch.distributed.launch \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --use_env ${ofa_dir}/evaluate.py \
    ${data} \
    --path=${path} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --task=image_gen \
    --batch-size=1 \
    --log-format=simple --log-interval=1 \
    --seed=42 \
    --gen-subset=${split} \
    --beam=16 \
    --min-len=1024 \
    --max-len-a=0 \
    --max-len-b=1024 \
    --sampling-topk=256 \
    --temperature=1.0 \
    --code-image-size=256 \
    --constraint-range=50265,58457 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"clip_model_path\":\"${CLIP_MODEL_PATH}\",\"vqgan_model_path\":\"${VQGAN_MODEL_PATH}\",\"vqgan_config_path\":\"${VQGAN_CONFIG_PATH}\",\"gen_images_path\":\"${GEN_IMAGES_PATH}\"}"


# compute IS
python image_gen/inception_score.py --gpu 4 --batch-size 128 --path1 ${GEN_IMAGES_PATH}/top1

# compute FID, download statistics for test dataset first.
python image_gen/fid_score.py --gpu 4 --batch-size 128 --path1 ${GEN_IMAGES_PATH}/top1 --path2 /lus/scratch/NAT/gda2204/SHARED/tmp/ofa/image_gen_data/gt_stat.npz

