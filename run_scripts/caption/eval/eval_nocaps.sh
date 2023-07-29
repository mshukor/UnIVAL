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


exp_name=eval_nocaps



ofa_dir=/lus/home/NAT/gda2204/mshukor/code/unival
base_data_dir=/lus/scratch/NAT/gda2204/SHARED/data
base_log_dir=/work/NAT/gda2204/mshukor/logs




bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module


data_dir=${base_data_dir}/ofa/caption_data
split=val # val  

zero_shot=''

read_from_img_path='--read-from-img-path' #'--read-from-img-path' # ''



new_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs



model_name=avg_postratafusevanilla
path=/lus/scratch/NAT/gda2204/SHARED/logs/ofa/pretrained_models/average_models/avg_postratafusevanilla.pt
zero_shot='--zero-shot'


result_path=${new_base_log_dir}/ofa/results/caption/eval_${model_name}_${split}
mkdir ${result_path}

selected_cols=1,4,2


image_encoder_name=timm_resnet #vit_base_patch16_224 timm_resnet resnet
resnet_type=resnet101



data=${data_dir}/nocaps_${split}.tsv # caption_val caption_test

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
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=22 \
    --unnormalized \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --patch-image-size=480 \
    ${zero_shot} \
    ${read_from_img_path} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"


python ${ofa_dir}/run_scripts/caption/coco_eval.py ${result_path}/${split}_predict.json ${data_dir}/nocaps_val_caption_coco_format.json




echo "In Domain Eval"
data=${data_dir}/nocaps_indomain_${split}.tsv # caption_val caption_test

result_path=${new_base_log_dir}/ofa/results/caption/eval_nocaps_indomain_${model_name}_${split}


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
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=22 \
    --unnormalized \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --patch-image-size=480 \
    ${zero_shot} \
    ${read_from_img_path} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"


python ${ofa_dir}/run_scripts/caption/coco_eval.py ${result_path}/${split}_predict.json ${data_dir}/nocaps_val_caption_coco_format.json



echo "Near Domain Eval"
data=${data_dir}/nocaps_neardomain_${split}.tsv # caption_val caption_test

result_path=${new_base_log_dir}/ofa/results/caption/eval_nocaps_neardomain_${model_name}_${split}


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
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=22 \
    --unnormalized \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --patch-image-size=480 \
    ${zero_shot} \
    ${read_from_img_path} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"


python ${ofa_dir}/run_scripts/caption/coco_eval.py ${result_path}/${split}_predict.json ${data_dir}/nocaps_val_caption_coco_format.json



echo "Out Domain Eval"
data=${data_dir}/nocaps_outdomain_${split}.tsv # caption_val caption_test

result_path=${new_base_log_dir}/ofa/results/caption/eval_nocaps_outdomain_${model_name}_${split}


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
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=22 \
    --unnormalized \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --patch-image-size=480 \
    ${zero_shot} \
    ${read_from_img_path} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"


python ${ofa_dir}/run_scripts/caption/coco_eval.py ${result_path}/${split}_predict.json ${data_dir}/nocaps_val_caption_coco_format.json

