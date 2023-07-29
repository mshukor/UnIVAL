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

exp_name=unival_image_gen_stage_2

image_dir=${base_data_dir}
data_dir=${base_data_dir}/ofa/image_gen_data

data_dir_train=/lus/scratch/NAT/gda2204/SHARED/tmp/ofa/image_gen_data
data=${data_dir_train}/coco_vqgan_train_1.tsv,${data_dir_train}/coco_vqgan_train_2.tsv,${data_dir_train}/coco_vqgan_train_3.tsv,${data_dir_train}/coco_vqgan_train_4.tsv,${data_dir_train}/coco_vqgan_train_5.tsv,${data_dir_train}/coco_vqgan_train_6.tsv,${data_dir_train}/coco_vqgan_train_7.tsv,${data_dir_train}/coco_vqgan_train_8.tsv,${data_dir_train}/coco_vqgan_train_9.tsv,${data_dir_train}/coco_vqgan_train_10.tsv,${data_dir}/coco_vqgan_dev.tsv


# Note: If you have shuffled the data in advance, please uncomment the line below.
# data=${data_dir}/coco_vqgan_train_1.tsv,${data_dir}/coco_vqgan_train_2.tsv,${data_dir}/coco_vqgan_train_3.tsv,${data_dir}/coco_vqgan_train_4.tsv,${data_dir}/coco_vqgan_train_5.tsv,${data_dir}/coco_vqgan_train_6.tsv,${data_dir}/coco_vqgan_train_7.tsv,${data_dir}/coco_vqgan_train_8.tsv,${data_dir}/coco_vqgan_train_9.tsv,${data_dir}/coco_vqgan_train_10.tsv,${data_dir}/coco_vqgan_dev.tsv

restore_file=/work/NAT/gda2204/mshukor/logs/ofa/checkpoints/image_gen/unival_image_gen_stage_1/50000_2000_1e-3/checkpoint_best.pt


selected_cols=0,2,1

# save_dir=${base_log_dir}/ofa/checkpoints/image_gen/${exp_name}
save_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs
save_dir=${save_base_log_dir}/ofa/checkpoints/image_gen/${exp_name}

log_dir=${save_dir}

mkdir -p $log_dir $save_dir



bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module


task=image_gen
arch=unival_base
criterion=clip_scst_reward_criterion
batch_size=4
update_freq=1
encoder_drop_path_rate=0.0
decoder_drop_path_rate=0.0
dropout=0.0
attention_dropout=0.0
max_src_length=64
max_tgt_length=1024
num_bins=1000
code_image_size=256
constraint_range=50265,58457

VQGAN_MODEL_PATH=${base_log_dir}/ofa/pretrained_models/vqgan/last.ckpt
VQGAN_CONFIG_PATH=${base_log_dir}/ofa/pretrained_models/vqgan/model.yaml
CLIP_MODEL_PATH=${base_log_dir}/ofa/pretrained_models/clip/ViT-B-16.pt
GEN_IMAGES_PATH=/lus/scratch/NAT/gda2204/SHARED/tmp/results/${exp_name}
mkdir -p $GEN_IMAGES_PATH



###
image_encoder_name=timm_resnet #vit_base_patch16_224
patch_image_size=480
resnet_type=resnet101

resnet_model_path=${base_log_dir}/pretrained_models/resnet101-5d3b4d8f.pth

# video
video_encoder_name=all_resnext101
patch_frame_size=384
video_model_path=${base_log_dir}/pretrained_models/3dcnn/resnext-101-kinetics.pth #${base_log_dir}/pretrained_models/TimeSformer_divST_8x32_224_K600.pyth
num_frames=4


sample_patch_num='--sample-patch-num=784' # ''

save_interval_updates=0


for total_num_updates in {5000,}; do
  echo "total_num_updates "${total_num_updates}
  for warmup_updates in {0,}; do
    echo "warmup_updates "${warmup_updates}  
    for lr in {1e-6,}; do
      echo "lr "${lr}

        log_file=${log_dir}/${total_num_updates}"_"${warmup_updates}"_"${lr}"_rank"${RANK}".log"
        save_path=${save_dir}/${total_num_updates}"_"${warmup_updates}"_"${lr}
        mkdir -p $save_path

        python3 -m torch.distributed.launch \
            --nnodes=${NUM_NODES} \
            --nproc_per_node=${GPUS_PER_NODE} \
            --master_port=${MASTER_PORT} \
            --node_rank=${RANK} \
            --master_addr=${MASTER_ADDR} \
            --use_env ${ofa_dir}/train.py \
            ${data} \
            --selected-cols=${selected_cols} \
            --bpe-dir=${bpe_dir} \
            --user-dir=${user_dir} \
            --restore-file=${restore_file} \
            --save-dir=${save_path} \
            --task=${task} \
            --arch=${arch} \
            --criterion=${criterion} \
            --batch-size=${batch_size} \
            --batch-size-valid=1 \
            --update-freq=${update_freq} \
            --encoder-normalize-before \
            --decoder-normalize-before \
            --share-decoder-input-output-embed \
            --share-all-embeddings \
            --layernorm-embedding \
            --patch-layernorm-embedding \
            --code-layernorm-embedding \
            --encoder-drop-path-rate=${encoder_drop_path_rate} \
            --decoder-drop-path-rate=${decoder_drop_path_rate} \
            --dropout=${dropout} \
            --attention-dropout=${attention_dropout} \
            --weight-decay=0.01 \
            --optimizer=adam \
            --adam-betas="(0.9,0.999)" \
            --adam-eps=1e-08 \
            --clip-norm=1.0 \
            --lr-scheduler=polynomial_decay \
            --lr=${lr} \
            --total-num-update=${total_num_updates} \
            --warmup-updates=${warmup_updates} \
            --log-format=simple \
            --log-interval=10 \
            --fixed-validation-seed=7 \
            --keep-last-epochs=15 \
            --save-interval=1 --validate-interval=1 \
            --save-interval-updates=100 --validate-interval-updates=200 \
            --freeze-resnet \
            --max-update=${total_num_updates} \
            --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
            --eval-args='{"beam":24,"min_len":1024,"max_len_a":0,"max_len_b":1024,"sampling_topk":256,"temperature":1.0}' \
            --scst \
            --scst-args='{"beam":5,"min_len":1024,"max_len_a":0,"max_len_b":1024,"sampling_topk":256,"temperature":1.0}' \
            --max-src-length=${max_src_length} \
            --max-tgt-length=${max_tgt_length} \
            --find-unused-parameters \
            --add-type-embedding \
            --scale-attn \
            --scale-fc \
            --scale-heads \
            --disable-entangle \
            --num-bins=${num_bins} \
            --code-image-size=${code_image_size} \
            --constraint-range=${constraint_range} \
            --vqgan-model-path=${VQGAN_MODEL_PATH} \
            --vqgan-config-path=${VQGAN_CONFIG_PATH} \
            --clip-model-path=${CLIP_MODEL_PATH} \
            --gen-images-path=${GEN_IMAGES_PATH} \
            --fp16 \
            --fp16-scale-window=256 \
            --num-workers=0 \
            --image-encoder-name=${image_encoder_name} \
            --image-dir=${image_dir} \
            --video-encoder-name=${video_encoder_name} \
            --video-model-path=${video_model_path} \
            --patch-frame-size=${patch_frame_size} \
            ${sample_patch_num} \
            --resnet-type=${resnet_type} \
            --resnet-model-path=${resnet_model_path} \
            --reset-optimizer --reset-dataloader --reset-meters 
    done
  done
done
