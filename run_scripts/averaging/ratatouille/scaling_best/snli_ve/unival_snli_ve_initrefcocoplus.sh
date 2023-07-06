#!/usr/bin/env

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



exp_name=unival_snli_ve_initrefcocoplus



ofa_dir=/lus/home/NAT/gda2204/mshukor/code/unival
base_data_dir=/lus/scratch/NAT/gda2204/SHARED/data
base_log_dir=/work/NAT/gda2204/mshukor/logs

save_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs
save_dir=${save_base_log_dir}/ofa/checkpoints/snli_ve/${exp_name}

# save_dir=${base_log_dir}/ofa/checkpoints/snli_ve/${exp_name}
log_dir=${save_dir}


mkdir -p $log_dir $save_dir

bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module

image_dir=${base_data_dir}




data_dir=${base_data_dir}/ofa/snli_ve_data
data=${data_dir}/snli_ve_train.tsv,${data_dir}/snli_ve_dev.tsv

restore_file=/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/refcocoplus/unival_refcocoplus/10_5e-5_512/checkpoint_best.pt


selected_cols=0,2,3,4,5

task=snli_ve
arch=unival_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=5e-5
max_epoch=10
warmup_ratio=0.06
batch_size=8
update_freq=4
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
patch_image_size=480
prompt_type="prev_output"



echo "max_epoch "${max_epoch}
echo "lr "${lr}

log_file=${log_dir}/${max_epoch}"_"${lr}".log"
save_path=${save_dir}/${max_epoch}"_"${lr}
mkdir -p $save_path






###
image_encoder_name=timm_resnet #vit_base_patch16_224 timm_resnet resnet
patch_image_size=480
resnet_type=resnet101

resnet_model_path=${base_log_dir}/pretrained_models/resnet101-5d3b4d8f.pth

# video
video_encoder_name=all_resnext101
patch_frame_size=384
video_model_path=${base_log_dir}/pretrained_models/3dcnn/resnext-101-kinetics.pth #${base_log_dir}/pretrained_models/TimeSformer_divST_8x32_224_K600.pyth
num_frames=4

save_interval=1
validate_interval_updates=50000
save_interval_updates=0


sample_patch_num='--sample-patch-num=784' # ''



python3 -m torch.distributed.launch \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --use_env ${ofa_dir}/train.py \
    $data \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --restore-file=${restore_file} \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-dir=${save_path} \
    --task=${task} \
    --arch=${arch} \
    --criterion=${criterion} \
    --label-smoothing=${label_smoothing} \
    --batch-size=${batch_size} \
    --update-freq=${update_freq} \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --layernorm-embedding \
    --patch-layernorm-embedding \
    --code-layernorm-embedding \
    --resnet-drop-path-rate=${resnet_drop_path_rate} \
    --encoder-drop-path-rate=${encoder_drop_path_rate} \
    --decoder-drop-path-rate=${decoder_drop_path_rate} \
    --dropout=${dropout} \
    --attention-dropout=${attention_dropout} \
    --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
    --lr-scheduler=polynomial_decay --lr=${lr} \
    --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
    --log-format=simple --log-interval=10 \
    --fixed-validation-seed=7 \
    --keep-best-checkpoints=1 \
    --no-epoch-checkpoints \
    --save-interval=1 --validate-interval=1 \
    --save-interval-updates=${save_interval_updates} --validate-interval-updates=${validate_interval_updates} \
    --best-checkpoint-metric=snli_score --maximize-best-checkpoint-metric \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --find-unused-parameters \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --prompt-type=${prompt_type} \
    --fp16 \
    --fp16-scale-window=512 \
    --num-workers=0 \
    --image-dir=${image_dir} \
    ${sample_patch_num} \
    --image-encoder-name=${image_encoder_name} \
    --image-dir=${image_dir} \
    --video-encoder-name=${video_encoder_name} \
    --video-model-path=${video_model_path} \
    --patch-frame-size=${patch_frame_size} \
    --reset-dataloader --reset-meters --reset-optimizer \
    --strict \
    --resnet-model-path=${resnet_model_path} 

        # --add-caption \
