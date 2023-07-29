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


exp_name=unival_refcocoplus



ofa_dir=/lus/home/NAT/gda2204/mshukor/code/unival
base_data_dir=/lus/scratch/NAT/gda2204/SHARED/data
base_log_dir=/work/NAT/gda2204/mshukor/logs

new_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs
save_dir=${new_base_log_dir}/ofa/checkpoints/refcocoplus/${exp_name}


log_dir=${save_dir}


mkdir -p $log_dir $save_dir

bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module

image_dir=${base_data_dir}

data_dir=${base_data_dir}/ofa/refcocoplus_data
data=${data_dir}/refcocoplus_train_1.tsv,${data_dir}/refcocoplus_train_2.tsv,${data_dir}/refcocoplus_train_3.tsv,${data_dir}/refcocoplus_train_4.tsv,${data_dir}/refcocoplus_train_5.tsv,${data_dir}/refcocoplus_train_6.tsv,${data_dir}/refcocoplus_train_7.tsv,${data_dir}/refcocoplus_train_8.tsv,${data_dir}/refcocoplus_train_9.tsv,${data_dir}/refcocoplus_train_10.tsv,${data_dir}/refcocoplus_val.tsv

restore_file=${base_log_dir}/ofa/checkpoints/pretrain/unival_s2_hs/checkpoint1.pt


selected_cols=0,4,2,3

task=refcoco
arch=unival_base
pretrained_model=  

criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
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
patch_image_size=512


image_encoder_name=timm_resnet #vit_base_patch16_224
resnet_type=resnet101

save_interval=1
validate_interval_updates=2000
save_interval_updates=0

sample_patch_num='--sample-patch-num=784' # ''


echo "max_epoch "${max_epoch}
echo "lr "${lr}
echo "patch_image_size "${patch_image_size}

log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}".log"
save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}
mkdir -p $save_path

acc_thresh=0.5

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
    --no-epoch-checkpoints --keep-best-checkpoints=1 \
    --save-interval=${save_interval} --validate-interval=1 \
    --save-interval-updates=${save_interval_updates} --validate-interval-updates=${validate_interval_updates} \
    --eval-acc \
    --eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}' \
    --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
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
    --fp16 \
    --fp16-scale-window=512 \
    --num-workers=0 \
    --image-dir=${image_dir} \
    ${sample_patch_num} \
    --acc-thresh=${acc_thresh} \
    --image-encoder-name=${image_encoder_name} \
    --strict
