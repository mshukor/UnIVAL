
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

exp_name=unival_s1

save_dir=${base_log_dir}/ofa/checkpoints/pretrain/${exp_name}

bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module


restore_file=${base_log_dir}/ofa/pretrained_models/bart.base/model.pt




image_dir=${base_data_dir}
data_dir=${base_data_dir}/ofa/pretrain_ours

mkdir -p $save_dir


neg_sample_dir=${data_dir}/negative_sample
data=${data_dir}/vision_language_caption.tsv #vision_language_mini.tsv
text_data= #${data_dir}/text_mini.tsv
image_data= #${data_dir}/image_mini.tsv
detection_data= #${data_dir}/detection_mini.tsv

image_text_data=${data_dir}/cc12m.tsv
image_text_cnt=8

image_text_vqa_data=${data_dir}/vision_language_qa.tsv
image_text_vqa_cnt=1

image_text_ground_data=${data_dir}/vision_language_ground.tsv
image_text_ground_cnt=1


selected_cols=0,1,2,3,4,5,6,7
text_selected_cols=0,1
image_selected_cols=0,1,2
detection_selected_cols=0,1,2


task=unify_task
arch=unival_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=2e-4
lr_scheduler=polynomial_decay
max_epoch=50
warmup_ratio=0.01
batch_size=2
update_freq=2
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=30
num_bins=1000

max_image_size=512



save_interval_updates=0

image_encoder_name=timm_resnet #vit_base_patch16_224
patch_image_size=384
resnet_type=resnet101
sample_patch_num=144

resnet_model_path=${base_log_dir}/pretrained_models/resnet101_a1h-36d3f2aa.pth


python3 -m torch.distributed.launch \
--nnodes=${NUM_NODES} \
--nproc_per_node=${GPUS_PER_NODE} \
--master_port=${MASTER_PORT} \
--node_rank=${RANK} \
--master_addr=${MASTER_ADDR} \
--use_env ${ofa_dir}/train.py \
  $data \
  --ddp-backend=no_c10d \
  --selected-cols=${selected_cols} \
  --text-selected-cols=${text_selected_cols} \
  --image-selected-cols=${image_selected_cols} \
  --detection-selected-cols=${detection_selected_cols} \
  --bpe-dir=${bpe_dir} \
  --user-dir=${user_dir} \
  --save-dir=${save_dir} \
  --neg-sample-dir=${neg_sample_dir} \
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
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=5.0 \
  --lr-scheduler=${lr_scheduler} --lr=${lr} \
  --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
  --log-format=simple --log-interval=10 \
  --fixed-validation-seed=7 \
  --keep-last-epochs=15 \
  --save-interval=1 \
  --save-interval-updates=${save_interval_updates} \
  --disable-validation \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --disable-entangle \
  --num-bins=${num_bins} \
  --patch-image-size=${patch_image_size} \
  --sample-patch-num=${sample_patch_num} \
  --max-image-size=${max_image_size} \
  --fp16 \
  --fp16-scale-window=128 \
  --num-workers=${num_workers} \
  --read-from-img-path \
  --image-dir=${image_dir} \
  --restore-file=${restore_file} \
  --image-encoder-name=${image_encoder_name} \
  --resnet-type=${resnet_type} \
  --resnet-model-path=${resnet_model_path} \
  --image-text-data=${image_text_data} \
  --image-text-cnt=${image_text_cnt} \
  --image-text-vqa-data=${image_text_vqa_data} \
  --image-text-vqa-cnt=${image_text_vqa_cnt} \
  --image-text-ground-data=${image_text_ground_data} \
  --image-text-ground-cnt=${image_text_ground_cnt} \
  --reset-dataloader --reset-meters --reset-optimizer 

