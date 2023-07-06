
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


exp_name=unival_vqa_initsnlive


image_dir=${base_data_dir}
data_dir=${base_data_dir}/ofa/vqa_data
# data=${data_dir}/vqa_train.tsv,${data_dir}/vqa_val.tsv
# Note: If you have shuffled the data in advance, please uncomment the line below.
data=${data_dir}/vqa_train_1.tsv,${data_dir}/vqa_train_2.tsv,${data_dir}/vqa_train_3.tsv,${data_dir}/vqa_train_4.tsv,${data_dir}/vqa_train_5.tsv,${data_dir}/vqa_train_6.tsv,${data_dir}/vqa_train_7.tsv,${data_dir}/vqa_train_8.tsv,${data_dir}/vqa_train_9.tsv,${data_dir}/vqa_train_10.tsv,${data_dir}/vqa_val.tsv
ans2label_file=${base_data_dir}/ofa/vqa_data/trainval_ans2label.pkl


selected_cols=0,5,2,3,4



save_base_log_dir=/lus/scratch/NAT/gda2204/SHARED/logs
save_dir=${save_base_log_dir}/ofa/checkpoints/vqa/${exp_name}

# save_dir=${base_log_dir}/ofa/checkpoints/vqa/${exp_name}
log_dir=${save_dir}

mkdir -p $log_dir $save_dir

restore_file=/lus/scratch/NAT/gda2204/SHARED/logs/ofa/checkpoints/snli_ve/unival_snli_ve/10_5e-5/checkpoint_best.pt

lr=1e-4


bpe_dir=${ofa_dir}/utils/BPE
user_dir=${ofa_dir}/ofa_module



task=vqa_gen
arch=unival_base


criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
batch_size=16
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_object_length=30
max_tgt_length=30
num_bins=1000
# patch_image_size=480

uses_ema="--uses-ema"
store_ema="--store-ema"
ema_fp32="--ema-fp32"
ema_decay=0.9999
ema_start_update=0

# Specify the inference type in validation after each fine-tuning epoch
# As mentioned in the readme, you can choose from allcand or beamsearch evaluation, default to allcand
val_inference_type=beamsearch

# Specify whether to activate unconstrained VQA finetuning, which does not use a pre-defined candidate answer set
# If --unconstrained-training is acitvated, --ans2label-file will **not be used even if it is specified**
# Meanwhile, --val-inference-type must be set to **beamsearch**
# By default, we follow the constrained finetuning as we mentioned in OFA paper, the candidate answer set shall be specified by --ans2label-file
# For more details about this option, please refer to issue #123 and PR #124
unconstrained_training_flag=""
# unconstrained_training_flag="--unconstrained-training"





save_interval_updates=0

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

eval_args='--eval-args={"beam":5,"unnormalized":true,"temperature":1.0,"stop_on_max_len":true}'

validate_interval_updates=2000
save_interval_updates=0


for max_epoch in {20,}; do
  echo "max_epoch "${max_epoch}
  for warmup_ratio in {0.04,}; do
    echo "warmup_updates "${warmup_updates}  
    for lr in {$lr,}; do
      echo "lr "${lr}
      for patch_image_size in {$patch_image_size,}; do
        echo "patch_image_size "${patch_image_size}

        log_file=${log_dir}/${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}"_rank"${RANK}".log"
        save_path=${save_dir}/${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}
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
            --weight-decay=0.01 \
            --optimizer=adam \
            --adam-betas="(0.9,0.999)" \
            --adam-eps=1e-08 \
            --clip-norm=1.0 \
            --lr-scheduler=polynomial_decay \
            --lr=${lr} \
            --max-epoch=${max_epoch} \
            --warmup-ratio=${warmup_ratio} \
            --log-format=simple \
            --log-interval=10 \
            --fixed-validation-seed=7 \
            --keep-best-checkpoints=1 \
            --no-epoch-checkpoints \
            --save-interval=1 --validate-interval=1 \
            --save-interval-updates=${save_interval_updates} --validate-interval-updates=${validate_interval_updates} \
            --best-checkpoint-metric=vqa_score --maximize-best-checkpoint-metric \
            --max-src-length=${max_src_length} \
            --max-object-length=${max_object_length} \
            --max-tgt-length=${max_tgt_length} \
            --find-unused-parameters \
            --freeze-encoder-embedding \
            --freeze-decoder-embedding \
            ${unconstrained_training_flag} \
            --ans2label-file=${ans2label_file} \
            --valid-batch-size=20 \
            --add-type-embedding \
            --scale-attn \
            --scale-fc \
            --scale-heads \
            --disable-entangle \
            --num-bins=${num_bins} \
            --patch-image-size=${patch_image_size} \
            --prompt-type=prev_output \
            --fp16 \
            --fp16-scale-window=512 \
            ${uses_ema} \
            ${store_ema} \
            ${ema_fp32} \
            --ema-decay=${ema_decay} \
            --ema-start-update=${ema_start_update} \
            --val-inference-type=${val_inference_type} \
            --num-workers=0 \
            --image-encoder-name=${image_encoder_name} \
            --image-dir=${image_dir} \
            --video-encoder-name=${video_encoder_name} \
            --video-model-path=${video_model_path} \
            --patch-frame-size=${patch_frame_size} \
            ${sample_patch_num} \
            ${eval_args} \
            --no-epoch-checkpoints \
            --resnet-type=${resnet_type} \
            --resnet-model-path=${resnet_model_path} \
            --reset-dataloader --reset-meters --reset-optimizer 

      done
    done
  done
done
