#!/bin/bash
   
#SBATCH --job-name=ofa_mini_video_caption_stage_1_video_audiovcc_onestage_viddatacapqa_pretrain_bart_allres_allresnxtvid_allpannc14mel64h200_init_8f_lr4_wav_audioembLN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=closest 
#SBATCH -C MI250
#SBATCH -A gda2204
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/ofa_mini_video_caption_stage_1_video_audiovcc_onestage_viddatacapqa_pretrain_bart_allres_allresnxtvid_allpannc14mel64h200_init_8f_lr4_wav_audioembLN.out
#SBATCH --exclusive
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun -l -N 1 -n 1 -c 128 --gpus=8 bash caption/video/ofa_mini_video_caption_stage_1_video_audiovcc_onestage_viddatacapqa_pretrain_bart_allres_allresnxtvid_allpannc14mel64h200_init_8f_lr4_wav_audioembLN.sh


