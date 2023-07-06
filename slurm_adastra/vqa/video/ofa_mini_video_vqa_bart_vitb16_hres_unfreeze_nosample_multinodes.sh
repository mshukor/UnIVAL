#!/bin/bash
   
#SBATCH --job-name=ofa_mini_video_vqa_bart_vitb16_hres_unfreeze_nosample_multinodes
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/ofa_mini_video_vqa_bart_vitb16_hres_unfreeze_nosample_multinodes.out
#SBATCH --exclusive
#SBATCH --time=100:00:00
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun bash vqa/video/ofa_mini_video_vqa_bart_vitb16_hres_unfreeze_nosample.sh


