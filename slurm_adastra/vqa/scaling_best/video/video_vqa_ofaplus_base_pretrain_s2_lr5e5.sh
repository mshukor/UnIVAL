#!/bin/bash
   
#SBATCH --job-name=video_vqa_ofaplus_base_pretrain_s2_lr5e5
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus=16
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=closest 
#SBATCH -C MI250
#SBATCH -A gda2204
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/video_vqa_ofaplus_base_pretrain_s2_lr5e5.out
#SBATCH --exclusive
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun -l -N 2 -n 2 -c 128 --gpus=16 bash vqa/scaling_best/video/video_vqa_ofaplus_base_pretrain_s2_lr5e5.sh


