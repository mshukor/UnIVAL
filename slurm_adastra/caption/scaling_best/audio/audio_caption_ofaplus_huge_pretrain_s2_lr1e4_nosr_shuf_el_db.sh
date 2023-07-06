#!/bin/bash
   
#SBATCH --job-name=audio_caption_ofaplus_huge_inittext_lr1e4_nosr_shuf_el_db
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=closest 
#SBATCH --time=5:00:00
#SBATCH -C MI250
#SBATCH -A gda2204
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/audio_caption_ofaplus_huge_inittext_lr1e4_nosr_shuf_el_db.out
#SBATCH --exclusive
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun -l -N 1 -n 1 -c 128 --gpus=8 bash caption/scaling_best/audio/audio_caption_ofaplus_huge_pretrain_s2_lr1e4_nosr_shuf_el_db.sh


