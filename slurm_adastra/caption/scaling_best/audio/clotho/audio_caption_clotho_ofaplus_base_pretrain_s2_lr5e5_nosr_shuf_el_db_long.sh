#!/bin/bash
   
#SBATCH --job-name=audio_caption_clotho_ofaplus_base_pretrain_s2_lr5e5_nosr_shuf_el_db_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=closest 
#SBATCH --time=10:00:00
#SBATCH -C MI250
#SBATCH -A gda2204
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/audio_caption_clotho_ofaplus_base_pretrain_s2_lr5e5_nosr_shuf_el_db_long.out
#SBATCH --exclusive
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun -l -N 1 -n 1 -c 128 --gpus=8 bash caption/scaling_best/audio/clotho/audio_caption_clotho_ofaplus_base_pretrain_s2_lr5e5_nosr_shuf_el_db_long.sh


