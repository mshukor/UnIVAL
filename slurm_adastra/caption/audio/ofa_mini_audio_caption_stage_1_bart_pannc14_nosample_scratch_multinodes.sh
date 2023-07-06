#!/bin/bash
   
#SBATCH --job-name=ofa_mini_audio_caption_stage_1_bart_pannc14_nosample_scratch_multinodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/ofa_mini_audio_caption_stage_1_bart_pannc14_nosample_scratch_multinodes_test_pannspec_init.out
#SBATCH --exclusive
#SBATCH --time=100:00:00
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun bash caption/audio/ofa_mini_audio_caption_stage_1_bart_pannc14_nosample_scratch.sh


