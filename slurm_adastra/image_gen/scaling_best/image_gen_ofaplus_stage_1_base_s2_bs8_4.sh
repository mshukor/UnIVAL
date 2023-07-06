#!/bin/bash
   
#SBATCH --job-name=image_gen_ofaplus_stage_1_base_s2_bs8_4
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus=32
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=closest 
#SBATCH -C MI250
#SBATCH -A gda2204
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/image_gen_ofaplus_stage_1_base_s2_bs8_4.out
#SBATCH --exclusive
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun -l -N 4 -n 4 -c 128 --gpus=32 bash image_gen/scaling_best/image_gen_ofaplus_stage_1_base_s2_bs8_4.sh


