#!/bin/bash
   
#SBATCH --job-name=ofa_mini_det_video_caption_stage_1_bart_viddatacapqa_allres_allresnxtvid_init_8f216f_vids2_lr4_nosample_hres_longt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=closest 
#SBATCH -C MI250
#SBATCH -A gda2204
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/ofa_mini_det_video_caption_stage_1_bart_viddatacapqa_allres_allresnxtvid_init_8f216f_vids2_lr4_nosample_hres_longt.out
#SBATCH --exclusive
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun -l -N 1 -n 1 -c 128 --gpus=8 bash caption/video/ofa_mini_det_video_caption_stage_1_bart_viddatacapqa_allres_allresnxtvid_init_8f216f_vids2_lr4_nosample_hres_longt.sh


