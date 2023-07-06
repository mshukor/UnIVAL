#!/bin/bash
   
#SBATCH --job-name=ofa_mini_msvd_video_vqa_bart_viddatacapqa_allres_allresnxtvid_init_8f_vids2_lr4_nosample_hres_longt2_bs8_1kans_lr1e4_freezeenc
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus=16
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=closest 
#SBATCH -C MI250
#SBATCH -A gda2204
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/ofa_mini_msvd_video_vqa_bart_viddatacapqa_allres_allresnxtvid_init_8f_vids2_lr4_nosample_hres_longt2_bs8_1kans_lr1e4_freezeenc.out
#SBATCH --exclusive
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun -l -N 2 -n 2 -c 128 --gpus=16 bash vqa/video/msvd/ofa_mini_msvd_video_vqa_bart_viddatacapqa_allres_allresnxtvid_init_8f_vids2_lr4_nosample_hres_longt2_bs8_1kans_lr1e4_freezeenc.sh


