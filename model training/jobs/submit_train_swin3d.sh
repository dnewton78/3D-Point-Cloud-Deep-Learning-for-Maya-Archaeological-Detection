#!/bin/bash

#SBATCH --job-name=swin3d_train
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_32gb

#SBATCH --output=/lustre/work/archcomplab/ssaraf/Maya\ Archarological\ Sites/Model\ Training/swin3d_project/jobs/logs/swin3d_train_%j.out
#SBATCH --error=/lustre/work/archcomplab/ssaraf/Maya\ Archarological\ Sites/Model\ Training/swin3d_project/jobs/logs/swin3d_train_%j.err

module purge
module load anaconda
module load cuda/11.8

source activate /work/archcomplab/ssaraf/envs/maya_env

cd "/lustre/work/archcomplab/ssaraf/Maya Archarological Sites/Model Training/swin3d_project"

python -u train_swin3d.py
