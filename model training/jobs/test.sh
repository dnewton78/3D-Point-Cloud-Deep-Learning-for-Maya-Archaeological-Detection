#!/bin/bash

#SBATCH --job-name=swin3d_test
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --output=/lustre/work/archcomplab/ssaraf/Maya\ Archarological\ Sites/Model\ Training/swin3d_project/jobs/logs/test_%j.out
#SBATCH --error=/lustre/work/archcomplab/ssaraf/Maya\ Archarological\ Sites/Model\ Training/swin3d_project/jobs/logs/test_%j.err

export OMP_NUM_THREADS=4

module load anaconda
module load cuda/11.8

source activate /work/archcomplab/ssaraf/envs/maya_env

echo "Node:"
hostname

echo
echo "Python:"
which python
python --version

echo
echo "GLIBC:"
ldd --version | head -1

echo
echo "PyTorch:"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

echo
echo "MinkowskiEngine:"
python -c "import MinkowskiEngine as ME; print('MinkowskiEngine loaded')"