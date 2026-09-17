#!/bin/bash
#SBATCH --job-name=swin3d_env_test
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

set -euo pipefail

# EDIT THESE FOR YOUR HPC ENVIRONMENT.
PROJECT_DIR="/path/to/repository/03_swin3d"
PYTHON="/path/to/environment/bin/python"

mkdir -p "$PROJECT_DIR/jobs/logs"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Node: $(hostname)"
"$PYTHON" --version
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
"$PYTHON" -c "import MinkowskiEngine as ME; print('MinkowskiEngine loaded')"
