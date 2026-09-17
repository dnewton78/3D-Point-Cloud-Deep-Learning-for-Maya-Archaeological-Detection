#!/bin/bash
#SBATCH --job-name=swin3d_train
#SBATCH --output=logs/swin3d_%j.out
#SBATCH --error=logs/swin3d_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

# EDIT THESE FOR YOUR HPC ENVIRONMENT.
PROJECT_DIR="/path/to/repository/03_swin3d"
PYTHON="/path/to/environment/bin/python"
TILE_MANIFEST="/path/to/private/authorized_tile_manifest.csv"

mkdir -p "$PROJECT_DIR/jobs/logs"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export MAYA_TILE_MANIFEST="$TILE_MANIFEST"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python executable not found: $PYTHON" >&2
    exit 1
fi

"$PYTHON" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
"$PYTHON" -u training/train_swin3d.py
