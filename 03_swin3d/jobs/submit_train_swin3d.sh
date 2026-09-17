#!/bin/bash
#SBATCH --job-name=swin3d_train
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/swin3d_train_%j.out
#SBATCH --error=logs/swin3d_train_%j.err

set -euo pipefail

# EDIT THESE FOR YOUR HPC ENVIRONMENT.
PROJECT_DIR="/path/to/repository/03_swin3d"
PYTHON="/path/to/environment/bin/python"
TILE_MANIFEST="/path/to/private/authorized_tile_manifest.csv"

mkdir -p "$PROJECT_DIR/jobs/logs"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export MAYA_TILE_MANIFEST="$TILE_MANIFEST"

"$PYTHON" -u training/train_swin3d.py
