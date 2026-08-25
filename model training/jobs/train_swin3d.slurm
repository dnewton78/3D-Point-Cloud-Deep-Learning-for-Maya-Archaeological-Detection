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

echo "=========================================="
echo "Swin3D Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# ============================================================
# Project path
# ============================================================

PROJECT_DIR="/lustre/work/archcomplab/ssaraf/Maya Archarological Sites/Model Training/swin3d_project"

cd "$PROJECT_DIR"

echo
echo "Project directory:"
pwd

# Make project packages importable
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ============================================================
# Use Python directly from your working Conda environment
# ============================================================

PYTHON="/work/archcomplab/ssaraf/envs/maya_env/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at:"
    echo "$PYTHON"
    exit 1
fi

echo
echo "Python executable:"
echo "$PYTHON"

"$PYTHON" --version

# ============================================================
# CPU settings
# ============================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================
# GPU / PyTorch check
# ============================================================

echo
echo "GPU:"
nvidia-smi || true

echo
echo "PyTorch CUDA check:"

"$PYTHON" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("PyTorch cannot see the allocated GPU.")
PY

# ============================================================
# Verify imports before training
# ============================================================

echo
echo "Checking project imports..."

"$PYTHON" - <<'PY'
from configs.swin3d_small import TRAIN_CONFIG
from datasets.belize_laz_dataset import create_dataloaders
from losses.generalized_dice import GeneralizedDiceLoss
from models.swin3d_model import Model

print("Imports successful.")
print("Batch size:", TRAIN_CONFIG["batch_size"])
print("Augmentation ratio:", TRAIN_CONFIG["augmentation_ratio"])
print("Warm-up epochs:", TRAIN_CONFIG["warmup_epochs"])
PY

# ============================================================
# Training
# ============================================================

echo
echo "Starting training..."
echo

"$PYTHON" -u scripts/train_swin3d.py

echo
echo "=========================================="
echo "Training finished"
echo "End: $(date)"
echo "=========================================="