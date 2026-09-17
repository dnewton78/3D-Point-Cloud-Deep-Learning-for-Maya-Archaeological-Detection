# 03 — Swin3D

Core model-development files are grouped here by responsibility:

- `configs/` — experiment/training configuration.
- `datasets/` — LAZ point-cloud dataset and dataloader code.
- `models/` — Swin3D model wrapper.
- `losses/` — Generalized Dice, Soft AVD, and archaeological vicinity losses.
- `training/` — validated production training script and the three retained experiment notebooks.
- `dataset_splits/` — public train/validation/test membership using opaque tile IDs.
- `jobs/` — portable shell/SLURM templates with placeholder paths.

## Class-label convention

The preprocessing LAZ classifications and the model labels are intentionally different:

| Stage | Ground | Structure | Plazuela |
|---|---:|---:|---:|
| Preprocessed LAZ | `2` | `6` | `27` |
| Swin3D / evaluation | `0` | `1` | `2` |

`datasets/belize_laz_dataset.py` performs the remapping `2→0`, `6→1`, `27→2`. Model losses and evaluation code use the remapped 0/1/2 labels.

## Training experiments

### Experiment 1 — Generalized Dice baseline

- `training/notebooks/Training_Generalized_Dice.ipynb`
- `losses/generalized_dice.py`
- `training/train_swin3d.py`

### Experiment 2 — Generalized Dice + Vicinity Loss

- `training/notebooks/Training_Dice_Vicinity.ipynb`
- `losses/swin3d_archaeological_vicinity_loss.py`

### Experiment 3 — Generalized Dice + Soft AVD

- `training/notebooks/Training_Dice_SoftAVD.ipynb`
- `losses/soft_avd.py`

These training losses are distinct from the **AVD evaluation metric** in `../04_evaluation/avd_metric/`. Dice/IoU-style metrics characterize semantic-segmentation behavior, while AVD evaluates archaeology-specific spatial usefulness of predictions.

## Public split files

`dataset_splits/train.csv`, `val.csv`, and `test.csv` publish only opaque `tile_id` values. The actual LAZ paths belong in a separate authorized manifest with columns `tile_id,path`; do not commit that manifest if it exposes restricted locations. The loader accepts it through the `MAYA_TILE_MANIFEST` environment variable or its `tile_manifest` argument.

`Training_old.ipynb`, `test.ipynb`, duplicate training scripts, Python caches, and historical HPC logs are intentionally omitted from the public release.
