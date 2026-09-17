# Experiment Index

This index links each retained training variant to its implementation and intended evaluation path. Fill in data-version and checkpoint identifiers for the exact archived NSF results before release.

| Experiment | Training variant | Model | Data version | Config | Checkpoint | Evaluation |
|---|---|---|---|---|---|---|
| `EXP-01` | Generalized Dice | SWIN3D-S | *document archived data version* | `03_swin3d/configs/swin3d_small.py` | *document released checkpoint* | Standard + AVD |
| `EXP-02` | Generalized Dice + Vicinity Loss | SWIN3D-S | *document archived data version* | `03_swin3d/configs/swin3d_small.py` | *document released checkpoint* | Standard + AVD |
| `EXP-03` | Generalized Dice + Soft AVD | SWIN3D-S | *document archived data version* | `03_swin3d/configs/swin3d_small.py` | *document released checkpoint* | Standard + AVD |

## EXP-01 — Generalized Dice baseline

- Notebook: `03_swin3d/training/notebooks/Training_Generalized_Dice.ipynb`
- Loss: `03_swin3d/losses/generalized_dice.py`
- Production script: `03_swin3d/training/train_swin3d.py`

## EXP-02 — Generalized Dice + Vicinity Loss

- Notebook: `03_swin3d/training/notebooks/Training_Dice_Vicinity.ipynb`
- Loss: `03_swin3d/losses/swin3d_archaeological_vicinity_loss.py`

## EXP-03 — Generalized Dice + Soft AVD

- Notebook: `03_swin3d/training/notebooks/Training_Dice_SoftAVD.ipynb`
- Loss: `03_swin3d/losses/soft_avd.py`

## Evaluation chain

All experiments may be evaluated with:

- standard semantic-segmentation evaluation: `04_evaluation/standard_metrics/`; and
- archaeology-specific AVD evaluation: `04_evaluation/avd_metric/`.

The AVD evaluation metric is not the same as the Soft AVD training loss. The evaluation metric measures spatial usefulness around archaeological features; the training losses are optimization objectives used while fitting the network.
