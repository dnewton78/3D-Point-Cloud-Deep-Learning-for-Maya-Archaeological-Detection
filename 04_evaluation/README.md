# 04 — Evaluation

Evaluation is separated into standard semantic-segmentation metrics and archaeology-specific AVD evaluation.

## Model label convention

All files in this directory operate on the **remapped Swin3D labels**:

- `0` = Ground
- `1` = Structure
- `2` = Plazuela

These are not the raw/preprocessed LAZ classifications (`2`, `6`, `27`).

## Standard metrics

`standard_metrics/` contains the standard model evaluation implementation and `Evaluate.ipynb`. These metrics describe semantic-segmentation behavior such as class agreement/overlap.

## Archaeological Vicinity Detection (AVD)

`avd_metric/` contains the archaeology-specific evaluation. The retained notebook is `notebooks/evaluate_archaeological_metric_cached_predictions.ipynb`, which reuses cached predictions for repeated metric analysis. Supporting code is `evaluate_avd.py` and `archaeological_vicinity_metric.py`.

AVD is an **evaluation metric**, distinct from the Soft AVD and vicinity losses used in training.
