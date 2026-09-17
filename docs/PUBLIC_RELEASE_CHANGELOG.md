# Public Release Cleanup Changelog

This file records the release-preparation changes made after the initial repository reorganization.

## Added / renamed

- `03_swin3d/training/notebooks/Training_Generalized_Dice.ipynb` — renamed from `Training.ipynb` so the baseline experiment is explicit; internal checkpoint paths were also replaced with repository-relative/generic paths.
- `03_swin3d/dataset_splits/README.md` — documents the public opaque `tile_id` format and the separate authorized `tile_id,path` manifest workflow.
- `03_swin3d/dataset_splits/authorized_tile_manifest.example.csv` — safe example/template only; contains no real archaeological source paths.
- `docs/PUBLIC_RELEASE_CHANGELOG.md` — this audit trail.

## Removed

- `03_swin3d/training/notebooks/Training.ipynb` — replaced by the explicitly named `Training_Generalized_Dice.ipynb`.
- `03_swin3d/training/scripts/train_swin3d.py` — duplicate of the production training script; the only authoritative script is now `03_swin3d/training/train_swin3d.py`.

The public tree also contains no `Training_old.ipynb`, `test.ipynb`, `__pycache__/`, `*.pyc`, or historical job-log artifacts.

## Modified files

### Repository documentation

- `README.md`
  - makes the two label systems prominent: LAZ `2/6/27` versus model/evaluation `0/1/2`;
  - documents the exact remapping `2→0`, `6→1`, `27→2`;
  - explains public split privacy and the private manifest workflow;
  - updates the repository tree and baseline notebook name.
- `03_swin3d/README.md`
  - adds the label-system warning;
  - explicitly documents EXP-01 Generalized Dice, EXP-02 Dice + Vicinity, and EXP-03 Dice + Soft AVD;
  - distinguishes training losses from the AVD evaluation metric;
  - documents opaque public split IDs and omitted development artifacts.
- `04_evaluation/README.md`
  - states that evaluation uses model labels `0=Ground`, `1=Structure`, `2=Plazuela`;
  - distinguishes standard segmentation metrics from AVD;
  - identifies the cached-predictions AVD notebook as the retained workflow.
- `data/README.md`
  - adds archaeological data/location-disclosure guidance and notes the authorized external path mapping.
- `docs/EXPERIMENT_INDEX.md`
  - rewrites the index around EXP-01/02/03;
  - adds model, data-version, config, checkpoint, and evaluation-chain fields;
  - distinguishes the Soft AVD training loss from the AVD evaluation metric.
- `docs/CURRENT_FILE_MIGRATION_MAP.md`
  - updates the baseline notebook name and removes the duplicate training-script destination.
- `docs/FILE_INVENTORY.txt`
  - regenerated from the cleaned public tree.

### Class mapping / evaluation correctness

- `04_evaluation/avd_metric/archaeological_vicinity_metric.py`
  - corrects the misleading header from `1=plazuela, 2=structure` to `1=Structure, 2=Plazuela`.
- `04_evaluation/avd_metric/notebooks/evaluate_archaeological_metric_cached_predictions.ipynb`
  - removes internal storage paths;
  - replaces them with portable/generic locations;
  - fixes a malformed `OUTPUT_COMPARE_DIR.mkdir(...)` cell discovered during cleanup.
- `04_evaluation/standard_metrics/notebooks/Evaluate.ipynb`
  - removes internal project/checkpoint/results paths and replaces them with portable/generic locations.

### Dataset privacy / portability

- `03_swin3d/dataset_splits/train.csv`
  - replaces 371 machine-specific coordinate-bearing LAZ paths with opaque `tile_id` values.
- `03_swin3d/dataset_splits/val.csv`
  - replaces 79 machine-specific coordinate-bearing LAZ paths with opaque `tile_id` values.
- `03_swin3d/dataset_splits/test.csv`
  - replaces 80 machine-specific coordinate-bearing LAZ paths with opaque `tile_id` values.
- `03_swin3d/datasets/belize_laz_dataset.py`
  - retains the correct LAZ-to-model class remapping;
  - adds support for public `tile_id` split files;
  - resolves IDs through a separate private `tile_id,path` manifest;
  - accepts the manifest via `tile_manifest=` or `MAYA_TILE_MANIFEST`;
  - retains legacy/private `path` split support.
- `03_swin3d/configs/swin3d_small.py`
  - adds a `tile_manifest` configuration entry and recommends the environment-variable route for private paths.
- `01_data_preparation/data_preparation.ipynb`
  - replaces internal storage paths with generic authorized-data placeholders;
  - redacts literal coordinate-bearing tile identifiers present in notebook text/output while preserving the preprocessing code and source LAZ class IDs `2/6/27`.
- `05_visualization/Visualization.ipynb`
  - removes internal checkpoint/storage paths;
  - redacts literal coordinate-bearing tile identifiers in notebook text/output.

### Training and HPC portability

- `03_swin3d/training/train_swin3d.py`
  - established as the single production training script;
  - removes the internal checkpoint path and uses a repository-relative `checkpoints/generalized_dice` directory;
  - passes the optional private tile manifest to the dataloader.
- `03_swin3d/training/notebooks/Training_Dice_Vicinity.ipynb`
  - replaces internal checkpoint paths with generic/portable locations.
- `03_swin3d/training/notebooks/Training_Dice_SoftAVD.ipynb`
  - replaces internal checkpoint paths with generic/portable locations.
- `03_swin3d/jobs/train_swin3d.slurm`
  - converts hard-coded UNL/HCC paths into editable `PROJECT_DIR`, `PYTHON`, and `TILE_MANIFEST` template variables;
  - points to the single production `training/train_swin3d.py` script.
- `03_swin3d/jobs/submit_train_swin3d.sh`
  - removes internal cluster/environment paths and converts the script into a portable template.
- `03_swin3d/jobs/test.sh`
  - removes internal cluster/environment paths and converts the environment test into a portable template.
- `.gitignore`
  - adds/ensures exclusions for Python caches, model checkpoints, job logs, notebook checkpoints, and the private authorized tile manifest.

## Validation performed

- Verified no public-tree matches remain for `/mnt/nrdstor`, `/lustre/work/archcomplab`, or `/work/archcomplab`.
- Verified no literal coordinate-style `100m_<x>_<y>` tile identifiers remain in the public tree.
- Verified no `Training_old.ipynb`, `test.ipynb`, duplicate `training/scripts/train_swin3d.py`, `__pycache__/`, or `*.pyc` artifacts remain.
- Verified the public split sizes remain 371 train / 79 validation / 80 test.
- Python source under Swin3D, evaluation, and visualization passed `compileall` syntax checking before cache files were removed.
