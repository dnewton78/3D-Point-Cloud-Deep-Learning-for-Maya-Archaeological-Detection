# 3D Point Cloud Deep Learning for Maya Archaeological Detection

A research repository for detecting archaeological features in 3D LiDAR point clouds using **Swin3D-based semantic segmentation**. The project organizes the complete workflow from point-cloud preparation and synthetic data generation through model training, archaeological evaluation, and visualization.

## Critical class-label convention

This workflow intentionally uses **two different class-numbering systems**. Do not interchange them.

### Preprocessed LAZ source classifications

| LAZ classification | Meaning |
|---:|---|
| `2` | Ground |
| `6` | Structure |
| `27` | Plazuela |

### Swin3D model / evaluation labels

The dataset loader remaps the LAZ classifications to contiguous model labels:

| Model label | Meaning | Source LAZ class |
|---:|---|---:|
| `0` | Ground | `2` |
| `1` | Structure | `6` |
| `2` | Plazuela | `27` |

All Swin3D losses, predictions, standard evaluation, and AVD evaluation use the **model labels (0/1/2)**. Raw/preprocessed LAZ files retain the **source classifications (2/6/27)**.

The goal is not only to measure exact point-wise segmentation performance, but also to evaluate whether predicted archaeological features occur within a useful spatial vicinity of the true archaeological locations.

---

## Project workflow

```text
Raw LAS / LAZ point clouds
        │
        ▼
01_data_preparation/
Point-cloud preparation and preprocessing
        │
        ├──────────────► 02_synthetic_data/
        │               Synthetic terrain + structure generation
        │
        ▼
03_swin3d/
Dataset loader → Swin3D → training experiments
        │
        ▼
04_evaluation/
Standard segmentation metrics + archaeological vicinity / AVD evaluation
        │
        ▼
05_visualization/
Point-cloud and prediction visualization
```

---

## Repository structure

```text
3D-Point-Cloud-Deep-Learning-for-Maya-Archaeological-Detection/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── data/
│   ├── README.md
│   └── sample_synthetic_output/
│
├── environment/
│   └── README.md
│
├── docs/
│   ├── README.md
│   ├── REPOSITORY_REORGANIZATION.md
│   ├── CURRENT_FILE_MIGRATION_MAP.md
│   ├── EXPERIMENT_INDEX.md
│   └── FILE_INVENTORY.txt
│
├── 01_data_preparation/
│   ├── README.md
│   └── data_preparation.ipynb
│
├── 02_synthetic_data/
│   ├── README.md
│   ├── maya_synthetic_data_generator.ipynb
│   ├── maya_synthetic_data_generator.py
│   └── maya_synthetic_data_generator.cs
│
├── 03_swin3d/
│   ├── README.md
│   ├── configs/
│   │   └── swin3d_small.py
│   ├── datasets/
│   │   └── belize_laz_dataset.py
│   ├── models/
│   │   └── swin3d_model.py
│   ├── losses/
│   │   ├── generalized_dice.py
│   │   ├── soft_avd.py
│   │   └── swin3d_archaeological_vicinity_loss.py
│   ├── training/
│   │   ├── train_swin3d.py
│   │   └── notebooks/
│   │       ├── Training_Generalized_Dice.ipynb
│   │       ├── Training_Dice_SoftAVD.ipynb
│   │       └── Training_Dice_Vicinity.ipynb
│   ├── dataset_splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── jobs/
│
├── 04_evaluation/
│   ├── README.md
│   ├── standard_metrics/
│   │   ├── evaluate.py
│   │   └── notebooks/
│   │       └── Evaluate.ipynb
│   └── avd_metric/
│       ├── archaeological_vicinity_metric.py
│       ├── evaluate_avd.py
│       └── notebooks/
│           └── evaluate_archaeological_metric_cached_predictions.ipynb
│
└── 05_visualization/
    ├── README.md
    ├── vis.py
    └── Visualization.ipynb
```

---

## 1. Data preparation

The canonical data-preparation workflow is located at:

```text
01_data_preparation/data_preparation.ipynb
```

This notebook contains the project-specific preprocessing used before Swin3D training. Large raw LAS/LAZ datasets are intentionally not committed to the repository.

The Swin3D dataset loader keeps only the three project classes and performs this required source-to-model remapping:

```python
2  -> 0   # Ground
6  -> 1   # Structure
27 -> 2   # Plazuela
```

The current model configuration uses four input channels:

```text
X, Y, Z, slope
```

The default configuration in `03_swin3d/configs/swin3d_small.py` specifies three output classes, a voxel size of `0.2`, and a default maximum of `65,000` points/voxels per sample.

> Large datasets, processed tiles, cached predictions, and model checkpoints should be stored outside Git or managed through an appropriate large-file storage workflow.

### Public dataset splits and archaeological location privacy

The committed `03_swin3d/dataset_splits/{train,val,test}.csv` files contain opaque `tile_id` values rather than internal filesystem paths or coordinate-bearing source filenames. Authorized researchers must provide a separate `tile_id,path` manifest outside the public repository. Set its location with `MAYA_TILE_MANIFEST` or pass `tile_manifest=` to the dataloader.

This design preserves the published train/validation/test membership while avoiding accidental disclosure of machine-specific storage paths or potentially sensitive archaeological tile coordinates.

---

## 2. Synthetic Maya data generation

The repository preserves all three versions of the supplied synthetic-data generator:

```text
02_synthetic_data/
├── maya_synthetic_data_generator.ipynb
├── maya_synthetic_data_generator.py
└── maya_synthetic_data_generator.cs
```

The original C# implementation was designed for the Rhino/Grasshopper workflow. The Python implementation reproduces the same overall process with NumPy-based geometry and can be used without Rhino.

The generator creates synthetic terrain containing:

- a base point grid,
- randomized topographic variation,
- archaeological structures,
- Gaussian spatial profiles,
- coordinate noise,
- slope values, and
- semantic class labels.

A small example output is included in:

```text
data/sample_synthetic_output/
```

---

## 3. Swin3D semantic segmentation

The model-development code is contained in `03_swin3d/`.

### Dataset loader

```text
03_swin3d/datasets/belize_laz_dataset.py
```

The loader handles LAS/LAZ point-cloud input, class filtering/remapping, slope-based features, point limiting, batching, and the structures needed by the Swin3D pipeline.

### Model

```text
03_swin3d/models/swin3d_model.py
```

The project uses a Swin3D semantic-segmentation workflow for three-class archaeological point-cloud classification.

### Configuration

```text
03_swin3d/configs/swin3d_small.py
```

Important configuration values should be reviewed before running an experiment, particularly:

- dataset paths,
- voxel size,
- voxel/point maximum,
- number of classes,
- input features,
- batch size,
- optimizer settings, and
- checkpoint/output paths.

---

## Training experiments

Three active training notebooks are retained:

| Notebook | Purpose |
|---|---|
| `Training_Generalized_Dice.ipynb` | Experiment 1 — Generalized Dice baseline |
| `Training_Dice_SoftAVD.ipynb` | Generalized Dice with differentiable Soft-AVD supervision |
| `Training_Dice_Vicinity.ipynb` | Generalized Dice with archaeological-vicinity supervision |

The corresponding reusable loss implementations are stored in:

```text
03_swin3d/losses/
├── generalized_dice.py
├── soft_avd.py
└── swin3d_archaeological_vicinity_loss.py
```

`Training_old.ipynb` was intentionally excluded from the reorganized repository.

For command-line or HPC training, see:

```text
03_swin3d/training/
03_swin3d/jobs/
```

The `jobs/` directory contains the retained shell and SLURM submission files used for cluster execution.

---

## 4. Evaluation

Evaluation is divided into two complementary workflows.

### Standard semantic-segmentation metrics

```text
04_evaluation/standard_metrics/
├── evaluate.py
└── notebooks/Evaluate.ipynb
```

Use this workflow for conventional point-wise semantic-segmentation evaluation.

### Archaeological Vicinity Detection / AVD

Exact point-wise overlap can be overly strict for archaeological interpretation. A prediction may still be useful when it identifies an archaeological feature close to the true location even if every predicted point does not overlap the reference label exactly.

The project therefore includes a spatial archaeological-vicinity evaluation workflow:

```text
04_evaluation/avd_metric/
├── archaeological_vicinity_metric.py
├── evaluate_avd.py
└── notebooks/
    └── evaluate_archaeological_metric_cached_predictions.ipynb
```

The selected notebook uses **cached model predictions**, allowing the archaeological metric to be recalculated without rerunning the full neural-network inference pipeline for every metric experiment.

This workflow separately supports the project's archaeological classes:

- Structure
- Plazuela

and evaluates predictions based on spatial proximity in addition to semantic identity.

---

## 5. Visualization

Visualization tools are stored in:

```text
05_visualization/
├── vis.py
└── Visualization.ipynb
```

These files support inspection of the point-cloud data, semantic labels, and model predictions.

Visualization is especially useful for understanding failure modes that are difficult to interpret from aggregate metrics alone, such as:

- missed structures,
- fragmented archaeological predictions,
- predictions adjacent to the true feature,
- confusion between Structure and Plazuela, and
- excessive archaeological predictions over Ground.

---

## Dataset splits

The retained train/validation/test split files are stored in:

```text
03_swin3d/dataset_splits/
├── train.csv
├── val.csv
└── test.csv
```

These split files should be used consistently when comparing training experiments so that performance differences are not caused by changing the evaluation data.

---

## Environment and dependencies

The supplied project files did not include one canonical environment specification, so the repository currently documents the environment separately in:

```text
environment/README.md
```

The codebase uses a Python/PyTorch point-cloud deep-learning stack and includes dependencies associated with Swin3D, sparse 3D processing, LAS/LAZ input, numerical computation, notebooks, and visualization.

Before reproducing the final experiments, create a project-specific `environment.yml`, `requirements.txt`, or container definition from the exact environment used for training.

Typical components of the environment include:

```text
Python
PyTorch
NumPy
laspy / LAZ support
MinkowskiEngine or the sparse-tensor backend required by the Swin3D implementation
Jupyter
Matplotlib
scikit-learn / scientific Python utilities used by the evaluation code
```

Exact versions should be taken from the final working training environment rather than inferred from this README.

---

## Recommended reproduction order

For a new experiment, the intended order is:

```text
1. Prepare LAS/LAZ data
       ↓
2. Verify train / validation / test CSVs
       ↓
3. Configure Swin3D parameters
       ↓
4. Train a baseline or archaeology-aware loss variant
       ↓
5. Save model checkpoints and predictions
       ↓
6. Run standard semantic evaluation
       ↓
7. Cache predictions
       ↓
8. Run archaeological vicinity / AVD evaluation
       ↓
9. Visualize representative predictions and failure cases
```

Synthetic data generation is an optional parallel workflow that can be used for controlled experimentation and data-development tasks.

---

## Repository organization notes

This repository was reorganized from the working research files to separate reusable source code from experiment artifacts.

Key decisions include:

- `Data Prep(5).ipynb` became `01_data_preparation/data_preparation.ipynb`.
- All supplied Maya synthetic-data generator formats were preserved.
- Active Swin3D training files were retained.
- `Training_old.ipynb` was removed.
- The cached-predictions archaeological evaluation notebook was selected as the canonical AVD evaluation workflow.
- Jupyter checkpoints, Python caches, temporary files, and historical HPC output logs were removed.
- Large raw datasets, trained model weights, and cached prediction arrays are intentionally not stored in Git.

For the complete source-to-destination mapping, see:

```text
docs/CURRENT_FILE_MIGRATION_MAP.md
```

For a summary of the retained experiments, see:

```text
docs/EXPERIMENT_INDEX.md
```

---

## Model weights and large artifacts

Model checkpoints are not included in this repository unless explicitly added later. Recommended storage layout for local use is:

```text
artifacts/
├── checkpoints/
├── predictions/
└── evaluation_outputs/
```

These directories should remain ignored by Git when they contain large generated files.

---

## Citation

If this repository is used in academic work, please cite the project using the metadata in:

```text
CITATION.cff
```
If you use the code, workflows, or datasets from this repository in academic research, please cite:

Newton, David, Heather Richards-Rissetto, Amy E. Thompson,
Sanyam Agrawal, Shubham Saraf, and Keith M. Prufer.
2026. *3D Point Cloud Deep Learning for Maya Archaeological
Detection*. GitHub.
https://github.com/dnewton78/3D-Point-Cloud-Deep-Learning-for-Maya-Archaeological-Detection

Citation metadata is also provided in the [CITATION.cff](CITATION.cff)
file. You can use GitHub's "Cite this repository" feature to
generate formatted citations.

---

## License

The original software developed for this project is released under
the MIT License. See the [LICENSE](LICENSE) file for details.

Third-party software components remain subject to their
respective licenses and copyright notices.

Datasets and archaeological information, where provided,
may be subject to separate licensing and access conditions.

---

## Documentation

Additional repository documentation is available in `docs/`:

- `REPOSITORY_REORGANIZATION.md` — rationale and organization decisions.
- `CURRENT_FILE_MIGRATION_MAP.md` — original file to new file mapping.
- `EXPERIMENT_INDEX.md` — retained training/evaluation experiment overview.
- `FILE_INVENTORY.txt` — repository file inventory.

---

## Project status

This repository represents an organized research codebase rather than a packaged software library. Some paths and environment-specific settings may need to be updated before running on a new workstation or HPC system.

The recommended next reproducibility step is to add the exact final software environment and document the final trained checkpoint(s), experiment configuration(s), and evaluation outputs used for the reported results.
