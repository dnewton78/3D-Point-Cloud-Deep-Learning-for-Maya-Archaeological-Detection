# Current File Migration Map

| Original file/location | New location | Action |
|---|---|---|
| `Data Prep(5).ipynb` | `01_data_preparation/data_preparation.ipynb` | Kept; canonical data-prep notebook |
| `Maya_Synthetic_Data_Generator(1).ipynb` | `02_synthetic_data/maya_synthetic_data_generator.ipynb` | Kept |
| `maya_synthetic_data_generator(1).py` | `02_synthetic_data/maya_synthetic_data_generator.py` | Kept |
| `Maya Project_Synthetic Data Generator(1).cs` | `02_synthetic_data/maya_synthetic_data_generator.cs` | Kept |
| `configs/swin3d_small.py` | `03_swin3d/configs/swin3d_small.py` | Kept |
| `datasets/*` source | `03_swin3d/datasets/` | Kept |
| `models/swin3d_model.py` | `03_swin3d/models/swin3d_model.py` | Kept |
| `losses/*.py` | `03_swin3d/losses/` | Kept |
| `Training_Generalized_Dice.ipynb` | `03_swin3d/training/notebooks/Training_Generalized_Dice.ipynb` | Kept |
| `Training_Dice_SoftAVD.ipynb` | `03_swin3d/training/notebooks/Training_Dice_SoftAVD.ipynb` | Kept |
| `Training_Dice_Vicinity.ipynb` | `03_swin3d/training/notebooks/Training_Dice_Vicinity.ipynb` | Kept |
| `Training_old.ipynb` | — | Excluded by request |
| root `train_swin3d.py` | `03_swin3d/training/train_swin3d.py` | Kept |
| `scripts/train_swin3d.py` | `03_swin3d/training/train_swin3d.py` | Kept; distinct file |
| `dataset_splits/*.csv` | `03_swin3d/dataset_splits/` | Kept |
| active `jobs/*` scripts | `03_swin3d/jobs/` | Kept |
| `evaluate.py` | `04_evaluation/standard_metrics/evaluate.py` | Kept |
| `Evaluate.ipynb` | `04_evaluation/standard_metrics/notebooks/Evaluate.ipynb` | Kept |
| `evaluation/evaluate_avd.py` | `04_evaluation/avd_metric/evaluate_avd.py` | Kept |
| `archaeological_vicinity_metric.py` | `04_evaluation/avd_metric/archaeological_vicinity_metric.py` | Kept |
| `evaluate_archaeological_metric_cached_predictions.ipynb` | `04_evaluation/avd_metric/notebooks/` | Kept; selected archaeological evaluation |
| other archaeological evaluation notebooks | — | Excluded in favor of cached-predictions version |
| `visualization/vis.py` | `05_visualization/vis.py` | Kept |
| `Visualization.ipynb` | `05_visualization/Visualization.ipynb` | Kept |
| `synthetic_output/0maya_synthetic.csv` | `data/sample_synthetic_output/0maya_synthetic.csv` | Kept as small example |
| `.ipynb_checkpoints/`, `__pycache__/`, old job logs | — | Excluded generated artifacts |
