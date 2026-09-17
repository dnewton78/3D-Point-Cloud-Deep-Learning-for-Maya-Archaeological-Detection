# Repository Reorganization

The supplied research files were reorganized by workflow rather than by execution history.

## Rules applied

1. Use the supplied `Data Prep(5).ipynb` as the canonical data-preparation notebook.
2. Preserve every supplied Maya synthetic-data generator format: Jupyter notebook, Python, and C#.
3. Preserve the Swin3D source components required for training: config, dataset loader, model wrapper, loss functions, splits, active training scripts/notebooks, and HPC job files.
4. Exclude `Training_old.ipynb` and its checkpoint copy.
5. Keep the standard evaluation workflow.
6. For the archaeological vicinity/AVD evaluation, retain the cached-predictions notebook and its supporting metric/evaluation Python modules, rather than the older alternate notebooks.
7. Keep the visualization module and notebook.
8. Exclude generated `.ipynb_checkpoints`, `__pycache__`, historical `.out/.err` logs, and scratch files.

No source code was rewritten during this reorganization; files were moved/renamed for clarity.
