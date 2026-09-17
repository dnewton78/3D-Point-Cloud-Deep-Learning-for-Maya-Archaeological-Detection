# Data

Large source datasets are intentionally not committed to this repository.

Expected project inputs include point-cloud data such as LAS/LAZ files and the train/validation/test paths referenced by `03_swin3d/dataset_splits/`.

`sample_synthetic_output/` contains the small synthetic CSV that was present in the supplied model-training archive. Raw data, processed data, cached predictions, and model checkpoints should remain outside Git or be managed with an appropriate large-file/data-storage workflow.

## Archaeological source data

Raw/processed archaeological LAZ files are not distributed here. Public split files use opaque tile IDs. Authorized path mappings should be stored outside the public repository and must be reviewed for location-disclosure restrictions before sharing.
