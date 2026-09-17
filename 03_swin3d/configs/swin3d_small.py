TRAIN_CONFIG = {
    # Dataset
    "dataset": "BelizeSlopeValue_plz_str_grd_100m_06mslope",
    "num_classes": 3,
    "input_features": ["x", "y", "z", "slope"],
    # Private tile_id -> path manifest. Prefer MAYA_TILE_MANIFEST env var.
    "tile_manifest": None,

    # Data processing
    "voxel_size": 0.2,
    "voxel_max": 65_000,
    "batch_size": 16,

    # Augmentation
    "augmentations": [
        "jitter",
        "flip",
        "rotation",
    ],
    "augmentation_ratio": 6,

    # Training
    "trainable_layers": "all",
    "max_epochs": 100,
    "warmup_epochs": 10,
    "early_stopping_patience": 10,

    # Loss
    "loss_function": "generalized_dice",

    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 3e-5,
    "weight_decay": 0.001,

    # Scheduler
    "scheduler": "cosine",
}