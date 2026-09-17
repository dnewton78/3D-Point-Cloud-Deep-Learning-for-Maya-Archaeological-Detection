from pathlib import Path
import os

import laspy
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader


CLASS_MAP = {
    2: 0,   # Ground
    6: 1,   # Structure
    27: 2,  # Plazuela
}

CLASS_NAMES = [
    "Ground",
    "Structure",
    "Plazuela",
]

NUM_CLASSES = 3


class BelizeLAZDataset(Dataset):
    """
    Dataset for Belize LAZ point-cloud tiles.

    Returns:
        coord:
            Float tensor with shape [N, 3].
            Contains localized X, Y, and Z coordinates.

        feat:
            Float tensor with shape [N, 1].
            Contains normalized slope.

        segment:
            Long tensor with shape [N].
            Contains mapped class labels 0, 1, and 2.

        path:
            Original LAZ file path.
    """
    
    def __init__(
        self,
        file_paths,
        augment=False,
        voxel_max=65_000,
        augmentation_ratio=1,
    ):
        self.file_paths = [
            Path(path)
            for path in file_paths
        ]
    
        self.augment = augment
        self.voxel_max = voxel_max
        self.augmentation_ratio = (
            augmentation_ratio if augment else 1
        )
    
        if len(self.file_paths) == 0:
            raise ValueError(
                "No LAZ files were provided to BelizeLAZDataset."
            )

    def __len__(self):
        return (
            len(self.file_paths)
            * self.augmentation_ratio
        )

    def __getitem__(self, index):
        file_index = index // self.augmentation_ratio
        path = self.file_paths[file_index]

        las = laspy.read(path)

        # ====================================================
        # XYZ coordinates
        # ====================================================

        xyz = np.column_stack(
            [
                np.asarray(las.x),
                np.asarray(las.y),
                np.asarray(las.z),
            ]
        ).astype(np.float32)

        # Preserve projected coordinates in meters for AVD evaluation.
        xyz_metric = xyz.copy().astype(np.float64)

        # ====================================================
        # Slope
        # ====================================================

        if "slope" not in list(las.point_format.extra_dimension_names):
            raise ValueError(
                f"The LAZ file does not contain a slope field: {path}"
            )

        slope = np.asarray(
            las.slope,
            dtype=np.float32,
        )

        # Normalize approximately from 090 degrees to 01.
        slope = slope / 90.0

        # Prevent values outside the expected range.
        slope = np.clip(
            slope,
            0.0,
            1.0,
        )

        # ====================================================
        # Labels
        # ====================================================

        raw_labels = np.asarray(
            las.classification,
            dtype=np.int64,
        )

        labels = np.full(
            raw_labels.shape,
            fill_value=-1,
            dtype=np.int64,
        )

        for original_label, mapped_label in CLASS_MAP.items():
            labels[raw_labels == original_label] = mapped_label

        # Keep only Ground, Structure, and Plazuela points.
        valid = labels >= 0

        xyz = xyz[valid]
        xyz_metric = xyz_metric[valid]
        slope = slope[valid]
        labels = labels[valid]

        if xyz.shape[0] == 0:
            raise ValueError(
                f"No valid class 2, 6, or 27 points were found in: {path}"
            )

        # ====================================================
        # Localize coordinates
        # ====================================================

        xyz = xyz.copy()

        xyz[:, 0] -= xyz[:, 0].min()
        xyz[:, 1] -= xyz[:, 1].min()
        xyz[:, 2] -= xyz[:, 2].min()

        # ====================================================
        # Limit number of points
        # ====================================================

        xyz, xyz_metric, slope, labels = self.limit_points(
            xyz=xyz,
            xyz_metric=xyz_metric,
            slope=slope,
            labels=labels,
        )

        # ====================================================
        # Data augmentation
        # ====================================================

        if self.augment:
            xyz = self.apply_augmentation(xyz)

        # ====================================================
        # Features
        # ====================================================

        # The model combines:
        #
        # coords    = X, Y, Z
        # features  = slope
        #
        # Final model input = X, Y, Z, slope = 4 channels.
        features = slope.reshape(-1, 1).astype(np.float32)

        return {
            "coord": torch.from_numpy(
                np.ascontiguousarray(xyz)
            ).float(),

            "metric_coord": torch.from_numpy(
                np.ascontiguousarray(xyz_metric)
            ).double(),

            "feat": torch.from_numpy(
                np.ascontiguousarray(features)
            ).float(),

            "segment": torch.from_numpy(
                np.ascontiguousarray(labels)
            ).long(),

            "path": str(path),
        }

    def limit_points(
        self,
        xyz,
        xyz_metric,
        slope,
        labels,
    ):
        """
        Limit each tile to voxel_max points while keeping all arrays aligned.
    
        Training:
            Randomly samples points.
    
        Validation/test:
            Uses evenly spaced indices for reproducible evaluation.
        """
    
        number_of_points = xyz.shape[0]
    
        if (
            self.voxel_max is None
            or number_of_points <= self.voxel_max
        ):
            return (
                xyz,
                xyz_metric,
                slope,
                labels,
            )
    
        if self.augment:
            selected_indices = np.random.choice(
                number_of_points,
                size=self.voxel_max,
                replace=False,
            )
        else:
            selected_indices = np.linspace(
                0,
                number_of_points - 1,
                num=self.voxel_max,
                dtype=np.int64,
            )
    
        xyz = xyz[selected_indices]
        xyz_metric = xyz_metric[selected_indices]
        slope = slope[selected_indices]
        labels = labels[selected_indices]
    
        return (
            xyz,
            xyz_metric,
            slope,
            labels,
        )

    def apply_augmentation(self, xyz):
        xyz = xyz.copy()
    
        # ====================================================
        # 1. Jitter -- ALWAYS applied
        # ====================================================
    
        jitter = np.random.normal(
            loc=0.0,
            scale=0.01,
            size=xyz.shape,
        ).astype(np.float32)
    
        jitter = np.clip(
            jitter,
            -0.05,
            0.05,
        )
    
        xyz += jitter
    
        # ====================================================
        # 2. X + Y flip -- 50% probability
        # ====================================================
    
        if np.random.random() < 0.5:
            xyz[:, 0] *= -1.0
            xyz[:, 1] *= -1.0
    
        # ====================================================
        # 3. Rotation -- 50% probability
        # ====================================================
    
        if np.random.random() < 0.5:
    
            angle_degrees = np.random.choice(
                [90, 180, 270]
            )
    
            angle = np.deg2rad(
                angle_degrees
            )
    
            cosine = np.cos(angle)
            sine = np.sin(angle)
    
            rotation = np.array(
                [
                    [cosine, -sine, 0.0],
                    [sine, cosine, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
    
            xyz = xyz @ rotation.T
    
        # ====================================================
        # Shift back to non-negative local coordinates
        # ====================================================
    
        xyz -= xyz.min(
            axis=0,
            keepdims=True,
        )
    
        return xyz.astype(np.float32)


def swin3d_collate_fn(batch):
    coords_list = []
    metric_coords_list = []
    features_list = []
    labels_list = []
    paths = []
    offsets = []

    cumulative_points = 0

    for sample in batch:
        coords = sample["coord"]
        metric_coords = sample["metric_coord"]
        features = sample["feat"]
        labels = sample["segment"]

        number_of_points = coords.shape[0]

        if metric_coords.shape[0] != number_of_points:
            raise ValueError(
                "Metric coordinate count does not match coordinates."
            )

        if features.shape[0] != number_of_points:
            raise ValueError(
                "Feature count does not match coordinates."
            )

        if labels.shape[0] != number_of_points:
            raise ValueError(
                "Label count does not match coordinates."
            )

        coords_list.append(coords)
        metric_coords_list.append(metric_coords)
        features_list.append(features)
        labels_list.append(labels)
        paths.append(sample["path"])

        cumulative_points += number_of_points
        offsets.append(cumulative_points)

    return (
        torch.cat(coords_list, dim=0),
        torch.cat(features_list, dim=0),
        torch.tensor(offsets, dtype=torch.long),
        torch.cat(labels_list, dim=0),
        paths,
        torch.cat(metric_coords_list, dim=0),
    )


def read_split_csv(csv_path, tile_manifest=None):
    """
    Resolve a public split CSV to LAZ file paths.

    Preferred public format:
        tile_id
        tile_0001
        tile_0002

    The corresponding ``tile_id,path`` mapping must be supplied separately via
    ``tile_manifest`` (or, through ``create_dataloaders``, MAYA_TILE_MANIFEST).
    This keeps machine-specific paths and potentially sensitive source filenames
    out of the public repository.

    A legacy ``path`` column is still supported for private/local split files.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {csv_path}")

    split_dataframe = pd.read_csv(csv_path)

    if "path" in split_dataframe.columns:
        file_paths = [Path(path) for path in split_dataframe["path"].tolist()]
    elif "tile_id" in split_dataframe.columns:
        if tile_manifest is None:
            raise ValueError(
                f"{csv_path} contains public tile IDs. Provide a private "
                "tile_id-to-path manifest using tile_manifest=... or the "
                "MAYA_TILE_MANIFEST environment variable."
            )

        manifest_path = Path(tile_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Tile manifest does not exist: {manifest_path}")

        manifest = pd.read_csv(manifest_path)
        required = {"tile_id", "path"}
        if not required.issubset(manifest.columns):
            raise ValueError(
                f"{manifest_path} must contain columns: tile_id,path"
            )

        if manifest["tile_id"].duplicated().any():
            raise ValueError(f"Duplicate tile_id values found in {manifest_path}.")

        path_by_id = dict(zip(manifest["tile_id"], manifest["path"]))
        missing_ids = [
            tile_id for tile_id in split_dataframe["tile_id"].tolist()
            if tile_id not in path_by_id
        ]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise KeyError(
                f"{len(missing_ids)} tile IDs from {csv_path} are missing from "
                f"{manifest_path}. First missing IDs: {preview}"
            )

        file_paths = [
            Path(path_by_id[tile_id])
            for tile_id in split_dataframe["tile_id"].tolist()
        ]
    else:
        raise ValueError(
            f"{csv_path} must contain either a 'tile_id' column (public format) "
            "or a 'path' column (legacy/private format)."
        )

    missing_files = [path for path in file_paths if not path.exists()]
    if missing_files:
        preview = "\n".join(str(path) for path in missing_files[:5])
        raise FileNotFoundError(
            f"{len(missing_files)} resolved files from {csv_path} do not exist. "
            f"First missing paths:\n{preview}"
        )

    return file_paths

def create_dataloaders(
    split_dir="dataset_splits",
    tile_manifest=None,
    batch_size=8,
    voxel_max=65_000,
    num_workers=4,
    augmentation_ratio=3,
):
    """
    Create train, validation, and test DataLoaders.
    """

    split_dir = Path(split_dir)
    if tile_manifest is None:
        tile_manifest = os.environ.get("MAYA_TILE_MANIFEST")

    train_files = read_split_csv(
        split_dir / "train.csv", tile_manifest=tile_manifest
    )

    validation_files = read_split_csv(
        split_dir / "val.csv", tile_manifest=tile_manifest
    )

    test_files = read_split_csv(
        split_dir / "test.csv", tile_manifest=tile_manifest
    )

    train_dataset = BelizeLAZDataset(
        file_paths=train_files,
        augment=True,
        voxel_max=voxel_max,
        augmentation_ratio=augmentation_ratio,
    )

    validation_dataset = BelizeLAZDataset(
        file_paths=validation_files,
        augment=False,
        voxel_max=voxel_max,
    )

    test_dataset = BelizeLAZDataset(
        file_paths=test_files,
        augment=False,
        voxel_max=voxel_max,
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": swin3d_collate_fn,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )