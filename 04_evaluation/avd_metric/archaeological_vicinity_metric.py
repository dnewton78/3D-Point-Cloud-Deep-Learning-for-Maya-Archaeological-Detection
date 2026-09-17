"""
Archaeological Vicinity Detection (AVD) metric for 3D point-cloud
semantic segmentation.

Designed for evaluating Swin3D-S predictions for:
    0 = Ground
    1 = Structure
    2 = Plazuela

The metric is computed in the horizontal XY plane. It rasterizes points into
fixed-size cells, assigns each cell to the nearest ground-truth feature, and
measures:

    vicinity precision:
        fraction of predicted archaeological cells within radius r of a
        ground-truth feature;

    feature detection recall:
        fraction of ground-truth features receiving at least
        min_evidence_cells predicted cells within radius r;

    AVD-F_beta:
        harmonic mean of vicinity precision and feature detection recall.

This is an evaluation metric, not a differentiable loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import (
    binary_closing,
    distance_transform_edt,
    generate_binary_structure,
    label as connected_components,
)


LabelCollection = Union[int, Sequence[int]]


@dataclass(frozen=True)
class ClassSpec:
    """Defines one semantic evaluation class or a combined discovery class."""

    name: str
    predicted_labels: LabelCollection
    target_labels: Optional[LabelCollection] = None

    def pred_tuple(self) -> Tuple[int, ...]:
        return _as_tuple(self.predicted_labels)

    def target_tuple(self) -> Tuple[int, ...]:
        labels = self.predicted_labels if self.target_labels is None else self.target_labels
        return _as_tuple(labels)


@dataclass(frozen=True)
class AVDConfig:
    """Configuration for Archaeological Vicinity Detection evaluation."""

    cell_size_m: float = 1.0
    radii_m: Tuple[float, ...] = (2.0, 5.0, 10.0)
    min_points_per_cell: int = 1
    min_evidence_cells: int = 3
    beta: float = 2.0
    ignore_label: Optional[int] = -100

    # Used only when explicit ground-truth feature IDs are unavailable.
    infer_instances_from_semantic_mask: bool = True
    connected_component_connectivity: int = 2  # 1=4-neighbor, 2=8-neighbor
    semantic_mask_closing_iterations: int = 0

    # Protects against accidental voxel-index coordinates or oversized tiles.
    max_grid_cells: int = 50_000_000

    def __post_init__(self) -> None:
        if self.cell_size_m <= 0:
            raise ValueError("cell_size_m must be greater than zero.")
        if not self.radii_m or any(r < 0 for r in self.radii_m):
            raise ValueError("radii_m must contain non-negative radii.")
        if self.min_points_per_cell < 1:
            raise ValueError("min_points_per_cell must be at least 1.")
        if self.min_evidence_cells < 1:
            raise ValueError("min_evidence_cells must be at least 1.")
        if self.beta <= 0:
            raise ValueError("beta must be greater than zero.")
        if self.connected_component_connectivity not in (1, 2):
            raise ValueError("connected_component_connectivity must be 1 or 2.")


@dataclass
class _Counts:
    predicted_cells: int = 0
    predicted_cells_near_gt: int = 0
    gt_features: int = 0
    detected_gt_features: int = 0

    def add(self, other: "_Counts") -> None:
        self.predicted_cells += int(other.predicted_cells)
        self.predicted_cells_near_gt += int(other.predicted_cells_near_gt)
        self.gt_features += int(other.gt_features)
        self.detected_gt_features += int(other.detected_gt_features)


def _as_tuple(value: LabelCollection) -> Tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    result = tuple(int(v) for v in value)
    if not result:
        raise ValueError("A class label collection cannot be empty.")
    return result


class ArchaeologicalVicinityMetric:
    """
    Dataset-level AVD metric accumulator.

    Call ``update`` once per independent scene/tile, then call ``compute``.
    Do not concatenate unrelated tiles before ``update`` because feature
    instances must not cross tile boundaries.
    """

    def __init__(
        self,
        class_specs: Sequence[ClassSpec],
        config: Optional[AVDConfig] = None,
    ) -> None:
        if not class_specs:
            raise ValueError("At least one ClassSpec is required.")
        names = [spec.name for spec in class_specs]
        if len(set(names)) != len(names):
            raise ValueError("ClassSpec names must be unique.")

        self.class_specs = tuple(class_specs)
        self.config = config or AVDConfig()
        self.reset()

    def reset(self) -> None:
        self._counts: Dict[Tuple[str, float], _Counts] = {
            (spec.name, float(radius)): _Counts()
            for spec in self.class_specs
            for radius in self.config.radii_m
        }

    def update(
        self,
        xyz_m: np.ndarray,
        predicted_labels: np.ndarray,
        target_labels: np.ndarray,
        gt_instance_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[float, Dict[str, float]]]:
        """
        Add one tile or scene.

        Parameters
        ----------
        xyz_m:
            [N, 3] or [N, 2]. XY coordinates must be in meters.
        predicted_labels:
            Point-level semantic predictions, shape [N].
        target_labels:
            Point-level ground-truth semantic labels, shape [N].
        gt_instance_ids:
            Optional ground-truth archaeological feature ID per point, shape
            [N]. Use a negative value for points without a feature ID. Explicit
            IDs are strongly recommended. If omitted, connected components are
            inferred separately for each semantic class.
        """
        xyz = np.asarray(xyz_m)
        pred = np.asarray(predicted_labels).reshape(-1)
        target = np.asarray(target_labels).reshape(-1)

        if xyz.ndim != 2 or xyz.shape[1] < 2:
            raise ValueError("xyz_m must have shape [N, 2] or [N, 3].")
        if xyz.shape[0] != pred.size or pred.size != target.size:
            raise ValueError("xyz_m, predictions, and targets must have equal length.")
        if xyz.shape[0] == 0:
            return {}

        if gt_instance_ids is not None:
            instance_ids = np.asarray(gt_instance_ids).reshape(-1)
            if instance_ids.size != target.size:
                raise ValueError("gt_instance_ids must match target_labels length.")
        else:
            instance_ids = None

        valid = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1])
        if self.config.ignore_label is not None:
            valid &= target != self.config.ignore_label

        xyz = xyz[valid]
        pred = pred[valid]
        target = target[valid]
        if instance_ids is not None:
            instance_ids = instance_ids[valid]

        if xyz.shape[0] == 0:
            return {}

        grid = self._make_grid(xyz[:, :2])
        flat_cells = grid["flat_cells"]
        height = grid["height"]
        width = grid["width"]
        num_cells = height * width

        scene_result: Dict[str, Dict[float, Dict[str, float]]] = {}

        for spec in self.class_specs:
            pred_point_mask = np.isin(pred, spec.pred_tuple())
            pred_counts = np.bincount(
                flat_cells[pred_point_mask], minlength=num_cells
            )
            pred_grid = (
                pred_counts.reshape(height, width)
                >= self.config.min_points_per_cell
            )
            predicted_cell_count = int(pred_grid.sum())

            target_point_mask = np.isin(target, spec.target_tuple())
            instance_grid, num_instances = self._make_instance_grid(
                flat_cells,
                target_point_mask,
                instance_ids,
                height,
                width,
            )

            scene_result[spec.name] = {}

            if num_instances > 0:
                distances_m, nearest_indices = distance_transform_edt(
                    instance_grid == 0,
                    sampling=(self.config.cell_size_m, self.config.cell_size_m),
                    return_indices=True,
                )
                nearest_instance = instance_grid[
                    nearest_indices[0], nearest_indices[1]
                ]
            else:
                distances_m = None
                nearest_instance = None

            for radius in self.config.radii_m:
                if num_instances > 0:
                    within_radius = distances_m <= (float(radius) + 1e-9)
                    near_pred_grid = pred_grid & within_radius
                    predicted_near_count = int(near_pred_grid.sum())

                    # Overlapping buffers cannot double count evidence:
                    # each cell is assigned to its nearest GT feature.
                    assigned = nearest_instance[near_pred_grid]
                    evidence_counts = np.bincount(
                        assigned.astype(np.int64),
                        minlength=num_instances + 1,
                    )[1:]
                    detected_count = int(
                        np.count_nonzero(
                            evidence_counts >= self.config.min_evidence_cells
                        )
                    )
                else:
                    predicted_near_count = 0
                    detected_count = 0

                counts = _Counts(
                    predicted_cells=predicted_cell_count,
                    predicted_cells_near_gt=predicted_near_count,
                    gt_features=num_instances,
                    detected_gt_features=detected_count,
                )
                self._counts[(spec.name, float(radius))].add(counts)
                scene_result[spec.name][float(radius)] = self._metrics_from_counts(
                    counts
                )

        return scene_result

    def compute(self) -> Dict[str, object]:
        """Return dataset-level per-class, per-radius, and macro results."""
        per_class: Dict[str, Dict[float, Dict[str, float]]] = {}
        macro_by_radius: Dict[float, float] = {}

        for spec in self.class_specs:
            per_class[spec.name] = {}
            for radius in self.config.radii_m:
                per_class[spec.name][float(radius)] = self._metrics_from_counts(
                    self._counts[(spec.name, float(radius))]
                )

        for radius in self.config.radii_m:
            values = [
                per_class[spec.name][float(radius)]["avd_fbeta"]
                for spec in self.class_specs
            ]
            values = [v for v in values if np.isfinite(v)]
            macro_by_radius[float(radius)] = (
                float(np.mean(values)) if values else float("nan")
            )

        return {
            "config": {
                "cell_size_m": self.config.cell_size_m,
                "radii_m": list(self.config.radii_m),
                "min_points_per_cell": self.config.min_points_per_cell,
                "min_evidence_cells": self.config.min_evidence_cells,
                "beta": self.config.beta,
            },
            "per_class": per_class,
            "macro_avd_fbeta": macro_by_radius,
        }

    def synchronize_between_processes(self) -> None:
        """Sum counts across initialized torch.distributed processes."""
        try:
            import torch
            import torch.distributed as dist
        except ImportError:
            return

        if not dist.is_available() or not dist.is_initialized():
            return

        keys = [
            (spec.name, float(radius))
            for spec in self.class_specs
            for radius in self.config.radii_m
        ]
        rows = []
        for key in keys:
            c = self._counts[key]
            rows.append(
                [
                    c.predicted_cells,
                    c.predicted_cells_near_gt,
                    c.gt_features,
                    c.detected_gt_features,
                ]
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor(rows, dtype=torch.long, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        for key, row in zip(keys, tensor.cpu().numpy()):
            self._counts[key] = _Counts(
                predicted_cells=int(row[0]),
                predicted_cells_near_gt=int(row[1]),
                gt_features=int(row[2]),
                detected_gt_features=int(row[3]),
            )

    def format_results(self, digits: int = 4) -> str:
        """Return a compact text table suitable for the Swin3D logger."""
        result = self.compute()
        lines = [
            (
                f"AVD metric: cell={self.config.cell_size_m:g} m, "
                f"minimum evidence={self.config.min_evidence_cells} cells, "
                f"F_beta beta={self.config.beta:g}"
            ),
            "class | radius_m | precision | recall | AVD-F_beta | "
            "near/total predicted cells | detected/total features",
        ]

        for spec in self.class_specs:
            for radius in self.config.radii_m:
                m = result["per_class"][spec.name][float(radius)]
                lines.append(
                    f"{spec.name} | {radius:g} | "
                    f"{_fmt(m['vicinity_precision'], digits)} | "
                    f"{_fmt(m['feature_recall'], digits)} | "
                    f"{_fmt(m['avd_fbeta'], digits)} | "
                    f"{int(m['predicted_cells_near_gt'])}/"
                    f"{int(m['predicted_cells'])} | "
                    f"{int(m['detected_gt_features'])}/"
                    f"{int(m['gt_features'])}"
                )

        for radius, value in result["macro_avd_fbeta"].items():
            lines.append(
                f"macro AVD-F_beta @ {radius:g} m = {_fmt(value, digits)}"
            )
        return "\n".join(lines)

    def _make_grid(self, xy_m: np.ndarray) -> Dict[str, object]:
        margin = max(float(r) for r in self.config.radii_m) + self.config.cell_size_m

        xy_min = np.floor(
            (np.min(xy_m, axis=0) - margin) / self.config.cell_size_m
        ) * self.config.cell_size_m
        xy_max = np.ceil(
            (np.max(xy_m, axis=0) + margin) / self.config.cell_size_m
        ) * self.config.cell_size_m

        width = int((xy_max[0] - xy_min[0]) / self.config.cell_size_m) + 1
        height = int((xy_max[1] - xy_min[1]) / self.config.cell_size_m) + 1
        num_cells = height * width

        if num_cells > self.config.max_grid_cells:
            raise MemoryError(
                f"AVD raster would contain {num_cells:,} cells "
                f"({height} x {width}), exceeding max_grid_cells="
                f"{self.config.max_grid_cells:,}. Confirm coordinates are in "
                "meters, process smaller tiles, or increase cell_size_m."
            )

        cols = np.floor(
            (xy_m[:, 0] - xy_min[0]) / self.config.cell_size_m
        ).astype(np.int64)
        rows = np.floor(
            (xy_m[:, 1] - xy_min[1]) / self.config.cell_size_m
        ).astype(np.int64)

        cols = np.clip(cols, 0, width - 1)
        rows = np.clip(rows, 0, height - 1)

        return {
            "height": height,
            "width": width,
            "flat_cells": rows * width + cols,
        }

    def _make_instance_grid(
        self,
        flat_cells: np.ndarray,
        target_point_mask: np.ndarray,
        target_instance_ids: Optional[np.ndarray],
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, int]:
        num_cells = height * width

        if target_instance_ids is not None:
            valid = target_point_mask & np.isfinite(target_instance_ids)
            valid &= target_instance_ids >= 0
            if np.any(valid):
                return self._instance_grid_from_ids(
                    flat_cells[valid],
                    target_instance_ids[valid].astype(np.int64),
                    height,
                    width,
                )

        if not self.config.infer_instances_from_semantic_mask:
            return np.zeros((height, width), dtype=np.int32), 0

        gt_counts = np.bincount(
            flat_cells[target_point_mask], minlength=num_cells
        ).reshape(height, width)
        gt_mask = gt_counts > 0

        if self.config.semantic_mask_closing_iterations > 0:
            gt_mask = binary_closing(
                gt_mask,
                structure=generate_binary_structure(
                    2, self.config.connected_component_connectivity
                ),
                iterations=self.config.semantic_mask_closing_iterations,
            )

        structure = generate_binary_structure(
            2, self.config.connected_component_connectivity
        )
        instance_grid, num_instances = connected_components(
            gt_mask, structure=structure
        )
        return instance_grid.astype(np.int32, copy=False), int(num_instances)

    @staticmethod
    def _instance_grid_from_ids(
        flat_cells: np.ndarray,
        instance_ids: np.ndarray,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, int]:
        """Rasterize IDs, selecting the modal ID when a cell contains several."""
        pairs = np.column_stack((flat_cells, instance_ids))
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

        order = np.lexsort((-counts, unique_pairs[:, 0]))
        sorted_pairs = unique_pairs[order]
        sorted_cells = sorted_pairs[:, 0]
        first_for_cell = np.r_[True, sorted_cells[1:] != sorted_cells[:-1]]
        modal_pairs = sorted_pairs[first_for_cell]

        original_ids = np.unique(modal_pairs[:, 1])
        compact_map = {
            int(original_id): compact_id
            for compact_id, original_id in enumerate(original_ids, start=1)
        }

        grid_flat = np.zeros(height * width, dtype=np.int32)
        compact_values = np.fromiter(
            (compact_map[int(v)] for v in modal_pairs[:, 1]),
            dtype=np.int32,
            count=modal_pairs.shape[0],
        )
        grid_flat[modal_pairs[:, 0].astype(np.int64)] = compact_values
        return grid_flat.reshape(height, width), len(original_ids)

    def _metrics_from_counts(self, counts: _Counts) -> Dict[str, float]:
        if counts.predicted_cells > 0:
            precision = counts.predicted_cells_near_gt / counts.predicted_cells
        elif counts.gt_features > 0:
            precision = 0.0
        else:
            precision = float("nan")

        recall = (
            counts.detected_gt_features / counts.gt_features
            if counts.gt_features > 0
            else float("nan")
        )

        if np.isfinite(precision) and np.isfinite(recall):
            beta_sq = self.config.beta ** 2
            denominator = beta_sq * precision + recall
            fbeta = (
                (1.0 + beta_sq) * precision * recall / denominator
                if denominator > 0
                else 0.0
            )
        else:
            fbeta = float("nan")

        return {
            "vicinity_precision": float(precision),
            "feature_recall": float(recall),
            "avd_fbeta": float(fbeta),
            "predicted_cells": float(counts.predicted_cells),
            "predicted_cells_near_gt": float(counts.predicted_cells_near_gt),
            "gt_features": float(counts.gt_features),
            "detected_gt_features": float(counts.detected_gt_features),
        }


def _fmt(value: float, digits: int) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"
