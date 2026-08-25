"""Simplified archaeological vicinity loss for Swin3D-S semantic segmentation.

Designed for the official Swin3D task interface:

    logits = model(feat, coord, batch)

Inputs
------
logits:
    Float tensor [N, C] containing per-voxel class logits.
target:
    Long tensor [N] containing semantic labels.
coord:
    Float tensor [N, 3] passed to Swin3D. In the official task code these
    coordinates have already been divided by the dataset voxel size.
batch:
    Long tensor [N] assigning each point/voxel to a crop in the batch.

The auxiliary loss:
1. Aggregates P(plazuela) + P(structure) on a coarse horizontal grid.
2. Rewards archaeological probability within a fixed vicinity of true
   structure/plazuela cells.
3. Penalizes confident archaeological probability outside that vicinity.

It intentionally does not perform connected-component extraction, pairwise
distance calculations, or instance matching. Use the instance-aware AVF metric
for whole-scene evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass
class Swin3DVicinityLossConfig:
    """Parameters for :class:`Swin3DSimpleVicinityLoss`.

    ``coordinate_scale_m`` converts the coordinates passed to Swin3D back to
    meters. In the official Swin3D task repository, use the dataset
    ``voxel_size`` here because data preparation divides coordinates by that
    value.

    ``horizontal_axes`` refers to the coordinate ordering *at the loss call*.
    For ordinary z-up XYZ coordinates use ``(0, 1)``. If ``yz_shift=True`` has
    changed XYZ to XZY before the model call, use ``(0, 2)``.
    """

    ground_label: int = 0
    plazuela_label: int = 1
    structure_label: int = 2
    ignore_label: int = 255

    vicinity_radius_m: float = 2.0
    grid_size_m: float = 0.50

    # Swin3D model coordinates -> physical meters.
    coordinate_scale_m: float = 0.04
    horizontal_axes: Tuple[int, int] = (0, 1)

    far_negative_weight: float = 0.50
    focal_gamma: float = 2.0

    # Safety guard for dense 2D grids.
    max_grid_cells: int = 1_000_000
    eps: float = 1.0e-6

    def validate(self) -> None:
        if self.vicinity_radius_m < 0:
            raise ValueError("vicinity_radius_m must be non-negative")
        if self.grid_size_m <= 0:
            raise ValueError("grid_size_m must be positive")
        if self.coordinate_scale_m <= 0:
            raise ValueError("coordinate_scale_m must be positive")
        if len(self.horizontal_axes) != 2:
            raise ValueError("horizontal_axes must contain two axis indices")
        if self.horizontal_axes[0] == self.horizontal_axes[1]:
            raise ValueError("horizontal_axes must refer to different axes")
        if min(self.horizontal_axes) < 0 or max(self.horizontal_axes) > 2:
            raise ValueError("horizontal_axes values must be in {0, 1, 2}")
        if self.far_negative_weight < 0:
            raise ValueError("far_negative_weight cannot be negative")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma cannot be negative")
        if self.max_grid_cells < 1:
            raise ValueError("max_grid_cells must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


class Swin3DSimpleVicinityLoss(nn.Module):
    """Low-cost vicinity-aware auxiliary loss for Swin3D-S.

    The positive term is class-balanced: structure and plazuela hit losses are
    calculated separately and averaged when both are present. The probability
    used for proximity is collapsed archaeological probability, while the base
    Generalized Dice loss remains responsible for exact semantic labels.
    """

    def __init__(self, config: Optional[Swin3DVicinityLossConfig] = None) -> None:
        super().__init__()
        self.config = config or Swin3DVicinityLossConfig()
        self.config.validate()

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        coord: Tensor,
        batch: Tensor,
        return_components: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        self._validate_inputs(logits, target, coord, batch)

        # Force the small loss calculation to float32 for stable logarithms
        # under Swin3D mixed-precision training. Gradients still flow to logits.
        probabilities = F.softmax(logits.float(), dim=1)

        scene_losses = []
        hit_losses = []
        far_losses = []

        for batch_id in torch.unique(batch.detach(), sorted=True):
            mask = batch == batch_id
            total, hit, far = self._crop_loss(
                probabilities[mask],
                target[mask],
                coord[mask],
            )
            scene_losses.append(total)
            hit_losses.append(hit)
            far_losses.append(far)

        zero = logits.float().sum() * 0.0
        if scene_losses:
            total_loss = torch.stack(scene_losses).mean()
            hit_loss = torch.stack(hit_losses).mean()
            far_loss = torch.stack(far_losses).mean()
        else:
            total_loss = zero
            hit_loss = zero
            far_loss = zero

        if return_components:
            return total_loss, {
                "vicinity_total": total_loss,
                "vicinity_hit": hit_loss,
                "vicinity_far_negative": far_loss,
            }
        return total_loss

    def _crop_loss(
        self,
        probabilities: Tensor,
        target: Tensor,
        coord: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        valid = target != cfg.ignore_label
        zero = probabilities.sum() * 0.0

        if not bool(valid.any().item()):
            return zero, zero, zero

        probabilities = probabilities[valid]
        target = target[valid]

        axes = list(cfg.horizontal_axes)
        xy_m = (
            coord[valid][:, axes].detach().float()
            * float(cfg.coordinate_scale_m)
        )

        p_arch_point = (
            probabilities[:, cfg.plazuela_label]
            + probabilities[:, cfg.structure_label]
        ).clamp(0.0, 1.0)

        # Localize the crop before grid quantization so large world coordinates
        # do not create an unnecessarily large dense tensor.
        xy_min = xy_m.amin(dim=0)
        cell = torch.floor((xy_m - xy_min) / cfg.grid_size_m).long()

        unique_cell, inverse = torch.unique(cell, dim=0, return_inverse=True)
        num_occupied = int(unique_cell.shape[0])
        if num_occupied == 0:
            return zero, zero, zero

        # Average archaeological probability for all Swin3D voxels falling in
        # the same loss-grid cell. index_add_ is differentiable with respect to
        # the values being accumulated.
        p_sum = p_arch_point.new_zeros(num_occupied)
        p_sum.index_add_(0, inverse, p_arch_point)

        counts = p_arch_point.new_zeros(num_occupied)
        counts.index_add_(0, inverse, torch.ones_like(p_arch_point))
        p_cell = p_sum / counts.clamp_min(1.0)

        structure_counts = p_arch_point.new_zeros(num_occupied)
        structure_counts.index_add_(
            0,
            inverse,
            (target == cfg.structure_label).to(p_arch_point.dtype),
        )
        plazuela_counts = p_arch_point.new_zeros(num_occupied)
        plazuela_counts.index_add_(
            0,
            inverse,
            (target == cfg.plazuela_label).to(p_arch_point.dtype),
        )

        gt_structure_cell = structure_counts > 0
        gt_plazuela_cell = plazuela_counts > 0
        gt_archaeology_cell = gt_structure_cell | gt_plazuela_cell

        width = int(unique_cell[:, 0].max().item()) + 1
        height = int(unique_cell[:, 1].max().item()) + 1
        dense_cell_count = width * height

        if dense_cell_count > cfg.max_grid_cells:
            raise RuntimeError(
                "Vicinity-loss grid is too large: "
                f"{width} x {height} = {dense_cell_count:,} cells. "
                "Increase vicinity_grid_size_m, reduce the training crop "
                "extent, or raise max_grid_cells if memory permits."
            )

        flat_index = unique_cell[:, 1] * width + unique_cell[:, 0]

        # Out-of-place scatter preserves the autograd path from p_cell.
        pred_flat = p_cell.new_zeros(dense_cell_count).scatter(
            0, flat_index, p_cell
        )

        occupied_flat = torch.zeros(
            dense_cell_count,
            dtype=torch.bool,
            device=probabilities.device,
        )
        occupied_flat[flat_index] = True

        gt_structure_flat = torch.zeros_like(occupied_flat)
        gt_structure_flat[flat_index] = gt_structure_cell

        gt_plazuela_flat = torch.zeros_like(occupied_flat)
        gt_plazuela_flat[flat_index] = gt_plazuela_cell

        gt_archaeology_flat = gt_structure_flat | gt_plazuela_flat

        pred_grid = pred_flat.view(1, 1, height, width)
        gt_grid = gt_archaeology_flat.to(pred_grid.dtype).view(
            1, 1, height, width
        )

        radius_cells = int(
            math.ceil(cfg.vicinity_radius_m / cfg.grid_size_m)
        )
        kernel_size = 2 * radius_cells + 1

        if radius_cells > 0:
            nearby_prediction = F.max_pool2d(
                pred_grid,
                kernel_size=kernel_size,
                stride=1,
                padding=radius_cells,
            )
            buffered_ground_truth = F.max_pool2d(
                gt_grid,
                kernel_size=kernel_size,
                stride=1,
                padding=radius_cells,
            ) > 0
        else:
            nearby_prediction = pred_grid
            buffered_ground_truth = gt_grid > 0

        nearby_flat = nearby_prediction.reshape(-1)

        # Equal class contribution when both classes appear in the crop.
        class_hit_terms = []
        if bool(gt_structure_flat.any().item()):
            q_structure = nearby_flat[gt_structure_flat].clamp_min(cfg.eps)
            class_hit_terms.append(-torch.log(q_structure).mean())

        if bool(gt_plazuela_flat.any().item()):
            q_plazuela = nearby_flat[gt_plazuela_flat].clamp_min(cfg.eps)
            class_hit_terms.append(-torch.log(q_plazuela).mean())

        hit_loss = (
            torch.stack(class_hit_terms).mean()
            if class_hit_terms
            else zero
        )

        # Ignore cells inside the accepted vicinity in this auxiliary negative
        # term. Exact labels in those cells are still supervised by GDL.
        far_mask = occupied_flat & ~buffered_ground_truth.reshape(-1)
        if bool(far_mask.any().item()):
            p_far = pred_flat[far_mask].clamp(
                min=cfg.eps,
                max=1.0 - cfg.eps,
            )
            far_loss = -(
                p_far.pow(cfg.focal_gamma) * torch.log1p(-p_far)
            ).mean()
        else:
            far_loss = zero

        total = hit_loss + cfg.far_negative_weight * far_loss
        return total, hit_loss, far_loss

    def _validate_inputs(
        self,
        logits: Tensor,
        target: Tensor,
        coord: Tensor,
        batch: Tensor,
    ) -> None:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [N, C]")
        if target.ndim != 1 or target.shape[0] != logits.shape[0]:
            raise ValueError("target must have shape [N] and match logits")
        if coord.ndim != 2 or coord.shape != (logits.shape[0], 3):
            raise ValueError("coord must have shape [N, 3] and match logits")
        if batch.ndim != 1 or batch.shape[0] != logits.shape[0]:
            raise ValueError("batch must have shape [N] and match logits")

        required_label = max(
            self.config.ground_label,
            self.config.plazuela_label,
            self.config.structure_label,
        )
        if logits.shape[1] <= required_label:
            raise ValueError(
                f"logits provides {logits.shape[1]} classes, but semantic "
                f"label ID {required_label} is required"
            )


class Swin3DGeneralizedDiceVicinityLoss(nn.Module):
    """Combine an existing Generalized Dice loss with vicinity supervision.

    The wrapper accepts the ordinary two-argument call used by the official
    Swin3D validation code:

        loss = criterion(logits, target)

    When ``coord`` and ``batch`` are omitted, only the base loss is returned.
    This allows the original validation loop to remain unchanged.

    During training call:

        loss = criterion(
            logits, target, coord=coord, batch=batch, epoch=epoch
        )
    """

    def __init__(
        self,
        generalized_dice_loss: nn.Module,
        vicinity_loss: Swin3DSimpleVicinityLoss,
        vicinity_weight: float = 0.10,
        warmup_epochs: int = 10,
    ) -> None:
        super().__init__()
        if vicinity_weight < 0:
            raise ValueError("vicinity_weight cannot be negative")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")

        self.generalized_dice_loss = generalized_dice_loss
        self.vicinity_loss = vicinity_loss
        self.vicinity_weight = float(vicinity_weight)
        self.warmup_epochs = int(warmup_epochs)

    def current_weight(self, epoch: int) -> float:
        if self.warmup_epochs == 0:
            return self.vicinity_weight
        progress = min(
            max(float(epoch) / float(self.warmup_epochs), 0.0),
            1.0,
        )
        return self.vicinity_weight * progress

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        coord: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        epoch: int = 0,
        return_components: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        dice = self.generalized_dice_loss(logits, target)
        weight = self.current_weight(epoch)

        # Validation compatibility and zero-weight warm-up.
        if coord is None or batch is None or weight == 0.0:
            zero = logits.float().sum() * 0.0
            if return_components:
                return dice, {
                    "total": dice,
                    "generalized_dice": dice,
                    "vicinity_weight": dice.new_tensor(0.0),
                    "vicinity_total": zero,
                    "vicinity_hit": zero,
                    "vicinity_far_negative": zero,
                }
            return dice

        vicinity, parts = self.vicinity_loss(
            logits,
            target,
            coord,
            batch,
            return_components=True,
        )
        total = dice + weight * vicinity

        if return_components:
            values = {
                "total": total,
                "generalized_dice": dice,
                "vicinity_weight": total.new_tensor(weight),
            }
            values.update(parts)
            return total, values
        return total


class GeneralizedDiceLoss(nn.Module):
    """Optional point-wise multiclass Generalized Dice implementation.

    Use your existing verified Generalized Dice implementation when available.
    This class is included so the integration example is self-contained.
    """

    def __init__(self, ignore_index: int = 255, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.ndim != 2 or target.ndim != 1:
            raise ValueError("Expected logits [N, C] and target [N]")

        valid = target != self.ignore_index
        if not bool(valid.any().item()):
            return logits.float().sum() * 0.0

        logits = logits[valid].float()
        target = target[valid]
        probabilities = F.softmax(logits, dim=1)
        one_hot = F.one_hot(
            target,
            num_classes=logits.shape[1],
        ).to(probabilities.dtype)

        class_volume = one_hot.sum(dim=0)
        present = class_volume > 0
        weights = torch.where(
            present,
            1.0 / (class_volume.square() + self.eps),
            torch.zeros_like(class_volume),
        )

        intersection = (probabilities * one_hot).sum(dim=0)
        denominator = (probabilities + one_hot).sum(dim=0)

        numerator = 2.0 * (weights * intersection).sum()
        denominator = (weights * denominator).sum()

        dice = (numerator + self.eps) / (denominator + self.eps)
        return 1.0 - dice
