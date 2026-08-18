import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss for multiclass point-cloud segmentation.

    logits:
        Shape [N, C]

    targets:
        Shape [N], containing integer class IDs:
        0, 1, ..., C-1
    """

    def __init__(
        self,
        num_classes: int = 3,
        smooth: float = 1e-6,
        ignore_index: int | None = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        targets = targets.long()

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            logits = logits[valid_mask]
            targets = targets[valid_mask]

        if targets.numel() == 0:
            return logits.sum() * 0.0

        probabilities = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes,
        ).float()

        # Number of ground-truth points belonging to each class.
        class_volume = targets_one_hot.sum(dim=0)

        # Generalized Dice gives greater weight to smaller classes.
        class_weights = 1.0 / (class_volume.pow(2) + self.smooth)

        intersection = (
            probabilities * targets_one_hot
        ).sum(dim=0)

        denominator = (
            probabilities + targets_one_hot
        ).sum(dim=0)

        numerator = 2.0 * (
            class_weights * intersection
        ).sum()

        denominator = (
            class_weights * denominator
        ).sum()

        dice_score = (
            numerator + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice_score