import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAVDLoss(nn.Module):
    """
    Differentiable surrogate for Archaeological Vicinity Detection (AVD).

    Classes:
        0 = Ground
        1 = Structure
        2 = Plazuela

    The loss rewards:
        1. archaeological predictions near true archaeological points
        2. coverage of true archaeological regions
        3. recall more strongly than precision through beta=2

    This is a TRAINING SURROGATE.

    The original hard AVD metric should still be used for validation/test
    reporting.
    """

    def __init__(
        self,
        radius_m=5.0,
        beta=2.0,
        semantic_weight=0.5,
        false_positive_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()

        self.radius_m = radius_m
        self.beta = beta
        self.semantic_weight = semantic_weight
        self.false_positive_weight = false_positive_weight
        self.eps = eps


    def forward(
        self,
        logits,
        targets,
        coords,
        offset,
    ):
        """
        logits:
            [N, 3]

        targets:
            [N]

        coords:
            [N, 3]
            Local metric XYZ coordinates.

        offset:
            cumulative batch point offsets.
        """

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        total_loss = logits.new_tensor(0.0)

        batch_start = 0
        valid_tiles = 0


        for batch_end in offset.tolist():

            tile_probabilities = probabilities[
                batch_start:batch_end
            ]

            tile_targets = targets[
                batch_start:batch_end
            ]

            tile_coords = coords[
                batch_start:batch_end
            ]

            tile_loss = self._tile_loss(
                probabilities=tile_probabilities,
                targets=tile_targets,
                coords=tile_coords,
            )

            if tile_loss is not None:
                total_loss = (
                    total_loss + tile_loss
                )

                valid_tiles += 1

            batch_start = batch_end


        if valid_tiles == 0:
            return logits.sum() * 0.0

        return total_loss / valid_tiles


    def _tile_loss(
        self,
        probabilities,
        targets,
        coords,
    ):

        # ====================================================
        # Archaeological probability
        #
        # Structure + Plazuela
        # ====================================================

        archaeology_probability = (
            probabilities[:, 1]
            +
            probabilities[:, 2]
        )


        archaeology_target = (
            (targets == 1)
            |
            (targets == 2)
        )


        if archaeology_target.sum() == 0:
            # ------------------------------------------------
            # No archaeological feature in this tile.
            #
            # Penalize archaeological predictions.
            # ------------------------------------------------

            false_archaeology = (
                archaeology_probability.mean()
            )

            return false_archaeology


        xy = coords[:, :2]

        gt_xy = xy[
            archaeology_target
        ]


        # ====================================================
        # Find distance to nearest GT archaeological point
        # ====================================================

        nearest_distance = (
            self._nearest_gt_distance(
                xy,
                gt_xy,
            )
        )


        # ====================================================
        # Soft vicinity mask
        #
        # Instead of:
        #
        # distance <= radius
        #
        # use a smooth transition around radius.
        # ====================================================

        temperature = 1.0

        vicinity_weight = torch.sigmoid(
            (
                self.radius_m
                -
                nearest_distance
            )
            / temperature
        )

        # ====================================================
        # False archaeological prediction penalty
        #
        # Predictions far from archaeological ground truth
        # should be strongly discouraged.
        # ====================================================
        
        outside_vicinity_weight = (
            1.0 - vicinity_weight
        )
        
        false_positive_loss = (
            archaeology_probability
            *
            outside_vicinity_weight
        ).mean()
        
        # ====================================================
        # Soft vicinity precision
        #
        # Equivalent idea:
        #
        # useful predicted archaeological evidence
        # -----------------------------------------
        # all predicted archaeological evidence
        # ====================================================

        useful_prediction = (
            archaeology_probability
            *
            vicinity_weight
        ).sum()


        total_prediction = (
            archaeology_probability.sum()
        )


        soft_precision = (
            useful_prediction
            + self.eps
        ) / (
            total_prediction
            + self.eps
        )


        # ====================================================
        # Soft archaeological recall
        #
        # Ask:
        #
        # How much probability does the model assign to
        # archaeology at real archaeological locations?
        # ====================================================

        soft_recall = (
            archaeology_probability[
                archaeology_target
            ].mean()
        )


        # ====================================================
        # Soft F-beta
        # ====================================================

        beta_squared = (
            self.beta ** 2
        )


        soft_fbeta = (
            (
                1.0
                +
                beta_squared
            )
            *
            soft_precision
            *
            soft_recall
        ) / (
            beta_squared
            *
            soft_precision
            +
            soft_recall
            +
            self.eps
        )


        discovery_loss = (
            1.0
            -
            soft_fbeta
        )


        # ====================================================
        # Semantic archaeological loss
        #
        # We still want:
        #
        # Structure -> Structure
        # Plazuela  -> Plazuela
        #
        # rather than only finding "something archaeological".
        # ====================================================

        semantic_losses = []


        for class_id in [1, 2]:

            class_mask = (
                targets == class_id
            )


            if class_mask.sum() == 0:
                continue


            class_probability = (
                probabilities[
                    class_mask,
                    class_id,
                ]
            )


            class_recall = (
                class_probability.mean()
            )


            semantic_losses.append(
                1.0
                -
                class_recall
            )


        if len(semantic_losses) > 0:

            semantic_loss = torch.stack(
                semantic_losses
            ).mean()

        else:

            semantic_loss = logits_zero(
                probabilities
            )

        return (
            discovery_loss
            +
            self.semantic_weight
            *
            semantic_loss
            +
            self.false_positive_weight
            *
            false_positive_loss
        )


    def _nearest_gt_distance(
        self,
        query_xy,
        gt_xy,
        chunk_size=4096,
    ):
        """
        Calculate nearest GT archaeological distance without
        constructing one enormous NxM distance matrix.
        """

        nearest_distances = []


        for start in range(
            0,
            query_xy.shape[0],
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                query_xy.shape[0],
            )


            query_chunk = query_xy[
                start:end
            ]


            distances = torch.cdist(
                query_chunk,
                gt_xy,
            )


            nearest = distances.min(
                dim=1
            ).values


            nearest_distances.append(
                nearest
            )


        return torch.cat(
            nearest_distances,
            dim=0,
        )


def logits_zero(probabilities):
    return probabilities.sum() * 0.0