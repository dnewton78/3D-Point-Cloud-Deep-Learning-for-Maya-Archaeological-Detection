import torch
import torch.nn as nn
import MinkowskiEngine as ME

from MinkowskiEngine import SparseTensor
from Swin3D.models import Swin3DUNet

def offset2batch(offset):
    """
    Convert cumulative point offsets into a batch index for each point.

    Example:
        offset = [100, 250, 400]

    Returns:
        [0 repeated 100 times,
         1 repeated 150 times,
         2 repeated 150 times]
    """
    offset = offset.long()

    counts = torch.diff(
        offset,
        prepend=offset.new_zeros(1),
    )

    batch_ids = torch.arange(
        len(offset),
        device=offset.device,
        dtype=torch.long,
    )

    return torch.repeat_interleave(batch_ids, counts)

class Model(nn.Module):
    """
    Swin3D-S model for three-class point-cloud segmentation.

    Expected input:
        pos_feat_off0 = (coords, features, offset)

        coords:
            Shape [N, 3]
            XYZ coordinates.

        features:
            Shape [N, 1]
            Slope value for every point.

        offset:
            Shape [batch_size]
            Cumulative number of points in each batch.

    Network input features:
        X, Y, Z, slope = 4 channels
    """

    def __init__(
        self,
        args=None,
        num_classes=3,
        in_channels=4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Swin3D-S architecture
        self.backbone = Swin3DUNet(
            [2, 4, 9, 4, 4],              # depths
            [48, 96, 192, 384, 384],      # channels
            [6, 6, 12, 24, 24],           # num_heads
            [5, 7, 7, 7, 7],              # window_sizes
            4,                             # quant_sizes
            up_k=3,
            drop_path_rate=0.3,
            num_classes=num_classes,
            num_layers=5,
            stem_transformer=True,
            upsample="linear_attn",
            first_down_stride=2,
            knn_down=True,
            in_channels=in_channels,
            cRSE="XYZ",
            fp16_mode=1,
        )

    def forward(self, pos_feat_off0):
        coords, features, offset = pos_feat_off0

        # Ensure correct data types.
        coords = coords.float()
        features = features.float()

        # Convert cumulative offsets into a batch index for each point.
        batch = offset2batch(offset).to(coords.device)

        # Input features:
        # [X, Y, Z, slope]
        point_features = torch.cat(
            [coords, features],
            dim=1,
        )

        if point_features.shape[1] != self.in_channels:
            raise ValueError(
                f"Swin3D expected {self.in_channels} input channels, "
                f"but received {point_features.shape[1]}. "
                f"Coordinates shape: {coords.shape}; "
                f"feature shape: {features.shape}"
            )

        # MinkowskiEngine coordinates:
        # [batch_index, X, Y, Z]
        sparse_coordinates = torch.cat(
            [batch.unsqueeze(1), coords],
            dim=1,
        )

        # TensorField features contain:
        # [batch_index, X, Y, Z, X, Y, Z, slope]
        #
        # The first four values are later used for coords_sp.
        field_features = torch.cat(
            [
                batch.unsqueeze(1).float(),
                coords,
                point_features,
            ],
            dim=1,
        )

        in_field = ME.TensorField(
            features=field_features,
            coordinates=sparse_coordinates,
            quantization_mode=(
                ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
            ),
            minkowski_algorithm=(
                ME.MinkowskiAlgorithm.SPEED_OPTIMIZED
            ),
            device=coords.device,
        )

        sparse_field = in_field.sparse()

        # Coordinate-related sparse tensor:
        # [batch_index, X, Y, Z]
        coords_sp = SparseTensor(
            features=sparse_field.F[:, :4],
            coordinate_map_key=sparse_field.coordinate_map_key,
            coordinate_manager=sparse_field.coordinate_manager,
        )

        # Model input:
        # [X, Y, Z, slope]
        sp = SparseTensor(
            features=sparse_field.F[:, 4:],
            coordinate_map_key=sparse_field.coordinate_map_key,
            coordinate_manager=sparse_field.coordinate_manager,
        )

        output = self.backbone(
            sp,
            coords_sp,
        )

        # ------------------------------------------------------------
        # Convert voxel-level predictions back to original point level
        # ------------------------------------------------------------
        
        if isinstance(output, ME.SparseTensor):
            # Backbone returned a MinkowskiEngine SparseTensor.
            output = output.slice(in_field).F
        
        elif torch.is_tensor(output):
            # Backbone returned only the voxel feature tensor.
            #
            # Each original point has an index identifying which
            # quantized voxel it belongs to.
            inverse_map = in_field.inverse_mapping(
                sparse_field.coordinate_map_key
            )
        
            if output.shape[0] != sparse_field.F.shape[0]:
                raise RuntimeError(
                    "Cannot map voxel predictions to original points. "
                    f"Backbone output has {output.shape[0]} rows, but "
                    f"the input sparse tensor has {sparse_field.F.shape[0]} rows."
                )
        
            output = output[inverse_map.long()]
        
        else:
            raise TypeError(
                f"Unexpected Swin3D output type: {type(output)}"
            )

        return output