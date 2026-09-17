from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

CLASS_NAMES = ["Ground", "Structure", "Plazuela"]
CLASS_COLORS = ["forestgreen", "red", "purple"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)

def get_class_handles():
    return [
        Line2D([0],[0], marker="o", linestyle="", markerfacecolor=c,
               markeredgecolor="none", markersize=9, label=n)
        for n, c in zip(CLASS_NAMES, CLASS_COLORS)
    ]

@torch.no_grad()
def predict_tile(model, sample, device):
    coords = sample["coord"].float().to(device)
    features = sample["feat"].float().to(device)
    labels = sample["segment"].long()
    offset = torch.tensor([coords.shape[0]], dtype=torch.long, device=device)

    model.eval()
    logits = model((coords, features, offset))
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = logits.argmax(dim=1).cpu()
    probabilities = torch.softmax(logits, dim=1).cpu()

    return {
        "coords": coords.cpu(),
        "features": features.cpu(),
        "labels": labels.cpu(),
        "predictions": predictions,
        "probabilities": probabilities,
        "path": sample.get("path", "unknown_tile"),
    }

def _plot_indices(n, max_plot_points):
    if max_plot_points is not None and n > max_plot_points:
        return np.linspace(0, n - 1, max_plot_points, dtype=np.int64)
    return np.arange(n)

def format_3d_axis(ax, x, y, z, elevation_angle=55, azimuth_angle=-45):
    ax.view_init(elev=elevation_angle, azim=azimuth_angle)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_zlim(float(z.min()), float(z.max()))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    try:
        ax.set_box_aspect((
            max(float(np.ptp(x)), 1e-6),
            max(float(np.ptp(y)), 1e-6),
            max(float(np.ptp(z)), 1e-6),
        ))
    except Exception:
        pass

def print_tile_statistics(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    print("\nGround truth counts:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:<10}: {int((labels == class_id).sum()):>10,}")

    print("\nPrediction counts:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:<10}: {int((predictions == class_id).sum()):>10,}")

    print(f"\nPoint accuracy: {(predictions == labels).mean():.4f}")

def visualize_tile(
    model,
    dataset,
    tile_index,
    device,
    max_plot_points=40000,
    point_size=1.5,
    elevation_angle=55,
    azimuth_angle=-45,
    save_path=None,
    show=True,
    print_stats=True,
):
    sample = dataset[tile_index]
    output = predict_tile(model, sample, device)

    coords = output["coords"].numpy()
    features = output["features"].numpy()
    labels = output["labels"].numpy()
    predictions = output["predictions"].numpy()
    probabilities = output["probabilities"].numpy()

    idx = _plot_indices(coords.shape[0], max_plot_points)
    xyz = coords[idx]
    labels_plot = labels[idx]
    pred_plot = predictions[idx]
    slope_plot = features[:, 0][idx] * 90.0

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    handles = get_class_handles()

    fig = plt.figure(figsize=(20, 13), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])

    ax_gt = fig.add_subplot(grid[0, 0], projection="3d")
    ax_gt.scatter(x, y, z, c=labels_plot, cmap=CLASS_CMAP,
                  vmin=0, vmax=2, s=point_size, alpha=0.9)
    ax_gt.set_title("Ground Truth", fontsize=15, fontweight="bold")
    ax_gt.legend(handles=handles, loc="upper right", fontsize=9)

    ax_elev = fig.add_subplot(grid[0, 1], projection="3d")
    sc1 = ax_elev.scatter(x, y, z, c=z, cmap="terrain",
                          s=point_size, alpha=0.9)
    ax_elev.set_title("Model Input - Elevation", fontsize=15, fontweight="bold")
    cb1 = fig.colorbar(sc1, ax=ax_elev, shrink=0.65, pad=0.02)
    cb1.set_label("Local Elevation (m)")

    ax_slope = fig.add_subplot(grid[0, 2], projection="3d")
    sc2 = ax_slope.scatter(x, y, z, c=slope_plot, cmap="turbo",
                           s=point_size, alpha=0.9)
    ax_slope.set_title("Model Input - Slope", fontsize=15, fontweight="bold")
    cb2 = fig.colorbar(sc2, ax=ax_slope, shrink=0.65, pad=0.02)
    cb2.set_label("Slope (degrees)")

    ax_pred = fig.add_subplot(grid[1, :], projection="3d")
    ax_pred.scatter(x, y, z, c=pred_plot, cmap=CLASS_CMAP,
                    vmin=0, vmax=2, s=point_size + 0.5, alpha=0.9)
    ax_pred.set_title("Swin3D Model Prediction", fontsize=17, fontweight="bold")
    ax_pred.legend(handles=handles, loc="upper right", fontsize=11, title="Class")

    for ax in [ax_gt, ax_elev, ax_slope, ax_pred]:
        format_3d_axis(ax, x, y, z, elevation_angle, azimuth_angle)

    tile_name = Path(str(output["path"])).name
    fig.suptitle(
        f"Swin3D Archaeological Point-Cloud Segmentation\n{tile_name}",
        fontsize=18,
        fontweight="bold",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved visualization:", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if print_stats:
        print("\nTile:", tile_name)
        print("Number of points:", f"{coords.shape[0]:,}")
        print_tile_statistics(labels, predictions)

    return {
        **output,
        "coords_np": coords,
        "features_np": features,
        "labels_np": labels,
        "predictions_np": predictions,
        "probabilities_np": probabilities,
        "plot_indices": idx,
    }

def visualize_prediction_errors(
    visualization_output,
    max_plot_points=40000,
    point_size=2.0,
    elevation_angle=55,
    azimuth_angle=-45,
    save_path=None,
    show=True,
):
    coords = visualization_output["coords_np"]
    gt = visualization_output["labels_np"]
    pred = visualization_output["predictions_np"]

    idx = _plot_indices(coords.shape[0], max_plot_points)
    xyz = coords[idx]
    gt = gt[idx]
    pred = pred[idx]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    correct = pred == gt
    false_structure = (pred == 1) & (gt != 1)
    false_plazuela = (pred == 2) & (gt != 2)
    missed_archaeology = (gt != 0) & (pred == 0)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x[correct], y[correct], z[correct], s=0.7,
               alpha=0.15, c="lightgray", label="Correct")
    ax.scatter(x[false_structure], y[false_structure], z[false_structure],
               s=point_size, alpha=0.9, c="red", label="False Structure")
    ax.scatter(x[false_plazuela], y[false_plazuela], z[false_plazuela],
               s=point_size, alpha=0.9, c="purple", label="False Plazuela")
    ax.scatter(x[missed_archaeology], y[missed_archaeology], z[missed_archaeology],
               s=point_size + 1, alpha=1.0, c="black", label="Missed Archaeology")

    format_3d_axis(ax, x, y, z, elevation_angle, azimuth_angle)
    ax.set_title("Prediction Error Map", fontsize=16, fontweight="bold")
    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved error map:", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "correct": int(correct.sum()),
        "false_structure": int(false_structure.sum()),
        "false_plazuela": int(false_plazuela.sum()),
        "missed_archaeology": int(missed_archaeology.sum()),
    }

def find_archaeological_tiles(
    dataset,
    minimum_archaeology_points=100,
    sort_by="archaeology",
    descending=True,
):
    results = []

    for index in range(len(dataset)):
        sample = dataset[index]
        labels = sample["segment"].cpu().numpy()

        structure = int((labels == 1).sum())
        plazuela = int((labels == 2).sum())
        archaeology = structure + plazuela

        if archaeology >= minimum_archaeology_points:
            results.append({
                "index": index,
                "structure": structure,
                "plazuela": plazuela,
                "archaeology": archaeology,
                "path": sample["path"],
            })

    results.sort(key=lambda x: x[sort_by], reverse=descending)
    return results

def print_archaeological_tiles(tiles, limit=20):
    print(f"Archaeological tiles found: {len(tiles)}\n")
    print(f"{'Index':>6}  {'Structure':>10}  {'Plazuela':>10}  {'Total Arch':>10}  File")
    print("-" * 90)

    for tile in tiles[:limit]:
        print(
            f"{tile['index']:>6}  "
            f"{tile['structure']:>10,}  "
            f"{tile['plazuela']:>10,}  "
            f"{tile['archaeology']:>10,}  "
            f"{Path(tile['path']).name}"
        )

@torch.no_grad()
def evaluate_test_tiles(
    model,
    dataset,
    device,
    print_progress=True,
):
    """
    Evaluate every tile independently.

    Returns statistics including:

        accuracy

        archaeological false positives:
            GT = Ground
            Prediction = Structure or Plazuela

        false_positive_count
        false_positive_rate

        structure_false_positives
        plazuela_false_positives
    """

    results = []

    number_of_tiles = len(dataset)

    model.eval()

    for tile_index in range(
        number_of_tiles
    ):

        sample = dataset[
            tile_index
        ]

        output = predict_tile(
            model=model,
            sample=sample,
            device=device,
        )

        labels = (
            output["labels"]
            .numpy()
        )

        predictions = (
            output["predictions"]
            .numpy()
        )

        # ====================================================
        # Accuracy
        # ====================================================

        correct = (
            predictions == labels
        )

        accuracy = (
            correct.mean()
        )


        # ====================================================
        # Archaeological false positives
        #
        # Ground truth = Ground
        # Prediction   = Structure or Plazuela
        # ====================================================

        ground_mask = (
            labels == 0
        )

        archaeological_prediction = (
            predictions != 0
        )

        false_positive_mask = (
            ground_mask
            &
            archaeological_prediction
        )

        false_positive_count = int(
            false_positive_mask.sum()
        )


        # ----------------------------------------------------
        # FP rate relative to number of GT ground points
        # ----------------------------------------------------

        number_of_ground_points = int(
            ground_mask.sum()
        )

        false_positive_rate = (
            false_positive_count
            /
            max(
                number_of_ground_points,
                1,
            )
        )


        # ====================================================
        # Separate Structure / Plazuela false positives
        # ====================================================

        structure_false_positives = int(
            (
                ground_mask
                &
                (predictions == 1)
            ).sum()
        )

        plazuela_false_positives = int(
            (
                ground_mask
                &
                (predictions == 2)
            ).sum()
        )


        # ====================================================
        # Archaeological GT counts
        # ====================================================

        structure_gt = int(
            (
                labels == 1
            ).sum()
        )

        plazuela_gt = int(
            (
                labels == 2
            ).sum()
        )


        results.append(
            {
                "index": tile_index,

                "accuracy": float(
                    accuracy
                ),

                "false_positive_count": (
                    false_positive_count
                ),

                "false_positive_rate": float(
                    false_positive_rate
                ),

                "structure_false_positives": (
                    structure_false_positives
                ),

                "plazuela_false_positives": (
                    plazuela_false_positives
                ),

                "ground_points": (
                    number_of_ground_points
                ),

                "structure_gt": (
                    structure_gt
                ),

                "plazuela_gt": (
                    plazuela_gt
                ),

                "path": output[
                    "path"
                ],
            }
        )


        if (
            print_progress
            and
            (
                tile_index == 0
                or
                (tile_index + 1) % 10 == 0
                or
                tile_index + 1 == number_of_tiles
            )
        ):

            print(
                f"Evaluated tile "
                f"{tile_index + 1:03d}/"
                f"{number_of_tiles:03d}"
            )


    return results


def find_extreme_test_tiles(
    tile_results,
    top_k=5,
):
    """
    Find top-K tiles for:

        1. Highest point accuracy
        2. Highest archaeological false-positive count
        3. Highest archaeological false-positive rate
    """

    if len(tile_results) == 0:
        raise ValueError(
            "tile_results is empty."
        )

    top_accuracy = sorted(
        tile_results,
        key=lambda item: item[
            "accuracy"
        ],
        reverse=True,
    )[:top_k]

    top_false_positive_count = sorted(
        tile_results,
        key=lambda item: item[
            "false_positive_count"
        ],
        reverse=True,
    )[:top_k]

    top_false_positive_rate = sorted(
        tile_results,
        key=lambda item: item[
            "false_positive_rate"
        ],
        reverse=True,
    )[:top_k]

    return {
        "highest_accuracy": (
            top_accuracy
        ),

        "highest_false_positive_count": (
            top_false_positive_count
        ),

        "highest_false_positive_rate": (
            top_false_positive_rate
        ),
    }

def print_extreme_test_tiles(
    extreme_tiles,
):
    """
    Print top-K tiles for each extreme category.
    """

    category_names = {
        "highest_accuracy":
            "TOP HIGHEST ACCURACY",

        "highest_false_positive_count":
            "TOP HIGHEST FALSE-POSITIVE COUNT",

        "highest_false_positive_rate":
            "TOP HIGHEST FALSE-POSITIVE RATE",
    }

    for key, name in category_names.items():

        tiles = extreme_tiles[
            key
        ]

        print()
        print("=" * 70)
        print(name)
        print("=" * 70)

        for rank, tile in enumerate(
            tiles,
            start=1,
        ):

            print()
            print(
                f"Rank {rank}"
            )

            print(
                "Tile index:",
                tile["index"],
            )

            print(
                "Path:",
                tile["path"],
            )

            print(
                "Accuracy:",
                f"{tile['accuracy']:.4f}",
            )

            print(
                "False-positive count:",
                f"{tile['false_positive_count']:,}",
            )

            print(
                "False-positive rate:",
                f"{tile['false_positive_rate']:.4f}",
            )

            print("-" * 70)

from scipy.ndimage import (
    binary_opening,
    binary_closing,
    label,
)


def spatial_filter_predictions(
    coords,
    predictions,
    cell_size=1.0,

    structure_min_cells=8,
    plazuela_min_cells=15,

    opening_iterations=1,
    closing_iterations=0,

    verbose=False,
):
    """
    Spatially clean Structure / Plazuela predictions.

    Classes
    -------
    0 = Ground
    1 = Structure
    2 = Plazuela

    Processing
    ----------
    1. Rasterize predictions to an XY grid.
    2. Morphological opening removes thin lines / isolated noise.
    3. Connected-component filtering removes small patches.
    4. Archaeological predictions rejected by the filter become Ground.

    Parameters
    ----------
    coords:
        [N, 3] numpy array or torch tensor.

    predictions:
        [N] numpy array or torch tensor.

    cell_size:
        XY grid resolution in meters.

    structure_min_cells:
        Minimum connected Structure area in occupied cells.

    plazuela_min_cells:
        Minimum connected Plazuela area in occupied cells.

    opening_iterations:
        Number of 3x3 morphological opening operations.

    closing_iterations:
        Optional closing after opening.

    Returns
    -------
    filtered_predictions:
        [N] numpy array.

    diagnostics:
        Dictionary describing removed points/components.
    """

    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()

    if torch.is_tensor(predictions):
        predictions = (
            predictions.detach().cpu().numpy()
        )

    coords = np.asarray(coords)
    predictions = np.asarray(
        predictions
    ).astype(np.int64)

    filtered_predictions = (
        predictions.copy()
    )

    # --------------------------------------------------------
    # Make local XY grid
    # --------------------------------------------------------

    x = coords[:, 0]
    y = coords[:, 1]

    x_min = x.min()
    y_min = y.min()

    cell_x = np.floor(
        (x - x_min) / cell_size
    ).astype(np.int64)

    cell_y = np.floor(
        (y - y_min) / cell_size
    ).astype(np.int64)

    grid_width = (
        cell_x.max() + 1
    )

    grid_height = (
        cell_y.max() + 1
    )

    diagnostics = {}

    # 8-connected neighborhood
    connectivity = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Process archaeological classes separately
    # --------------------------------------------------------

    settings = {
        1: structure_min_cells,
        2: plazuela_min_cells,
    }

    for class_id, min_cells in settings.items():

        class_point_mask = (
            predictions == class_id
        )

        before_points = int(
            class_point_mask.sum()
        )

        if before_points == 0:

            diagnostics[class_id] = {
                "before_points": 0,
                "after_points": 0,
                "removed_points": 0,
                "components_before": 0,
                "components_after": 0,
            }

            continue

        # ----------------------------------------------------
        # Build binary occupancy grid
        # ----------------------------------------------------

        occupancy = np.zeros(
            (
                grid_height,
                grid_width,
            ),
            dtype=bool,
        )

        occupancy[
            cell_y[class_point_mask],
            cell_x[class_point_mask],
        ] = True

        # ----------------------------------------------------
        # Remove narrow/speckled predictions
        # ----------------------------------------------------

        cleaned = occupancy.copy()

        if opening_iterations > 0:

            cleaned = binary_opening(
                cleaned,
                structure=connectivity,
                iterations=opening_iterations,
            )

        if closing_iterations > 0:

            cleaned = binary_closing(
                cleaned,
                structure=connectivity,
                iterations=closing_iterations,
            )

        # ----------------------------------------------------
        # Connected components
        # ----------------------------------------------------

        components_before, n_before = label(
            occupancy,
            structure=connectivity,
        )

        component_map, n_components = label(
            cleaned,
            structure=connectivity,
        )

        keep_grid = np.zeros_like(
            cleaned,
            dtype=bool,
        )

        kept_components = 0

        for component_id in range(
            1,
            n_components + 1,
        ):

            component_mask = (
                component_map
                ==
                component_id
            )

            component_cells = int(
                component_mask.sum()
            )

            if (
                component_cells
                >=
                min_cells
            ):

                keep_grid[
                    component_mask
                ] = True

                kept_components += 1

        # ----------------------------------------------------
        # Map cleaned grid back to points
        # ----------------------------------------------------

        point_cell_is_kept = (
            keep_grid[
                cell_y,
                cell_x,
            ]
        )

        remove_mask = (
            class_point_mask
            &
            ~point_cell_is_kept
        )

        # Rejected archaeology -> Ground
        filtered_predictions[
            remove_mask
        ] = 0

        after_points = int(
            (
                filtered_predictions
                ==
                class_id
            ).sum()
        )

        diagnostics[class_id] = {
            "before_points": before_points,
            "after_points": after_points,

            "removed_points": (
                before_points
                -
                after_points
            ),

            "components_before": int(
                n_before
            ),

            "components_after": int(
                kept_components
            ),
        }

    if verbose:

        print()
        print("Spatial filtering")
        print("-----------------")

        for class_id in [1, 2]:

            name = CLASS_NAMES[
                class_id
            ]

            info = diagnostics[
                class_id
            ]

            print()
            print(name)

            print(
                "  Points before:",
                f"{info['before_points']:,}",
            )

            print(
                "  Points after:",
                f"{info['after_points']:,}",
            )

            print(
                "  Points removed:",
                f"{info['removed_points']:,}",
            )

            print(
                "  Components before:",
                info[
                    "components_before"
                ],
            )

            print(
                "  Components after:",
                info[
                    "components_after"
                ],
            )

    return (
        filtered_predictions,
        diagnostics,
    )

def compare_before_after_filter(
    labels,
    original_predictions,
    filtered_predictions,
):
    """
    Compare point-level performance before and after filtering.
    """

    if torch.is_tensor(labels):
        labels = (
            labels.detach()
            .cpu()
            .numpy()
        )

    if torch.is_tensor(
        original_predictions
    ):
        original_predictions = (
            original_predictions
            .detach()
            .cpu()
            .numpy()
        )

    labels = np.asarray(labels)

    original_predictions = np.asarray(
        original_predictions
    )

    filtered_predictions = np.asarray(
        filtered_predictions
    )

    accuracy_before = np.mean(
        original_predictions
        ==
        labels
    )

    accuracy_after = np.mean(
        filtered_predictions
        ==
        labels
    )

    print()
    print("=" * 55)
    print("FILTER PERFORMANCE")
    print("=" * 55)

    print(
        f"Accuracy before: "
        f"{accuracy_before:.4f}"
    )

    print(
        f"Accuracy after:  "
        f"{accuracy_after:.4f}"
    )

    print(
        f"Difference:      "
        f"{accuracy_after - accuracy_before:+.4f}"
    )

    print()

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        gt = (
            labels == class_id
        )

        before = (
            original_predictions
            ==
            class_id
        )

        after = (
            filtered_predictions
            ==
            class_id
        )

        tp_before = (
            gt & before
        ).sum()

        tp_after = (
            gt & after
        ).sum()

        fp_before = (
            ~gt & before
        ).sum()

        fp_after = (
            ~gt & after
        ).sum()

        recall_before = (
            tp_before
            /
            max(
                gt.sum(),
                1,
            )
        )

        recall_after = (
            tp_after
            /
            max(
                gt.sum(),
                1,
            )
        )

        precision_before = (
            tp_before
            /
            max(
                before.sum(),
                1,
            )
        )

        precision_after = (
            tp_after
            /
            max(
                after.sum(),
                1,
            )
        )

        print(class_name)

        print(
            f"  Precision: "
            f"{precision_before:.4f}"
            f" -> "
            f"{precision_after:.4f}"
        )

        print(
            f"  Recall:    "
            f"{recall_before:.4f}"
            f" -> "
            f"{recall_after:.4f}"
        )

        print(
            f"  FP:        "
            f"{fp_before:,}"
            f" -> "
            f"{fp_after:,}"
        )

        print()

    return {
        "accuracy_before":
            float(accuracy_before),

        "accuracy_after":
            float(accuracy_after),

        "accuracy_change":
            float(
                accuracy_after
                -
                accuracy_before
            ),
    }

def visualize_filtered_prediction(
    visualization_output,
    filtered_predictions,
    max_plot_points=50_000,
    point_size=2.0,
    elevation_angle=55,
    azimuth_angle=-45,
):
    coords = visualization_output[
        "coords_np"
    ]

    original_predictions = (
        visualization_output[
            "predictions_np"
        ]
    )

    labels = visualization_output[
        "labels_np"
    ]

    indices = _plot_indices(
        coords.shape[0],
        max_plot_points,
    )

    xyz = coords[
        indices
    ]

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    fig = plt.figure(
        figsize=(18, 7)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    ax1 = fig.add_subplot(
        121,
        projection="3d",
    )

    ax1.scatter(
        x,
        y,
        z,
        c=original_predictions[
            indices
        ],
        cmap=CLASS_CMAP,
        vmin=0,
        vmax=2,
        s=point_size,
        alpha=0.9,
    )

    ax1.set_title(
        "Original Model Prediction",
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Filtered
    # --------------------------------------------------------

    ax2 = fig.add_subplot(
        122,
        projection="3d",
    )

    ax2.scatter(
        x,
        y,
        z,
        c=filtered_predictions[
            indices
        ],
        cmap=CLASS_CMAP,
        vmin=0,
        vmax=2,
        s=point_size,
        alpha=0.9,
    )

    ax2.set_title(
        "Spatially Filtered Prediction",
        fontweight="bold",
    )

    for ax in [
        ax1,
        ax2,
    ]:

        format_3d_axis(
            ax,
            x,
            y,
            z,
            elevation_angle,
            azimuth_angle,
        )

        ax.legend(
            handles=get_class_handles(),
            loc="upper right",
        )

    plt.tight_layout()
    plt.show()

    compare_before_after_filter(
        labels=labels,
        original_predictions=(
            original_predictions
        ),
        filtered_predictions=(
            filtered_predictions
        ),
    )

