from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from archaeological_vicinity_metric import (
    AVDConfig,
    ArchaeologicalVicinityMetric,
    ClassSpec,
)


# Your project mapping:
#
# 0 = Ground
# 1 = Structure
# 2 = Plazuela
PROJECT_CLASS_NAMES = {
    0: "Ground",
    1: "Structure",
    2: "Plazuela",
}


def create_metric(
    cell_size_m: float = 1.0,
    radii_m: Sequence[float] = (2.0, 5.0, 10.0),
    min_points_per_cell: int = 1,
    min_evidence_cells: int = 3,
    beta: float = 2.0,
) -> ArchaeologicalVicinityMetric:
    """
    Create an AVD metric configured for this project's class mapping.

    Includes:
        - semantic Structure evaluation
        - semantic Plazuela evaluation
        - combined archaeological discovery evaluation
    """

    return ArchaeologicalVicinityMetric(
        class_specs=[
            ClassSpec(
                name="structure",
                predicted_labels=1,
                target_labels=1,
            ),
            ClassSpec(
                name="plazuela",
                predicted_labels=2,
                target_labels=2,
            ),
            ClassSpec(
                name="archaeological_discovery",
                predicted_labels=(1, 2),
                target_labels=(1, 2),
            ),
        ],
        config=AVDConfig(
            cell_size_m=cell_size_m,
            radii_m=tuple(float(r) for r in radii_m),
            min_points_per_cell=min_points_per_cell,
            min_evidence_cells=min_evidence_cells,
            beta=beta,
            ignore_label=-100,

            # Fallback because the current dataset does not provide
            # explicit archaeological feature-instance IDs.
            infer_instances_from_semantic_mask=True,
            connected_component_connectivity=2,
            semantic_mask_closing_iterations=0,
        ),
    )


def extract_logits(model_output: object) -> torch.Tensor:
    """
    Extract logits from the possible Swin3D output formats.
    """

    if torch.is_tensor(model_output):
        return model_output

    if hasattr(model_output, "F"):
        return model_output.F

    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise RuntimeError(
                "The model returned an empty tuple or list."
            )

        first_item = model_output[0]

        if torch.is_tensor(first_item):
            return first_item

        if hasattr(first_item, "F"):
            return first_item.F

    raise TypeError(
        "Unsupported model output type: "
        f"{type(model_output)}"
    )


def offsets_to_ranges(
    offsets: torch.Tensor,
) -> Tuple[Tuple[int, int], ...]:
    """
    Convert cumulative point offsets into per-tile ranges.

    Example:
        [100, 250, 400]

    becomes:
        (0, 100), (100, 250), (250, 400)
    """

    offset_values = (
        offsets
        .detach()
        .cpu()
        .long()
        .tolist()
    )

    ranges = []
    start = 0

    for end in offset_values:
        end = int(end)
        ranges.append((start, end))
        start = end

    return tuple(ranges)


def load_checkpoint(
    model,
    checkpoint_path,
    device: torch.device,
) -> Dict:
    """
    Load the saved best-checkpoint weights into the model.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return checkpoint


@torch.inference_mode()
def run_avd_inference(
    model,
    test_loader,
    device: torch.device,
    metric: ArchaeologicalVicinityMetric,
    print_progress: bool = True,
) -> Dict:
    """
    Run the best model over the test set and update the AVD metric.

    Each tile is supplied separately to metric.update(), even when the
    DataLoader batch contains multiple tiles.
    """

    model.eval()
    metric.reset()

    total_batches = len(test_loader)

    for batch_index, batch in enumerate(test_loader):
        if len(batch) != 6:
            raise ValueError(
                "Expected each test batch to contain:\n"
                "coords, features, offset, labels, paths, metric_coords\n"
                f"Received {len(batch)} values."
            )

        (
            coords,
            features,
            offset,
            labels,
            paths,
            metric_coords,
        ) = batch

        coords_gpu = coords.to(
            device,
            non_blocking=True,
        ).float()

        features_gpu = features.to(
            device,
            non_blocking=True,
        ).float()

        offset_gpu = offset.to(
            device,
            non_blocking=True,
        ).long()

        model_output = model(
            (
                coords_gpu,
                features_gpu,
                offset_gpu,
            )
        )

        logits = extract_logits(
            model_output
        )

        if logits.ndim != 2:
            raise RuntimeError(
                "Expected logits with shape [N, C], "
                f"received {tuple(logits.shape)}."
            )

        if logits.shape[1] != 3:
            raise RuntimeError(
                "Expected three output classes, "
                f"received {logits.shape[1]}."
            )

        if logits.shape[0] != labels.shape[0]:
            raise RuntimeError(
                "Point-level prediction count does not match labels.\n"
                f"Logits: {tuple(logits.shape)}\n"
                f"Labels: {tuple(labels.shape)}\n"
                "The model must return predictions mapped back to "
                "the original input points."
            )

        predictions = (
            logits
            .argmax(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        targets = (
            labels
            .detach()
            .cpu()
            .numpy()
        )

        original_xyz = (
            metric_coords
            .detach()
            .cpu()
            .numpy()
        )

        tile_ranges = offsets_to_ranges(
            offset
        )

        for tile_index, (start, end) in enumerate(tile_ranges):
            tile_xyz = original_xyz[start:end]
            tile_predictions = predictions[start:end]
            tile_targets = targets[start:end]

            if tile_xyz.shape[0] == 0:
                continue

            if not (
                tile_xyz.shape[0]
                == tile_predictions.shape[0]
                == tile_targets.shape[0]
            ):
                raise RuntimeError(
                    "Tile arrays have different lengths for "
                    f"batch {batch_index}, tile {tile_index}."
                )

            # Explicit feature IDs are unavailable in the current LAZ
            # dataset, so the metric infers instances using connected
            # components.
            metric.update(
                xyz_m=tile_xyz,
                predicted_labels=tile_predictions,
                target_labels=tile_targets,
                gt_instance_ids=None,
            )

        if print_progress:
            should_print = (
                batch_index == 0
                or (batch_index + 1) % 5 == 0
                or batch_index + 1 == total_batches
            )

            if should_print:
                print(
                    f"Evaluated AVD batch "
                    f"{batch_index + 1}/{total_batches}"
                )

    return metric.compute()


def create_results_dataframe(
    avd_results: Dict,
) -> pd.DataFrame:
    """
    Convert per-class AVD results into a table.
    """

    rows = []
    config = avd_results["config"]

    for class_name, radius_results in (
        avd_results["per_class"].items()
    ):
        for radius_m, values in radius_results.items():
            total_predicted_cells = int(
                values["predicted_cells"]
            )

            useful_predicted_cells = int(
                values["predicted_cells_near_gt"]
            )

            false_predicted_cells = (
                total_predicted_cells
                - useful_predicted_cells
            )

            rows.append(
                {
                    "Mode / Class": class_name,
                    "Radius (m)": float(radius_m),
                    "Grid Cell Size (m)":
                        config["cell_size_m"],
                    "Minimum Points Per Cell":
                        config["min_points_per_cell"],
                    "Minimum Evidence Cells":
                        config["min_evidence_cells"],
                    "Beta":
                        config["beta"],
                    "GT Features":
                        int(values["gt_features"]),
                    "Detected Features":
                        int(
                            values[
                                "detected_gt_features"
                            ]
                        ),
                    "Vicinity Recall":
                        values["feature_recall"],
                    "Vicinity Precision":
                        values["vicinity_precision"],
                    "AVD-F2":
                        values["avd_fbeta"],
                    "Predicted Cells":
                        total_predicted_cells,
                    "Useful Predicted Cells":
                        useful_predicted_cells,
                    "False Predicted Cells":
                        false_predicted_cells,
                }
            )

    return pd.DataFrame(rows)


def create_macro_dataframe(
    avd_results: Dict,
) -> pd.DataFrame:
    """
    Calculate the archaeological macro AVD-F2 using only
    Structure and Plazuela.

    Ground is intentionally excluded.
    """

    rows = []

    radii = avd_results["config"]["radii_m"]

    for radius_m in radii:
        radius_m = float(radius_m)

        structure_score = (
            avd_results["per_class"]
            ["structure"]
            [radius_m]
            ["avd_fbeta"]
        )

        plazuela_score = (
            avd_results["per_class"]
            ["plazuela"]
            [radius_m]
            ["avd_fbeta"]
        )

        scores = np.asarray(
            [
                structure_score,
                plazuela_score,
            ],
            dtype=np.float64,
        )

        finite_scores = scores[
            np.isfinite(scores)
        ]

        macro_score = (
            float(finite_scores.mean())
            if finite_scores.size > 0
            else float("nan")
        )

        rows.append(
            {
                "Mode / Class":
                    "archaeological_macro",
                "Radius (m)":
                    radius_m,
                "Structure AVD-F2":
                    structure_score,
                "Plazuela AVD-F2":
                    plazuela_score,
                "Macro AVD-F2":
                    macro_score,
            }
        )

    return pd.DataFrame(rows)


def save_results(
    results_dataframe: pd.DataFrame,
    macro_dataframe: pd.DataFrame,
    output_dir,
) -> Dict[str, Path]:
    """
    Save AVD results to CSV.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir
        / "best_model_avd_results.csv"
    )

    macro_path = (
        output_dir
        / "best_model_avd_macro_results.csv"
    )

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    macro_dataframe.to_csv(
        macro_path,
        index=False,
    )

    return {
        "avd_results": results_path,
        "avd_macro_results": macro_path,
    }


def evaluate_best_model_avd(
    model,
    test_loader,
    checkpoint_path,
    output_dir,
    device: Optional[torch.device] = None,
    cell_size_m: float = 1.0,
    radii_m: Sequence[float] = (2.0, 5.0, 10.0),
    min_points_per_cell: int = 1,
    min_evidence_cells: int = 3,
    beta: float = 2.0,
    print_progress: bool = True,
    save_csv: bool = True,
) -> Dict:
    """
    Complete AVD evaluation wrapper used by the notebook.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    metric = create_metric(
        cell_size_m=cell_size_m,
        radii_m=radii_m,
        min_points_per_cell=min_points_per_cell,
        min_evidence_cells=min_evidence_cells,
        beta=beta,
    )

    avd_results = run_avd_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        metric=metric,
        print_progress=print_progress,
    )

    results_dataframe = create_results_dataframe(
        avd_results
    )

    macro_dataframe = create_macro_dataframe(
        avd_results
    )

    saved_paths = {}

    if save_csv:
        saved_paths = save_results(
            results_dataframe=results_dataframe,
            macro_dataframe=macro_dataframe,
            output_dir=output_dir,
        )

    print("\n" + metric.format_results())

    print(
        "\nCheckpoint epoch:",
        checkpoint.get(
            "epoch",
            "not stored",
        ),
    )

    return {
        "checkpoint": checkpoint,
        "raw_results": avd_results,
        "results_df": results_dataframe,
        "macro_df": macro_dataframe,
        "saved_paths": saved_paths,
    }