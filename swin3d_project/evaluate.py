from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch


DEFAULT_CLASS_NAMES = [
    "Ground",
    "Structure",
    "Plazuela",
]


def extract_logits(model_output):
    """
    Extract the logits tensor from the model output.

    Supports:
        - torch.Tensor
        - tuple/list whose first item is the logits tensor
        - object containing an .F feature tensor
    """

    if torch.is_tensor(model_output):
        return model_output

    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise RuntimeError("The model returned an empty tuple/list.")

        model_output = model_output[0]

        if torch.is_tensor(model_output):
            return model_output

        if hasattr(model_output, "F"):
            return model_output.F

    if hasattr(model_output, "F"):
        return model_output.F

    raise TypeError(
        "Unsupported model output type: "
        f"{type(model_output)}"
    )


def unpack_test_batch(
    batch,
    device,
):
    """
    Supports batches containing:

    4 items:
        coords, features, offset, labels

    5 items:
        coords, features, offset, labels, paths

    6 items:
        coords, features, offset, labels, paths, metric_coords
    """

    if len(batch) == 4:
        (
            coords,
            features,
            offset,
            labels,
        ) = batch

        paths = None
        metric_coords = None

    elif len(batch) == 5:
        (
            coords,
            features,
            offset,
            labels,
            paths,
        ) = batch

        metric_coords = None

    elif len(batch) == 6:
        (
            coords,
            features,
            offset,
            labels,
            paths,
            metric_coords,
        ) = batch

    else:
        raise ValueError(
            "Expected a batch with 4, 5, or 6 items, "
            f"but received {len(batch)} items."
        )

    coords = coords.to(
        device,
        non_blocking=True,
    ).float()

    features = features.to(
        device,
        non_blocking=True,
    ).float()

    offset = offset.to(
        device,
        non_blocking=True,
    ).long()

    labels = labels.to(
        device,
        non_blocking=True,
    ).long()

    if metric_coords is not None:
        metric_coords = metric_coords.to(
            device,
            non_blocking=True,
        )

    return (
        coords,
        features,
        offset,
        labels,
        paths,
        metric_coords,
    )


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Add one batch to the confusion matrix.

    Matrix orientation:

        rows    = actual/ground-truth classes
        columns = predicted classes
    """

    predictions = predictions.reshape(-1).long()
    labels = labels.reshape(-1).long()

    valid = (
        (labels >= 0)
        & (labels < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )

    labels = labels[valid]
    predictions = predictions[valid]

    encoded = (
        labels * num_classes
        + predictions
    )

    batch_confusion_matrix = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(
        num_classes,
        num_classes,
    )

    confusion_matrix += batch_confusion_matrix.cpu()

    return confusion_matrix


@torch.inference_mode()
def collect_test_confusion_matrix(
    model,
    test_loader,
    device: torch.device,
    num_classes: int,
    print_progress: bool = True,
) -> torch.Tensor:
    """
    Run test-set inference and return the accumulated confusion matrix.
    """

    model.eval()

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
    )

    total_batches = len(test_loader)

    for batch_index, batch in enumerate(test_loader):
        (
            coords,
            features,
            offset,
            labels,
            _,
            _,
        ) = unpack_test_batch(
            batch=batch,
            device=device,
        )

        model_output = model(
            (coords, features, offset)
        )

        logits = extract_logits(
            model_output
        )

        if logits.ndim != 2:
            raise RuntimeError(
                "Expected logits with shape [N, num_classes], "
                f"but received {tuple(logits.shape)}."
            )

        if logits.shape[1] != num_classes:
            raise RuntimeError(
                f"Expected {num_classes} output classes, but logits "
                f"have shape {tuple(logits.shape)}."
            )

        if logits.shape[0] != labels.shape[0]:
            raise RuntimeError(
                "The number of model predictions does not match the "
                "number of point labels.\n"
                f"Logits: {tuple(logits.shape)}\n"
                f"Labels: {tuple(labels.shape)}\n"
                "Make sure your model maps voxel predictions back to "
                "the original input points."
            )

        predictions = logits.argmax(
            dim=1
        )

        confusion_matrix = update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            predictions=predictions,
            labels=labels,
            num_classes=num_classes,
        )

        if print_progress:
            should_print = (
                batch_index == 0
                or (batch_index + 1) % 5 == 0
                or batch_index + 1 == total_batches
            )

            if should_print:
                print(
                    f"Evaluated test batch "
                    f"{batch_index + 1}/{total_batches}"
                )

    return confusion_matrix


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """
    Divide while returning zero when the denominator is zero.
    """

    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def calculate_metrics(
    confusion_matrix: torch.Tensor,
    class_names: List[str],
) -> Dict:
    """
    Calculate overall and per-class segmentation metrics.
    """

    if confusion_matrix.shape[0] != len(class_names):
        raise ValueError(
            "The confusion-matrix size and number of class names "
            "do not match."
        )

    cm = confusion_matrix.to(
        dtype=torch.float64
    )

    total = cm.sum()

    true_positive = torch.diag(cm)

    false_positive = (
        cm.sum(dim=0)
        - true_positive
    )

    false_negative = (
        cm.sum(dim=1)
        - true_positive
    )

    true_negative = (
        total
        - true_positive
        - false_positive
        - false_negative
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    iou = safe_divide(
        true_positive,
        (
            true_positive
            + false_positive
            + false_negative
        ),
    )

    if total > 0:
        overall_accuracy = (
            true_positive.sum() / total
        ).item()
    else:
        overall_accuracy = 0.0

    # Ignore only classes that are completely absent from both
    # ground truth and predictions.
    valid_classes = (
        true_positive
        + false_positive
        + false_negative
    ) > 0

    if valid_classes.any():
        precision_average = precision[
            valid_classes
        ].mean().item()

        recall_average = recall[
            valid_classes
        ].mean().item()

        f1_average = f1[
            valid_classes
        ].mean().item()

        mean_iou = iou[
            valid_classes
        ].mean().item()

    else:
        precision_average = 0.0
        recall_average = 0.0
        f1_average = 0.0
        mean_iou = 0.0

    per_class = {}

    for class_index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": precision[class_index].item(),
            "recall": recall[class_index].item(),
            "f1": f1[class_index].item(),
            "iou": iou[class_index].item(),
            "tp": int(true_positive[class_index].item()),
            "fp": int(false_positive[class_index].item()),
            "fn": int(false_negative[class_index].item()),
            "tn": int(true_negative[class_index].item()),
            "support": int(
                cm[class_index, :].sum().item()
            ),
        }

    return {
        "overall_accuracy": overall_accuracy,
        "precision_average": precision_average,
        "recall_average": recall_average,
        "f1_average": f1_average,
        "mean_iou": mean_iou,
        "per_class": per_class,
    }


def create_results_dataframe(
    metrics: Dict,
) -> pd.DataFrame:
    """
    Create one spreadsheet-style row matching the requested columns.
    """

    ground = metrics["per_class"]["Ground"]
    structure = metrics["per_class"]["Structure"]
    plazuela = metrics["per_class"]["Plazuela"]

    result = {
        "Test Set: Overall (Pixel / Voxel) Accuracy (OA)":
            metrics["overall_accuracy"],

        "Test Set: F1 Average":
            metrics["f1_average"],

        "Test Set: F1 - Ground":
            ground["f1"],

        "Test Set: F1 - Structure":
            structure["f1"],

        "Test Set: F1 - Plazuelas":
            plazuela["f1"],

        "Test Set: Precision Average":
            metrics["precision_average"],

        "Test Set: Precision - Ground":
            ground["precision"],

        "Test Set: Precision - Structures":
            structure["precision"],

        "Test Set: Precision - Plazuelas":
            plazuela["precision"],

        "Test Set: Recall Average":
            metrics["recall_average"],

        "Test Set: Recall - Ground":
            ground["recall"],

        "Test Set: Recall - Structure":
            structure["recall"],

        "Test Set: Recall - Plazuelas":
            plazuela["recall"],

        "Test Set: mIoU (mean Intersection over Union)":
            metrics["mean_iou"],

        "Test Set: IoU Ground":
            ground["iou"],

        "Test Set: IoU Structure":
            structure["iou"],

        "Test Set: IoU Plazuela":
            plazuela["iou"],

        "Test Set: Confusion Matrix TP / FP / FN / TN: Ground":
            (
                f"TP={ground['tp']}; "
                f"FP={ground['fp']}; "
                f"FN={ground['fn']}; "
                f"TN={ground['tn']}"
            ),

        "Test Set: Confusion Matrix TP / FP / FN / TN: Structures":
            (
                f"TP={structure['tp']}; "
                f"FP={structure['fp']}; "
                f"FN={structure['fn']}; "
                f"TN={structure['tn']}"
            ),

        "Test Set: Confusion Matrix TP / FP / FN / TN: Plazuelas":
            (
                f"TP={plazuela['tp']}; "
                f"FP={plazuela['fp']}; "
                f"FN={plazuela['fn']}; "
                f"TN={plazuela['tn']}"
            ),
    }

    return pd.DataFrame(
        [result]
    )


def create_per_class_dataframe(
    metrics: Dict,
) -> pd.DataFrame:
    """
    Create a readable per-class metrics table.
    """

    rows = []

    for class_name, values in metrics["per_class"].items():
        rows.append(
            {
                "Class": class_name,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1": values["f1"],
                "IoU": values["iou"],
                "TP": values["tp"],
                "FP": values["fp"],
                "FN": values["fn"],
                "TN": values["tn"],
                "Support": values["support"],
            }
        )

    return pd.DataFrame(rows)


def create_confusion_matrix_dataframe(
    confusion_matrix: torch.Tensor,
    class_names: List[str],
) -> pd.DataFrame:
    """
    Convert the confusion matrix to a labeled DataFrame.
    """

    return pd.DataFrame(
        confusion_matrix.numpy(),
        index=[
            f"Actual {name}"
            for name in class_names
        ],
        columns=[
            f"Predicted {name}"
            for name in class_names
        ],
    )


def print_metrics(
    metrics: Dict,
    best_epoch: Optional[int] = None,
):
    """
    Print overall and per-class results.
    """

    print("\n" + "=" * 70)
    print("BEST-CHECKPOINT TEST RESULTS")
    print("=" * 70)

    if best_epoch is not None:
        print(f"Best epoch:        {best_epoch}")

    print(
        f"Overall Accuracy:  "
        f"{metrics['overall_accuracy']:.6f}"
    )

    print(
        f"F1 Average:        "
        f"{metrics['f1_average']:.6f}"
    )

    print(
        f"Precision Average: "
        f"{metrics['precision_average']:.6f}"
    )

    print(
        f"Recall Average:    "
        f"{metrics['recall_average']:.6f}"
    )

    print(
        f"mIoU:              "
        f"{metrics['mean_iou']:.6f}"
    )

    for class_name, values in metrics["per_class"].items():
        print(f"\n{class_name}")
        print(f"  Precision: {values['precision']:.6f}")
        print(f"  Recall:    {values['recall']:.6f}")
        print(f"  F1:        {values['f1']:.6f}")
        print(f"  IoU:       {values['iou']:.6f}")
        print(f"  Support:   {values['support']:,}")

        print(
            "  Confusion: "
            f"TP={values['tp']:,}, "
            f"FP={values['fp']:,}, "
            f"FN={values['fn']:,}, "
            f"TN={values['tn']:,}"
        )


def load_best_checkpoint(
    model,
    checkpoint_path,
    device: torch.device,
) -> Dict:
    """
    Load model weights from the best checkpoint.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return checkpoint


def save_evaluation_results(
    output_dir,
    results_dataframe: pd.DataFrame,
    per_class_dataframe: pd.DataFrame,
    confusion_dataframe: pd.DataFrame,
    best_epoch: Optional[int],
):
    """
    Save all evaluation results as CSV files.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir
        / "best_epoch_test_metrics.csv"
    )

    per_class_path = (
        output_dir
        / "best_epoch_per_class_metrics.csv"
    )

    confusion_path = (
        output_dir
        / "best_epoch_confusion_matrix.csv"
    )

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    per_class_dataframe.to_csv(
        per_class_path,
        index=False,
    )

    confusion_dataframe.to_csv(
        confusion_path,
    )

    metadata_path = (
        output_dir
        / "best_epoch_information.txt"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"Best epoch: {best_epoch}\n"
        )

    return {
        "metrics": results_path,
        "per_class": per_class_path,
        "confusion_matrix": confusion_path,
        "metadata": metadata_path,
    }


def evaluate_best_model(
    model,
    test_loader,
    checkpoint_path,
    output_dir,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
    print_progress: bool = True,
    save_results: bool = True,
) -> Tuple[
    Dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict,
]:
    """
    Complete best-checkpoint evaluation pipeline.

    This function:

        1. Loads the best checkpoint.
        2. Evaluates the test set.
        3. Builds the confusion matrix.
        4. Calculates all requested metrics.
        5. Creates result DataFrames.
        6. Saves CSV files.
        7. Returns all results.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    num_classes = len(class_names)

    checkpoint = load_best_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    best_epoch = checkpoint.get(
        "epoch"
    )

    print(
        f"Loaded best checkpoint from:\n"
        f"{checkpoint_path}"
    )

    if best_epoch is not None:
        print(f"Checkpoint epoch: {best_epoch}")

    confusion_matrix = collect_test_confusion_matrix(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        print_progress=print_progress,
    )

    metrics = calculate_metrics(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
    )

    results_dataframe = create_results_dataframe(
        metrics=metrics
    )

    per_class_dataframe = create_per_class_dataframe(
        metrics=metrics
    )

    confusion_dataframe = (
        create_confusion_matrix_dataframe(
            confusion_matrix=confusion_matrix,
            class_names=class_names,
        )
    )

    print_metrics(
        metrics=metrics,
        best_epoch=best_epoch,
    )

    saved_paths = {}

    if save_results:
        saved_paths = save_evaluation_results(
            output_dir=output_dir,
            results_dataframe=results_dataframe,
            per_class_dataframe=per_class_dataframe,
            confusion_dataframe=confusion_dataframe,
            best_epoch=best_epoch,
        )

        print("\nSaved evaluation files:")

        for name, path in saved_paths.items():
            print(f"  {name}: {path}")

    return (
        metrics,
        results_dataframe,
        per_class_dataframe,
        confusion_dataframe,
        saved_paths,
    )