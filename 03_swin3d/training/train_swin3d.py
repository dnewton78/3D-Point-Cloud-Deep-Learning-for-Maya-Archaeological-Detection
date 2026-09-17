#!/usr/bin/env python3
"""
Production Swin3D-S training script for the Generalized Dice baseline.

Run this script from the project root containing:
    configs/
    datasets/
    losses/
    models/
    dataset_splits/

The script uses TRAIN_CONFIG from configs/swin3d_small.py.
"""

# ============================================================
# 1. Imports, reproducibility, and device
# ============================================================

import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.swin3d_small import TRAIN_CONFIG
from datasets.belize_laz_dataset import create_dataloaders
from losses.generalized_dice import GeneralizedDiceLoss
from models.swin3d_model import Model


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. Training configuration
# ============================================================

print("Training configuration:")
for key, value in TRAIN_CONFIG.items():
    print(f"  {key}: {value}")

required_keys = [
    "num_classes",
    "voxel_max",
    "batch_size",
    "augmentation_ratio",
    "max_epochs",
    "warmup_epochs",
    "early_stopping_patience",
    "learning_rate",
    "weight_decay",
]

missing_keys = [
    key for key in required_keys
    if key not in TRAIN_CONFIG
]

if missing_keys:
    raise KeyError(
        f"Missing TRAIN_CONFIG keys: {missing_keys}"
    )

if TRAIN_CONFIG["warmup_epochs"] >= TRAIN_CONFIG["max_epochs"]:
    raise ValueError(
        "warmup_epochs must be smaller than max_epochs."
    )


# ============================================================
# 3. Data loaders
# ============================================================

train_loader, validation_loader, test_loader = create_dataloaders(
    split_dir="dataset_splits",
    tile_manifest=TRAIN_CONFIG.get("tile_manifest"),
    batch_size=TRAIN_CONFIG["batch_size"],
    voxel_max=TRAIN_CONFIG["voxel_max"],
    num_workers=4,
    augmentation_ratio=TRAIN_CONFIG["augmentation_ratio"],
)

print("Training samples:", len(train_loader.dataset))
print("Validation samples:", len(validation_loader.dataset))
print("Test samples:", len(test_loader.dataset))

print()
print("Training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))
print("Test batches:", len(test_loader))

print()
print(
    "Augmentation ratio:",
    TRAIN_CONFIG["augmentation_ratio"],
)


# ============================================================
# 4. Batch and epoch helper functions
# ============================================================

def move_batch_to_device(batch, device):
    (
        coords,
        features,
        offset,
        labels,
        paths,
        metric_coords,
    ) = batch

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

    return coords, features, offset, labels


def calculate_batch_accuracy(logits, labels):
    predictions = logits.argmax(dim=1)

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.numel()

    return correct, total


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_points = 0

    for batch_index, batch in enumerate(loader):
        coords, features, offset, labels = (
            move_batch_to_device(
                batch,
                device,
            )
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            (coords, features, offset)
        )

        if isinstance(logits, tuple):
            logits = logits[0]

        if logits.shape[0] != labels.shape[0]:
            raise RuntimeError(
                "Logit and label point counts do not match: "
                f"logits={logits.shape}, labels={labels.shape}"
            )

        loss = criterion(
            logits,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss in batch {batch_index}: "
                f"{loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        correct, number_of_points = (
            calculate_batch_accuracy(
                logits,
                labels,
            )
        )

        total_loss += loss.item() * number_of_points
        total_correct += correct
        total_points += number_of_points

        if (
            batch_index == 0
            or (batch_index + 1) % 10 == 0
            or batch_index + 1 == len(loader)
        ):
            print(
                f"  Batch {batch_index + 1:04d}/"
                f"{len(loader):04d} | "
                f"Loss: {loss.item():.6f}"
            )

    average_loss = (
        total_loss / max(total_points, 1)
    )

    accuracy = (
        total_correct / max(total_points, 1)
    )

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_points = 0

    for batch in loader:
        coords, features, offset, labels = (
            move_batch_to_device(
                batch,
                device,
            )
        )

        logits = model(
            (coords, features, offset)
        )

        if isinstance(logits, tuple):
            logits = logits[0]

        if logits.shape[0] != labels.shape[0]:
            raise RuntimeError(
                "Logit and label point counts do not match: "
                f"logits={logits.shape}, labels={labels.shape}"
            )

        loss = criterion(
            logits,
            labels,
        )

        correct, number_of_points = (
            calculate_batch_accuracy(
                logits,
                labels,
            )
        )

        total_loss += loss.item() * number_of_points
        total_correct += correct
        total_points += number_of_points

    average_loss = (
        total_loss / max(total_points, 1)
    )

    accuracy = (
        total_correct / max(total_points, 1)
    )

    return average_loss, accuracy


# ============================================================
# 5. Model, loss, optimizer, and scheduler
# ============================================================

model = Model(
    num_classes=TRAIN_CONFIG["num_classes"],
    in_channels=4,
).to(device)

for parameter in model.parameters():
    parameter.requires_grad = True


criterion = GeneralizedDiceLoss(
    num_classes=TRAIN_CONFIG["num_classes"],
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=TRAIN_CONFIG["learning_rate"],
    weight_decay=TRAIN_CONFIG["weight_decay"],
)


warmup_epochs = TRAIN_CONFIG["warmup_epochs"]
max_epochs = TRAIN_CONFIG["max_epochs"]

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_epochs,
)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epochs - warmup_epochs,
    eta_min=1e-7,
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        warmup_scheduler,
        cosine_scheduler,
    ],
    milestones=[
        warmup_epochs,
    ],
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Loss:", criterion.__class__.__name__)
print("Optimizer:", optimizer.__class__.__name__)
print("Trainable parameters:", f"{trainable_parameters:,}")
print("Base learning rate:", TRAIN_CONFIG["learning_rate"])
print("Initial warm-up LR:", optimizer.param_groups[0]["lr"])
print("Warm-up epochs:", warmup_epochs)
print("Cosine epochs:", max_epochs - warmup_epochs)


# ============================================================
# 6. One-batch sanity check
# ============================================================

batch = next(iter(train_loader))

coords, features, offset, labels = move_batch_to_device(
    batch,
    device,
)

print("Coordinates:", coords.shape)
print("Features:", features.shape)
print("Offsets:", offset.shape)
print("Labels:", labels.shape)
print(
    "Input channels:",
    coords.shape[1] + features.shape[1],
)

model.eval()

with torch.no_grad():
    logits = model(
        (coords, features, offset)
    )

if isinstance(logits, tuple):
    logits = logits[0]

print("Logits:", logits.shape)

if logits.shape[0] != labels.shape[0]:
    raise RuntimeError(
        "Sanity check failed: model output and labels "
        "have different point counts."
    )

if logits.shape[1] != TRAIN_CONFIG["num_classes"]:
    raise RuntimeError(
        "Sanity check failed: wrong number of classes."
    )

print("Sanity check passed.")


# ============================================================
# 7. Checkpoint paths
# ============================================================

checkpoint_dir = Path("checkpoints/generalized_dice")

checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True,
)

best_checkpoint_path = (
    checkpoint_dir /
    "swin3d_generalized_dice_best_01.pth"
)

last_checkpoint_path = (
    checkpoint_dir /
    "swin3d_generalized_dice_last_01.pth"
)

print("Best checkpoint:", best_checkpoint_path)
print("Last checkpoint:", last_checkpoint_path)


# ============================================================
# 8. Train
# ============================================================

best_validation_loss = float("inf")
epochs_without_improvement = 0

patience = TRAIN_CONFIG["early_stopping_patience"]

training_history = {
    "train_loss": [],
    "train_accuracy": [],
    "validation_loss": [],
    "validation_accuracy": [],
    "learning_rate": [],
}


for epoch in range(max_epochs):

    epoch_number = epoch + 1
    current_learning_rate = optimizer.param_groups[0]["lr"]

    print(
        f"\n{'=' * 65}\n"
        f"Epoch {epoch_number}/{max_epochs} | "
        f"LR: {current_learning_rate:.8f}\n"
        f"{'=' * 65}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_loss, validation_accuracy = evaluate(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    training_history["train_loss"].append(
        train_loss
    )

    training_history["train_accuracy"].append(
        train_accuracy
    )

    training_history["validation_loss"].append(
        validation_loss
    )

    training_history["validation_accuracy"].append(
        validation_accuracy
    )

    training_history["learning_rate"].append(
        current_learning_rate
    )

    print(
        f"\nEpoch {epoch_number:03d} results\n"
        f"Train loss:       {train_loss:.6f}\n"
        f"Train accuracy:   {train_accuracy:.4f}\n"
        f"Validation loss:  {validation_loss:.6f}\n"
        f"Validation acc:   {validation_accuracy:.4f}\n"
        f"Learning rate:    {current_learning_rate:.8f}"
    )

    # --------------------------------------------------------
    # Best-model logic
    # --------------------------------------------------------

    improved = (
        validation_loss < best_validation_loss
    )

    if improved:
        best_validation_loss = validation_loss
        epochs_without_improvement = 0

        print(
            f"New best validation loss: "
            f"{best_validation_loss:.6f}"
        )

    elif epoch_number > warmup_epochs:
        epochs_without_improvement += 1

        print(
            "No validation improvement: "
            f"{epochs_without_improvement}/{patience}"
        )

    else:
        print(
            "No validation improvement. "
            "Early stopping is disabled during warm-up."
        )

    # --------------------------------------------------------
    # Step scheduler
    # --------------------------------------------------------

    scheduler.step()

    # --------------------------------------------------------
    # Save checkpoint AFTER scheduler step
    # --------------------------------------------------------

    checkpoint = {
        "epoch": epoch_number,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "training_history": training_history,
        "config": TRAIN_CONFIG,
    }

    torch.save(
        checkpoint,
        last_checkpoint_path,
    )

    if improved:
        torch.save(
            checkpoint,
            best_checkpoint_path,
        )

        print(
            "Saved best model:",
            best_checkpoint_path,
        )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epoch_number > warmup_epochs
        and epochs_without_improvement >= patience
    ):
        print(
            f"\nEarly stopping at epoch {epoch_number}."
        )

        print(
            "Best validation loss:",
            f"{best_validation_loss:.6f}",
        )

        break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 9. Save training history plots
# ============================================================

epochs = np.arange(
    1,
    len(training_history["train_loss"]) + 1,
)

loss_plot_path = checkpoint_dir / "training_validation_loss_01.png"
accuracy_plot_path = checkpoint_dir / "training_validation_accuracy_01.png"
lr_plot_path = checkpoint_dir / "learning_rate_schedule_01.png"

plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    training_history["train_loss"],
    label="Train",
)
plt.plot(
    epochs,
    training_history["validation_loss"],
    label="Validation",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_plot_path, dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    training_history["train_accuracy"],
    label="Train",
)
plt.plot(
    epochs,
    training_history["validation_accuracy"],
    label="Validation",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(accuracy_plot_path, dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(
    epochs,
    training_history["learning_rate"],
)
plt.axvline(
    warmup_epochs,
    linestyle="--",
    label="End of warm-up",
)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(lr_plot_path, dpi=200)
plt.close()

print("Saved loss plot:", loss_plot_path)
print("Saved accuracy plot:", accuracy_plot_path)
print("Saved learning-rate plot:", lr_plot_path)


# ============================================================
# 10. Load best checkpoint and evaluate test set
# ============================================================

best_checkpoint = torch.load(
    best_checkpoint_path,
    map_location=device,
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

print(
    "Loaded best checkpoint from epoch:",
    best_checkpoint["epoch"],
)

print(
    "Best validation loss:",
    best_checkpoint["validation_loss"],
)


test_loss, test_accuracy = evaluate(
    model=model,
    loader=test_loader,
    criterion=criterion,
    device=device,
)

print()
print("Final test results")
print("------------------")
print(f"Test loss:     {test_loss:.6f}")
print(f"Test accuracy: {test_accuracy:.4f}")
