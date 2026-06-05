"""
train_model.py — IDISCDualTower Training Script
================================================
Project:  Translation Task Assignment System (IDISC)
Module:   Backend / Algorithms / Dual-Tower MLP

This script orchestrates the full training pipeline for the IDISCDualTower
Recommender System. It handles tensor loading, model instantiation, the full
training loop with validation, a ReduceLROnPlateau scheduler, early stopping,
model checkpointing, and a terminal loss-curve plot.

Pipeline Stage: Step 3 of 4
    [✓] Step 1 — Preprocessing (preprocessing_pipeline.py)
    [✓] Step 2 — Tensor Creation (data_to_tensors.py)
    [►] Step 3 — Model Training  (this file)
    [ ] Step 4 — Evaluation & Inference

Usage:
    python train_model.py

Output:
    best_idisc_model.pth — Saved state_dict of the best checkpoint.
    training_log.csv     — Epoch-by-epoch loss history for offline analysis.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import sys
import time
import math
import csv
import pathlib

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------------
# Local Architecture Import
# ---------------------------------------------------------------------------
# We import directly from the Architecture sub-package.
# The `sys.path` insertion ensures this works regardless of the CWD from
# which the script is invoked (e.g., from the repo root or this directory).
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "Architecture"))

from IDISC_DualTower import IDISCDualTower  # noqa: E402  (after sys.path edit)


# ===========================================================================
# --- CONFIGURATION ----------------------------------------------------------
# ===========================================================================
# All hyperparameters are declared here as named constants so they are easy
# to find, adjust, and document. No magic numbers inside the training loop.

# --- Paths ---
TENSORS_DIR        = _SCRIPT_DIR / "Data to Tensors" / "tensors"
CHECKPOINT_PATH    = _SCRIPT_DIR / "best_idisc_model.pth"
LOG_PATH           = _SCRIPT_DIR / "training_log.csv"

# --- Model ---
EMBEDDING_DIM      = 64     # Shared projection space; must match your architecture expectations

# --- Training ---
BATCH_SIZE         = 128     # Pairs per gradient update step
NUM_EPOCHS         = 50     # Maximum training epochs before forced stop
LEARNING_RATE      = 3e-3   # AdamW initial learning rate
WEIGHT_DECAY       = 1e-4   # L2 regularisation coefficient (mild)

# --- Scheduler (ReduceLROnPlateau) ---
LR_FACTOR          = 0.5    # Multiply LR by this factor on plateau
LR_PATIENCE        = 3      # Epochs with no val improvement before LR reduction
LR_MIN             = 1e-6   # Lower bound on LR to prevent it from hitting zero

# --- Early Stopping ---
EARLY_STOP_PATIENCE = 7     # Epochs with no val improvement before training halts

# --- Reproducibility ---
SEED               = 42


# ===========================================================================
# --- UTILITIES --------------------------------------------------------------
# ===========================================================================

def set_seed(seed: int) -> None:
    """Pin all random sources for reproducible runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon fallback
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tensor(path: pathlib.Path, label: str) -> torch.Tensor:
    """Load a .pt tensor from disk with a validation print."""
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] Required tensor not found: {path}\n"
            f"        Re-run data_to_tensors.py to regenerate all tensors."
        )
    t = torch.load(path, weights_only=True)
    print(f"  Loaded {label:<30}  shape={tuple(t.shape)}  dtype={t.dtype}")
    return t


def ascii_loss_plot(
    train_losses: list[float],
    val_losses:   list[float],
    width:        int = 65,
    height:       int = 12,
) -> None:
    """
    Render a compact ASCII loss curve directly in the terminal.

    Draws two lines (train = '*', val = 'o') scaled to the same y-axis.
    This gives a quick, dependency-free visual of convergence.
    """
    if len(train_losses) < 2:
        return  # Nothing meaningful to plot yet

    all_vals = train_losses + val_losses
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = y_max - y_min or 1e-8  # Avoid div-by-zero on flat loss curves

    n_epochs = len(train_losses)
    # Map each epoch to a column index in [0, width-1]
    def col(ep: int) -> int:
        return int((ep / (n_epochs - 1)) * (width - 1)) if n_epochs > 1 else 0

    # Pre-fill a blank canvas (list of lists for mutability)
    canvas = [[" "] * width for _ in range(height)]

    # Plot both series
    for series, marker in [(train_losses, "*"), (val_losses, "o")]:
        for ep, val in enumerate(series):
            # Invert y: row 0 is top (high loss), row height-1 is bottom (low loss)
            row = int((1.0 - (val - y_min) / y_range) * (height - 1))
            row = max(0, min(height - 1, row))
            c   = col(ep)
            canvas[row][c] = marker

    # Print the chart
    print("\n" + "=" * (width + 6))
    print(f"  Loss Curve  (* Train  o Val)   epochs: {n_epochs}")
    print("=" * (width + 6))
    print(f"  {y_max:.4f} ┐")
    for row in canvas:
        print("          │" + "".join(row))
    print(f"  {y_min:.4f} ┘")
    print("           " + "-" * width)
    print(f"           Epoch 1{' ' * (width - 14)}Epoch {n_epochs}")
    print("=" * (width + 6) + "\n")


# ===========================================================================
# --- MAIN TRAINING FUNCTION -------------------------------------------------
# ===========================================================================

def train() -> None:
    """
    Full training pipeline for IDISCDualTower.

    Steps executed:
        1. Seed & device selection
        2. Load .pt tensors from disk
        3. Build TensorDataset and DataLoader objects
        4. Dynamically instantiate IDISCDualTower
        5. Define loss, optimizer, and LR scheduler
        6. Execute training loop with:
               - model.train() / model.eval() toggling
               - torch.no_grad() during validation
               - per-epoch logging
               - best-model checkpointing (val loss)
               - ReduceLROnPlateau scheduler step
               - early stopping
        7. Print ASCII loss curve
        8. Save epoch-by-epoch CSV log
    """

    # -----------------------------------------------------------------------
    # 1. Seed & Device
    # -----------------------------------------------------------------------
    set_seed(SEED)
    device = get_device()

    print("=" * 65)
    print("  IDISCDualTower — Training Pipeline")
    print("=" * 65)
    print(f"\n  Device  : {device}")
    print(f"  Tensors : {TENSORS_DIR}")
    print(f"  Seed    : {SEED}\n")

    # -----------------------------------------------------------------------
    # 2. Load Tensors
    # -----------------------------------------------------------------------
    print("-" * 65)
    print("  STEP 1 / 5 — Loading tensors from disk")
    print("-" * 65)

    train_a      = load_tensor(TENSORS_DIR / "train_tower_a.pt", "train_tower_a")
    train_b      = load_tensor(TENSORS_DIR / "train_tower_b.pt", "train_tower_b")
    train_target = load_tensor(TENSORS_DIR / "train_target.pt",  "train_target")

    val_a        = load_tensor(TENSORS_DIR / "val_tower_a.pt",   "val_tower_a")
    val_b        = load_tensor(TENSORS_DIR / "val_tower_b.pt",   "val_tower_b")
    val_target   = load_tensor(TENSORS_DIR / "val_target.pt",    "val_target")

    # Guarantee float32 on all tensors (scalers may have produced float64)
    for name, t in [("train_a", train_a), ("train_b", train_b),
                    ("val_a",   val_a),   ("val_b",   val_b)]:
        if t.dtype != torch.float32:
            print(f"  [WARN] {name} is {t.dtype} — casting to float32.")

    train_a      = train_a.float()
    train_b      = train_b.float()
    train_target = train_target.float()
    val_a        = val_a.float()
    val_b        = val_b.float()
    val_target   = val_target.float()

    # MSELoss expects (B, 1) targets; unsqueeze if targets are flat (B,) vectors
    if train_target.dim() == 1:
        train_target = train_target.unsqueeze(1)  # (N,) -> (N, 1)
    if val_target.dim() == 1:
        val_target = val_target.unsqueeze(1)

    print(f"\n  Train pairs : {train_a.shape[0]:,}")
    print(f"  Val   pairs : {val_a.shape[0]:,}")

    # -----------------------------------------------------------------------
    # 3. Build DataLoaders
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("  STEP 2 / 5 — Building DataLoaders")
    print("-" * 65)

    train_dataset = TensorDataset(train_a, train_b, train_target)
    val_dataset   = TensorDataset(val_a,   val_b,   val_target)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,          # Shuffle train set each epoch to decorrelate batches
        drop_last=True,        # Drop last incomplete batch (BatchNorm needs ≥2 samples)
        pin_memory=(device.type == "cuda"),  # Async CPU->GPU transfer when on CUDA
        num_workers=2,         # Keep 0 for Windows (avoids multiprocessing fork issues)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,         # DO NOT shuffle validation — deterministic evaluation
        drop_last=False,       # Evaluate every sample
        pin_memory=(device.type == "cuda"),
        num_workers=2,
    )

    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)
    print(f"  Train batches / epoch : {n_train_batches:,}  (batch_size={BATCH_SIZE}, drop_last=True)")
    print(f"  Val   batches / epoch : {n_val_batches:,}  (batch_size={BATCH_SIZE})")

    # -----------------------------------------------------------------------
    # 4. Instantiate Model — Dynamically from tensor shapes
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("  STEP 3 / 5 — Instantiating IDISCDualTower")
    print("-" * 65)

    # Extract feature dimensions directly from the loaded tensors.
    # This means we never have to manually update these constants when
    # the feature engineering step adds or removes columns.
    tower_a_dim = train_a.shape[1]   # e.g. 132 (Translator/Employee features)
    tower_b_dim = train_b.shape[1]   # e.g. 140 (Task features)

    model = IDISCDualTower(
        tower_a_input_dim=tower_a_dim,
        tower_b_input_dim=tower_b_dim,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    # Count trainable parameters for the training header
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  tower_a_input_dim : {tower_a_dim}")
    print(f"  tower_b_input_dim : {tower_b_dim}")
    print(f"  embedding_dim     : {EMBEDDING_DIM}")
    print(f"  Trainable params  : {n_params:,}")

    # -----------------------------------------------------------------------
    # 5. Loss, Optimizer, Scheduler
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("  STEP 4 / 5 — Loss / Optimizer / Scheduler")
    print("-" * 65)

    # MSELoss — our targets are continuous affinity scores in [0, 1]
    criterion = nn.BCELoss()

    # AdamW — Adam with decoupled weight decay (better regularisation than Adam+L2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ReduceLROnPlateau — halve the LR whenever val_loss stops improving
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #        optimizer,
    #        mode="min",           # We want to minimise val_loss
    #        factor=LR_FACTOR,
    #        patience=LR_PATIENCE,
    #        min_lr=LR_MIN,
    #        verbose=False,        # We'll log LR changes ourselves for clean output
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=n_train_batches,
        anneal_strategy='cos',
        verbose=False,        # We'll log LR changes ourselves for clean output
    )

    print(f"  Loss      : nn.MSELoss")
    print(f"  Optimizer : AdamW  (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler : OneCycleLR  (factor={LR_FACTOR}, patience={LR_PATIENCE})")

    # -----------------------------------------------------------------------
    # 6. Training Loop
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("  STEP 5 / 5 — Training Loop")
    print("-" * 65)
    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  Early Stop    : patience={EARLY_STOP_PATIENCE}")
    print(f"  Checkpoint    : {CHECKPOINT_PATH.name}")
    print("-" * 65)

    # -- State trackers --------------------------------------------------------
    best_val_loss    = math.inf    # Best validation loss seen so far
    no_improve_count = 0           # Consecutive epochs without improvement
    train_loss_hist  = []          # History for the ASCII plot
    val_loss_hist    = []          # History for the ASCII plot
    log_rows         = []          # Rows for CSV export

    # -- Column header for epoch table ----------------------------------------
    COL_W = 12
    header = (
        f"{'Epoch':>{COL_W}}"
        f"{'Train Loss':>{COL_W}}"
        f"{'Val Loss':>{COL_W}}"
        f"{'LR':>{COL_W}}"
        f"  {'Status'}"
    )
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    epoch_start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):

        # -- 6a. TRAINING PHASE ---------------------------------------------
        model.train()   # Activates Dropout layers and BatchNorm training stats
        running_train_loss = 0.0

        for batch_a, batch_b, batch_y in train_loader:
            # Move batch to the target device (CUDA/MPS/CPU)
            batch_a = batch_a.to(device, non_blocking=True)
            batch_b = batch_b.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # Zero gradients from the previous step before computing new ones
            optimizer.zero_grad()

            # Forward pass — returns affinity_score of shape (B, 1)
            predictions = model(batch_a, batch_b)

            # Compute Mean Squared Error loss
            loss = criterion(predictions, batch_y)

            # Backpropagation: compute ∂loss/∂θ for all parameters
            loss.backward()

            # Gradient clipping: prevent catastrophic gradient explosions.
            # Clips the global gradient norm to 1.0. This is a safety net
            # particularly important in the early epochs of training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters via AdamW step
            optimizer.step()

            running_train_loss += loss.item()

        # Average loss across all training batches this epoch
        avg_train_loss = running_train_loss / n_train_batches

        # -- 6b. VALIDATION PHASE -------------------------------------------
        model.eval()    # Disables Dropout; BatchNorm uses running statistics
        running_val_loss = 0.0

        with torch.no_grad():  # No gradients needed — saves memory & compute
            for batch_a, batch_b, batch_y in val_loader:
                batch_a = batch_a.to(device, non_blocking=True)
                batch_b = batch_b.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                predictions      = model(batch_a, batch_b)
                loss             = criterion(predictions, batch_y)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / n_val_batches

        # -- 6c. SCHEDULER STEP ---------------------------------------------
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        curr_lr = optimizer.param_groups[0]["lr"]
        lr_changed = curr_lr < prev_lr  # Flag to notify user of LR reduction

        # -- 6d. CHECKPOINTING ----------------------------------------------
        status_parts = []
        saved = False

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            no_improve_count = 0   # Reset the patience counter on improvement

            # Save only the learnable parameters (state_dict), not the full
            # model object. This is the standard PyTorch portability practice.
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            status_parts.append("✓ SAVED")
            saved = True
        else:
            no_improve_count += 1
            status_parts.append(f"no improv {no_improve_count}/{EARLY_STOP_PATIENCE}")

        if lr_changed:
            status_parts.append(f"LR->{curr_lr:.2e}")

        # -- 6e. LOGGING ----------------------------------------------------
        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        log_rows.append({
            "epoch":      epoch,
            "train_loss": avg_train_loss,
            "val_loss":   avg_val_loss,
            "lr":         curr_lr,
            "saved":      saved,
        })

        # Print one compact row per epoch
        status_str = " | ".join(status_parts)
        print(
            f"{epoch:>{COL_W}}"
            f"{avg_train_loss:>{COL_W}.6f}"
            f"{avg_val_loss:>{COL_W}.6f}"
            f"{curr_lr:>{COL_W}.2e}"
            f"  {status_str}"
        )

        # -- 6f. EARLY STOPPING ---------------------------------------------
        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(
                f"\n  [Early Stop] No improvement for {EARLY_STOP_PATIENCE} consecutive "
                f"epochs. Stopping at epoch {epoch}."
            )
            break

    # -----------------------------------------------------------------------
    # Post-Training Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - epoch_start_time
    epochs_run = len(train_loss_hist)

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Epochs run      : {epochs_run} / {NUM_EPOCHS}")
    print(f"  Best val loss   : {best_val_loss:.6f}")
    print(f"  Time elapsed    : {elapsed / 60:.1f} min  ({elapsed:.1f} s)")
    print(f"  Checkpoint      : {CHECKPOINT_PATH}")

    # -- ASCII Loss Curve ----------------------------------------------------
    ascii_loss_plot(train_loss_hist, val_loss_hist)

    # -- Save CSV Log --------------------------------------------------------
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr", "saved"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"  Training log    : {LOG_PATH}\n")


# ===========================================================================
# --- ENTRY POINT ------------------------------------------------------------
# ===========================================================================

if __name__ == "__main__":
    train()
