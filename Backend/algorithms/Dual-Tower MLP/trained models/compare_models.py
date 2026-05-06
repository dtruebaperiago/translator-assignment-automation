"""
compare_models.py — Multi-Model Training Comparison Plotter
============================================================
Discovers every model subfolder inside 'trained models/' that contains a
training_log.csv, reads an optional config.json for hyperparameter metadata,
then groups models by their 'loss_criterion' field and produces one separate
3-panel comparison figure per criterion group:

  Panel 1 — Training Loss vs Epoch  (solid lines)
  Panel 2 — Validation Loss vs Epoch (dashed lines, ⭐ = best epoch)
  Panel 3 — Learning Rate vs Epoch   (log scale)

Models that share the same loss criterion are plotted together so the
comparison is always apples-to-apples. Models without a loss_criterion in
their config.json are grouped under "Unknown".

Usage:
    python compare_models.py            # auto-discovers all model folders
    python compare_models.py m1 m3      # compare only specific folders

config.json schema (inside each model folder):
    {
      "model_name":     "m1",
      "description":    "Baseline",
      "loss_criterion": "MSELoss",      ← used for grouping
      "hyperparameters": { ... },
      "notes": "..."
    }
"""

from __future__ import annotations

import sys
import json
import pathlib
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent   # .../trained models/
LOG_FILENAME = "training_log.csv"
CFG_FILENAME = "config.json"

# ── Colour palette (extended so ≥10 models per group stay distinct) ─────────
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

# ── Matplotlib global dark style ───────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#e0e0e0",
    "axes.titlecolor":   "#ffffff",
    "axes.grid":         True,
    "grid.color":        "#2e3040",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "xtick.color":       "#a0a0b0",
    "ytick.color":       "#a0a0b0",
    "text.color":        "#e0e0e0",
    "legend.facecolor":  "#1e2130",
    "legend.edgecolor":  "#3a3d4d",
    "legend.fontsize":   8.5,
    "font.family":       "DejaVu Sans",
    "lines.linewidth":   1.8,
})


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def discover_models(requested: list[str] | None = None) -> list[pathlib.Path]:
    """Return sorted list of model dirs that contain a training_log.csv."""
    if requested:
        folders = [SCRIPT_DIR / name for name in requested]
    else:
        folders = sorted(p for p in SCRIPT_DIR.iterdir() if p.is_dir())

    valid: list[pathlib.Path] = []
    for folder in folders:
        if (folder / LOG_FILENAME).exists():
            valid.append(folder)
        else:
            print(f"  [skip] {folder.name}: no {LOG_FILENAME} found")
    return valid


def load_config(folder: pathlib.Path) -> dict:
    """Load config.json if present; return empty dict otherwise."""
    cfg_path = folder / CFG_FILENAME
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_criterion(cfg: dict) -> str:
    """Return the loss criterion name, normalised to title-case. Fallback: 'Unknown'."""
    return cfg.get("loss_criterion", "Unknown").strip()


def group_by_criterion(
    folders: list[pathlib.Path],
) -> dict[str, list[pathlib.Path]]:
    """Group model folders by their declared loss_criterion."""
    groups: dict[str, list[pathlib.Path]] = defaultdict(list)
    for folder in folders:
        cfg = load_config(folder)
        groups[get_criterion(cfg)].append(folder)
    return dict(groups)


def build_label(folder_name: str, cfg: dict) -> str:
    """
    Compact legend label: model_name — description
    followed by a line of key hyperparameter values.
    """
    name = cfg.get("model_name") or folder_name
    desc = cfg.get("description", "")
    hp   = cfg.get("hyperparameters", {})

    highlight = [
        ("learning_rate",       "lr"),
        ("lr",                  "lr"),
        ("embedding_dim",       "emb"),
        ("batch_size",          "bs"),
        ("weight_decay",        "wd"),
        ("num_workers",         "wk"),
        ("lr_patience",         "lr_pat"),
        ("early_stop_patience", "es_pat"),
    ]
    seen: set[str] = set()
    parts: list[str] = []
    for key, short in highlight:
        if key in hp and short not in seen:
            seen.add(short)
            parts.append(f"{short}={hp[key]}")
        if len(parts) == 4:
            break

    label = name
    if desc:
        label += f" — {desc}"
    if parts:
        label += f"\n  [{', '.join(parts)}]"
    return label


def format_epoch_int(x, _):
    return f"{int(x)}"


def best_val_loss(df: pd.DataFrame) -> float | None:
    return df["val_loss"].min() if "val_loss" in df.columns else None


# ════════════════════════════════════════════════════════════════════════════
# Per-group figure
# ════════════════════════════════════════════════════════════════════════════

def plot_group(criterion: str, folders: list[pathlib.Path]) -> None:
    """Produce one 3-panel figure for all models sharing the same criterion."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_train, ax_val, ax_lr = axes

    loss_label = f"{criterion} Loss"

    ax_train.set_title("Training Loss",   fontsize=13, fontweight="bold", pad=10)
    ax_val.set_title  ("Validation Loss", fontsize=13, fontweight="bold", pad=10)
    ax_lr.set_title   ("Learning Rate",   fontsize=13, fontweight="bold", pad=10)

    for ax in axes:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_epoch_int))

    ax_train.set_ylabel(loss_label, fontsize=10)
    ax_val.set_ylabel  (loss_label, fontsize=10)
    ax_lr.set_ylabel   ("Learning Rate", fontsize=10)
    ax_lr.set_yscale("log")

    summary_rows: list[str] = []

    for idx, folder in enumerate(folders):
        colour = PALETTE[idx % len(PALETTE)]
        cfg    = load_config(folder)
        label  = build_label(folder.name, cfg)

        df = pd.read_csv(folder / LOG_FILENAME)
        df.columns = df.columns.str.strip()

        x = df["epoch"] if "epoch" in df.columns else df.index + 1

        # Train loss
        train_col = (
            "train_loss" if "train_loss" in df.columns
            else "loss" if "loss" in df.columns
            else None
        )
        if train_col:
            ax_train.plot(x, df[train_col], color=colour, label=label)

        # Val loss + best-epoch star
        if "val_loss" in df.columns:
            ax_val.plot(x, df["val_loss"], color=colour, linestyle="--", label=label)
            bv = best_val_loss(df)
            if bv is not None and "epoch" in df.columns:
                best_ep = df.loc[df["val_loss"].idxmin(), "epoch"]
                ax_val.scatter(best_ep, bv, color=colour, s=100, zorder=5, marker="*")
        else:
            bv = None

        # Learning rate
        lr_col = (
            "lr" if "lr" in df.columns
            else "learning_rate" if "learning_rate" in df.columns
            else None
        )
        if lr_col:
            ax_lr.plot(x, df[lr_col], color=colour, label=label)

        # Console summary row
        notes = cfg.get("notes", "")
        row = f"  {folder.name:>6}  best_val={bv:.6f}" if bv is not None else f"  {folder.name:>6}  best_val=N/A"
        if notes:
            row += f"  | {notes}"
        summary_rows.append(row)

    # Per-axis legends (only for multi-model groups; single model still gets one)
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right",
                      framealpha=0.85, labelspacing=0.6)

    # Bottom colour key when >1 model
    if len(folders) > 1:
        proxy = [
            Line2D([0], [0], color=PALETTE[i % len(PALETTE)], linewidth=2,
                   label=build_label(f.name, load_config(f)))
            for i, f in enumerate(folders)
        ]
        fig.legend(
            handles=proxy,
            loc="lower center",
            ncol=min(len(folders), 4),
            bbox_to_anchor=(0.5, -0.22),
            fontsize=8.5,
            framealpha=0.85,
            title="Models",
            title_fontsize=9,
        )

    fig.suptitle(
        f"IDISCDualTower — {criterion} Models Comparison",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    # Console output for this group
    print(f"\n{'=' * 60}")
    print(f"  GROUP: {criterion}  ({len(folders)} model(s))")
    print(f"{'=' * 60}")
    for row in summary_rows:
        print(row)
    print(f"{'=' * 60}")

    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else None

    print(f"\nScanning: {SCRIPT_DIR}")
    folders = discover_models(requested)
    print(f"Found {len(folders)} model(s): {[f.name for f in folders]}")

    if not folders:
        print("Nothing to plot.")
        sys.exit(0)

    groups = group_by_criterion(folders)

    print(f"\nLoss-criterion groups discovered:")
    for crit, grp in sorted(groups.items()):
        print(f"  {crit:20s}: {[f.name for f in grp]}")

    for criterion, grp_folders in sorted(groups.items()):
        plot_group(criterion, grp_folders)
