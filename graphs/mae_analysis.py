"""
MAE analysis: how much does removing rare language pairs improve model accuracy?

Runs model m1 inference on the full test set (149,690 pairs),
maps each prediction to a language-pair frequency quintile, then reports:

  "By removing X% of predictions (rare pairs) the MAE improves by Y%"

Output
------
  - Console table with per-quintile MAE and data-share
  - graphs/output/04_mae_by_quintile.png
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH_DIR    = os.path.join(BASE, "Backend", "algorithms", "Dual-Tower MLP", "Architecture")
TENSOR_DIR  = os.path.join(BASE, "Backend", "algorithms", "Dual-Tower MLP", "Data to Tensors", "tensors")
MODEL_PATH  = os.path.join(BASE, "Backend", "algorithms", "Dual-Tower MLP", "trained models", "m1", "best_idisc_model.pth")
LABELS_TEST = os.path.join(BASE, "DATA", "Processed", "test_labels.csv")
DATA_RAW    = os.path.join(BASE, "DATA", "Initial Dataset", "CSV", "Data.csv")
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ARCH_DIR)
from IDISC_DualTower import IDISCDualTower   # noqa: E402

# ── Style ──────────────────────────────────────────────────────────────────────
C = {
    "green":  "#27AE60",
    "red":    "#E74C3C",
    "blue":   "#2980B9",
    "lgrey":  "#D5D8DC",
    "dgrey":  "#85929E",
    "dark":   "#2C3E50",
    "orange": "#E67E22",
    "teal":   "#16A085",
}
QUINTILE_ORDER  = ["Very Rare", "Rare", "Medium", "Common", "Very Common"]
QUINTILE_COLORS = [C["red"], C["orange"], C["dgrey"], C["teal"], C["blue"]]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD TEST TENSORS
# ══════════════════════════════════════════════════════════════════════════════
print("Loading test tensors ...", end="", flush=True)
test_a = torch.load(os.path.join(TENSOR_DIR, "test_tower_a.pt"), map_location="cpu", weights_only=True)
test_b = torch.load(os.path.join(TENSOR_DIR, "test_tower_b.pt"), map_location="cpu", weights_only=True)
test_y = torch.load(os.path.join(TENSOR_DIR, "test_target.pt"),  map_location="cpu", weights_only=True)
test_y = test_y.squeeze().float()
N = len(test_y)
print(f"  {N:,} samples  |  Tower-A dim={test_a.shape[1]}  Tower-B dim={test_b.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD & RUN MODEL M1
# ══════════════════════════════════════════════════════════════════════════════
print("Loading model m1 ...", end="", flush=True)
model = IDISCDualTower(
    tower_a_input_dim=test_a.shape[1],
    tower_b_input_dim=test_b.shape[1],
    embedding_dim=64,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print("  done")

print("Running inference on all test pairs ...", end="", flush=True)
BATCH = 8192
preds_list = []
with torch.no_grad():
    for i in range(0, N, BATCH):
        a_b = test_a[i : i + BATCH]
        b_b = test_b[i : i + BATCH]
        p   = model(a_b, b_b).squeeze(-1)
        preds_list.append(p.numpy())
preds  = np.concatenate(preds_list).astype(np.float32)
actual = test_y.numpy().astype(np.float32)
errors = np.abs(preds - actual)

overall_mae = float(errors.mean())
print(f"  done  |  Overall MAE = {overall_mae:.4f}  (paper value: 0.0871)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. MAP EACH TEST PAIR TO A LANGUAGE-PAIR QUINTILE
# ══════════════════════════════════════════════════════════════════════════════
print("Mapping test pairs to language-pair quintiles ...", end="", flush=True)

# --- 3a. Load test_labels.csv (must be same order as tensors) ---
labels_df = pd.read_csv(LABELS_TEST)
assert len(labels_df) == N, (
    f"Row count mismatch: test_labels.csv={len(labels_df)} vs tensors={N}. "
    "Re-run data_to_tensors.py to regenerate tensors."
)
# Quick sanity: first 5 affinities should match
pt_first5  = actual[:5]
csv_first5 = labels_df["AFFINITY_LABEL"].values[:5].astype(np.float32)
if not np.allclose(pt_first5, csv_first5, atol=1e-4):
    print("\n  WARNING: tensor affinities don't match CSV — row order may differ.")

# --- 3b. Load Data.csv to get SOURCE_LANG / TARGET_LANG per TASK_ID ---
df_raw = pd.read_csv(
    DATA_RAW, sep=";",
    usecols=["TASK_ID", "SOURCE_LANG", "TARGET_LANG"],
    encoding="utf-8", low_memory=False,
)
df_raw = df_raw.dropna(subset=["SOURCE_LANG", "TARGET_LANG"])
df_raw["TASK_ID"] = pd.to_numeric(df_raw["TASK_ID"], errors="coerce")
df_raw = df_raw.dropna(subset=["TASK_ID"]).astype({"TASK_ID": int}).drop_duplicates("TASK_ID")

# --- 3c. Merge: labels row index == tensor index ---
merged = labels_df.merge(df_raw[["TASK_ID", "SOURCE_LANG", "TARGET_LANG"]], on="TASK_ID", how="left")
merged["lang_pair"] = merged["SOURCE_LANG"].fillna("?") + " -> " + merged["TARGET_LANG"].fillna("?")

# --- 3d. Build pair frequency table on ALL labels (train+test) for quintile
#         consistency with generate_graphs.py ---
all_labels = pd.concat([
    pd.read_csv(os.path.join(BASE, "DATA", "Processed", "target_labels.csv")),
    labels_df,
], ignore_index=True)
pair_freq = (
    all_labels
    .merge(df_raw[["TASK_ID", "SOURCE_LANG", "TARGET_LANG"]], on="TASK_ID", how="left")
    .dropna(subset=["SOURCE_LANG", "TARGET_LANG"])
    .groupby(["SOURCE_LANG", "TARGET_LANG"])
    .size()
    .reset_index(name="count")
)
pair_freq["lang_pair"] = pair_freq["SOURCE_LANG"] + " -> " + pair_freq["TARGET_LANG"]
pair_freq["quintile"]  = pd.qcut(pair_freq["count"], q=5, labels=QUINTILE_ORDER)

# --- 3e. Attach quintile to each test pair ---
pair_to_quintile = pair_freq.set_index("lang_pair")["quintile"].to_dict()
merged["quintile"] = merged["lang_pair"].map(pair_to_quintile)
print(f"  done  |  {merged['quintile'].notna().sum():,}/{N:,} pairs mapped to a quintile")

# Add prediction errors
merged["error"] = errors

# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTE NUMBERS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*64)
print(f"  OVERALL MAE (all {N:,} test pairs): {overall_mae:.4f}")
print("="*64)

# Per-quintile stats
rows = []
for q in QUINTILE_ORDER:
    mask = merged["quintile"] == q
    n_q  = mask.sum()
    if n_q == 0:
        continue
    mae_q = float(merged.loc[mask, "error"].mean())
    pct   = n_q / N * 100
    rows.append({"quintile": q, "n": n_q, "pct_of_total": pct, "mae": mae_q})

summary = pd.DataFrame(rows)
print(f"\n{'Quintile':<15} {'N':>8} {'% of Test':>10} {'MAE':>8}")
print("-"*45)
for _, r in summary.iterrows():
    print(f"  {r['quintile']:<13} {int(r['n']):>8,} {r['pct_of_total']:>9.1f}%  {r['mae']:>8.4f}")

# Subsets: what happens if we exclude Very Rare, then Very Rare + Rare
def mae_excluding(exclude_quintiles):
    mask = ~merged["quintile"].isin(exclude_quintiles) & merged["quintile"].notna()
    n_kept   = mask.sum()
    n_removed = (merged["quintile"].isin(exclude_quintiles)).sum()
    pct_removed = n_removed / N * 100
    mae = float(merged.loc[mask, "error"].mean())
    return n_kept, n_removed, pct_removed, mae

n1, r1, p1, mae1 = mae_excluding(["Very Rare"])
n2, r2, p2, mae2 = mae_excluding(["Very Rare", "Rare"])

print("\n" + "="*64)
print("  IMPACT OF EXCLUDING RARE PAIRS")
print("="*64)
print(f"\n  Exclude 'Very Rare' only:")
print(f"    Pairs removed : {r1:,}  ({p1:.1f}% of test set)")
print(f"    MAE           : {overall_mae:.4f}  -->  {mae1:.4f}")
print(f"    Improvement   : {(overall_mae - mae1) / overall_mae * 100:.1f}% reduction")

print(f"\n  Exclude 'Very Rare' + 'Rare':")
print(f"    Pairs removed : {r2:,}  ({p2:.1f}% of test set)")
print(f"    MAE           : {overall_mae:.4f}  -->  {mae2:.4f}")
print(f"    Improvement   : {(overall_mae - mae2) / overall_mae * 100:.1f}% reduction")

# MAE for Medium + Common + Very Common only
mask_mcu = merged["quintile"].isin(["Medium", "Common", "Very Common"])
mae_mcu  = float(merged.loc[mask_mcu, "error"].mean())
pct_mcu  = mask_mcu.sum() / N * 100
print(f"\n  Keep only Medium + Common + Very Common:")
print(f"    Coverage      : {mask_mcu.sum():,} pairs ({pct_mcu:.1f}% of test set)")
print(f"    MAE           : {overall_mae:.4f}  -->  {mae_mcu:.4f}")
print(f"    Improvement   : {(overall_mae - mae_mcu) / overall_mae * 100:.1f}% reduction")

# ══════════════════════════════════════════════════════════════════════════════
# 5. GRAPH:  04_mae_by_quintile.png
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating graph 04_mae_by_quintile.png ...", end="", flush=True)

# Pre-compute per-quintile relative MAE vs Very Common baseline
vc_mae = summary.loc[summary["quintile"] == "Very Common", "mae"].values[0]

q_labels      = summary["quintile"].tolist()
q_maes        = summary["mae"].tolist()
q_pcts        = summary["pct_of_total"].tolist()
q_ns          = summary["n"].tolist()
q_colors_plot = [dict(zip(QUINTILE_ORDER, QUINTILE_COLORS))[q] for q in q_labels]
q_worse_pct   = [(m / vc_mae - 1) * 100 for m in q_maes]   # % worse than Very Common

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.5))

# ── Left: MAE per quintile (horizontal bars) ─────────────────────────────────
bars = ax1.barh(q_labels, q_maes, color=q_colors_plot, edgecolor="white",
                linewidth=1.5, height=0.55)

ax1.axvline(overall_mae, color=C["dark"], linewidth=1.8, linestyle="--",
            label=f"Overall MAE = {overall_mae:.4f}")

for bar, mae_v, pct_v, worse in zip(bars, q_maes, q_pcts, q_worse_pct):
    # MAE label
    ax1.text(mae_v + 0.0008, bar.get_y() + bar.get_height() / 2 + 0.06,
             f"{mae_v:.4f}", va="center", ha="left",
             fontsize=10.5, fontweight="bold", color=C["dark"])
    # +X% worse tag (only for non-Very-Common quintiles)
    if worse > 0.5:
        ax1.text(mae_v + 0.0008, bar.get_y() + bar.get_height() / 2 - 0.12,
                 f"+{worse:.0f}% vs Very Common",
                 va="center", ha="left", fontsize=9, color=C["red"], style="italic")

ax1.set_xlabel("Mean Absolute Error (MAE)", fontsize=13)
ax1.set_xlim(0, max(q_maes) * 1.52)
ax1.legend(fontsize=11, frameon=False)
ax1.set_title("MAE per Language Pair\nFrequency Quintile  (model m1)",
              fontsize=14, fontweight="bold", color=C["dark"], pad=14)
ax1.grid(axis="x", linestyle=":", alpha=0.35)
ax1.tick_params(axis="y", which="both", left=False)

# ── Right: Volume vs MAE — the key contrast ──────────────────────────────────
# Primary axis: data share (log scale)
ax2_r = ax2.twinx()

x = np.arange(len(q_labels))
bar_width = 0.38

b1 = ax2.bar(x - bar_width / 2, q_pcts, width=bar_width,
             color=q_colors_plot, alpha=0.85, edgecolor="white", label="Data share (%)", zorder=3)
b2 = ax2_r.bar(x + bar_width / 2, q_maes, width=bar_width,
               color=q_colors_plot, alpha=0.40, edgecolor="white",
               hatch="///", label="MAE", zorder=3)

# Annotate data share
for bar, pct, n_q in zip(b1, q_pcts, q_ns):
    label = f"{pct:.2f}%" if pct < 1 else f"{pct:.1f}%"
    ax2.text(bar.get_x() + bar.get_width() / 2,
             max(pct + 0.5, 0.05),
             f"{label}\n(n={n_q:,})",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C["dark"])

# Annotate MAE
for bar, mae_v in zip(b2, q_maes):
    ax2_r.text(bar.get_x() + bar.get_width() / 2,
               mae_v + 0.003,
               f"{mae_v:.4f}",
               ha="center", va="bottom", fontsize=9, color=C["dark"])

ax2.set_yscale("symlog", linthresh=0.1)
ax2.set_ylabel("Share of Test Set  (%, log scale)", fontsize=12)
ax2_r.set_ylabel("MAE", fontsize=12, color=C["dark"])
ax2_r.tick_params(axis="y", colors=C["dark"])
ax2_r.spines["right"].set_edgecolor(C["dark"])
ax2_r.spines["top"].set_visible(False)

ax2.set_xticks(x)
ax2.set_xticklabels(q_labels, rotation=15, ha="right", fontsize=11)
ax2.set_title("Data Volume vs. Prediction Error\nper Frequency Quintile",
              fontsize=14, fontweight="bold", color=C["dark"], pad=14)
ax2.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)

# Combined legend
h1 = mpatches.Patch(facecolor=C["teal"], alpha=0.85, label="Data share  (left axis)")
h2 = mpatches.Patch(facecolor=C["teal"], alpha=0.40, hatch="///", label="MAE  (right axis)")
ax2.legend(handles=[h1, h2], loc="upper left", fontsize=10, frameon=True, framealpha=0.9)

# Callout box: the key number
vr_row    = summary[summary["quintile"] == "Very Rare"].iloc[0]
vr_worse  = (vr_row["mae"] / vc_mae - 1) * 100
callout   = (f"Very Rare MAE is {vr_worse:.0f}% worse\n"
             f"than Very Common, but represents\n"
             f"only {vr_row['pct_of_total']:.3f}% of the test set")
ax2.text(0.97, 0.97, callout, transform=ax2.transAxes,
         fontsize=9.5, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C["red"], alpha=0.9))

# Footer
fig.text(
    0.5, -0.03,
    f"Model: m1  |  Test set: {N:,} pairs  |  "
    f"Quintiles defined by language-pair frequency across all {len(pair_freq):,} pairs in training+test data",
    ha="center", fontsize=9.5, color=C["dgrey"], style="italic",
)

plt.tight_layout(pad=3)
out4 = os.path.join(OUT_DIR, "04_mae_by_quintile.png")
plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f" saved: {out4}")
print("\nDone.")
