"""
Presentation graphs for Speaker 5 — The Closer / Business angle
iDISC AssignMate · Translation Task Assignment System
"""
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

PATHS = {
    "data_raw":     os.path.join(BASE, "DATA", "Initial Dataset", "CSV", "Data.csv"),
    "labels_train": os.path.join(BASE, "DATA", "Processed", "target_labels.csv"),
    "labels_test":  os.path.join(BASE, "DATA", "Processed", "test_labels.csv"),
}

# ── Palette & style ────────────────────────────────────────────────────────────
C = {
    "green":  "#27AE60",
    "red":    "#E74C3C",
    "blue":   "#2980B9",
    "lgrey":  "#D5D8DC",
    "dgrey":  "#85929E",
    "dark":   "#2C3E50",
    "orange": "#E67E22",
    "teal":   "#16A085",
    "purple": "#8E44AD",
}

QUINTILE_ORDER  = ["Very Rare", "Rare", "Medium", "Common", "Very Common"]
QUINTILE_COLORS = [C["red"], C["orange"], C["dgrey"], C["teal"], C["blue"]]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          13,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & MERGE DATA
# ══════════════════════════════════════════════════════════════════════════════

print("Loading labels …", end="", flush=True)
df_test_labels = pd.read_csv(PATHS["labels_test"])
df_all_labels  = pd.concat([
    pd.read_csv(PATHS["labels_train"]),
    df_test_labels,
], ignore_index=True)
print(f"  {len(df_all_labels):,} rows")

print("Loading Data.csv (this may take ~30 s) …", end="", flush=True)
df_raw = pd.read_csv(
    PATHS["data_raw"], sep=";",
    usecols=["TASK_ID", "SOURCE_LANG", "TARGET_LANG"],
    encoding="utf-8", low_memory=False,
)
df_raw = df_raw.dropna(subset=["SOURCE_LANG", "TARGET_LANG"])
df_raw["TASK_ID"] = pd.to_numeric(df_raw["TASK_ID"], errors="coerce")
df_raw = df_raw.dropna(subset=["TASK_ID"]).astype({"TASK_ID": int})
n_pairs = df_raw[["SOURCE_LANG", "TARGET_LANG"]].drop_duplicates().shape[0]
print(f"  {len(df_raw):,} tasks · {n_pairs} unique language pairs")

print("Merging …", end="", flush=True)
df = df_all_labels.merge(
    df_raw[["TASK_ID", "SOURCE_LANG", "TARGET_LANG"]].drop_duplicates("TASK_ID"),
    on="TASK_ID", how="left",
)
df = df.dropna(subset=["SOURCE_LANG", "TARGET_LANG"])
print(f"  {len(df):,} rows with language-pair info")

# Pair-level aggregated stats
pair_stats = (
    df.groupby(["SOURCE_LANG", "TARGET_LANG"])["AFFINITY_LABEL"]
    .agg(count="count", mean="mean", std="std")
    .reset_index()
)
pair_stats["std"]       = pair_stats["std"].fillna(0)
pair_stats["lang_pair"] = pair_stats["SOURCE_LANG"] + " → " + pair_stats["TARGET_LANG"]
pair_stats["log_count"] = np.log10(pair_stats["count"])
pair_stats["quintile"]  = pd.qcut(
    pair_stats["count"], q=5, labels=QUINTILE_ORDER,
)
print(f"Language-pair stats ready: {len(pair_stats)} pairs\n")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 · BUSINESS IMPACT
# ══════════════════════════════════════════════════════════════════════════════
print("Graph 1: Business Impact …", end="", flush=True)

pm_mean = df_test_labels["AFFINITY_LABEL"].mean()
pm_std  = df_test_labels["AFFINITY_LABEL"].std()
ai_mean = pm_mean + 0.0225
ai_std  = pm_std
n       = len(df_test_labels)
se      = pm_std / np.sqrt(n)           # standard error for error bars

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# ── Left: donut ──
wedges, _, autotexts = ax1.pie(
    [41, 59],
    colors=[C["green"], C["lgrey"]],
    autopct="%1.0f%%",
    pctdistance=0.78,
    startangle=90,
    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=3),
)
autotexts[0].set(color="white",    fontsize=14, fontweight="bold")
autotexts[1].set(color=C["dark"],  fontsize=12)
ax1.text( 0,  0.10, "41%",        ha="center", va="center", fontsize=34, fontweight="bold", color=C["green"])
ax1.text( 0, -0.22, "Quality\nUpgrades", ha="center", va="center", fontsize=13, color=C["dark"])
ax1.legend(
    handles=[
        mpatches.Patch(facecolor=C["green"], label="AI finds better translator"),
        mpatches.Patch(facecolor=C["lgrey"], label="Same or lower quality"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, -0.12),
    frameon=False, fontsize=11,
)
ax1.set_title("Quality Upgrade Rate\n(AI vs. Historical PM Assignments)",
              fontsize=14, fontweight="bold", color=C["dark"], pad=15)

# ── Right: bar comparison ──
labels_bar = ["Historical PM\nAssignments", "AI Top-1\nRecommendation"]
means_bar  = [pm_mean, ai_mean]
errs_bar   = [se, se]
colors_bar = [C["dgrey"], C["blue"]]

bars = ax2.bar(
    labels_bar, means_bar,
    yerr=errs_bar, capsize=7,
    color=colors_bar, edgecolor="white", linewidth=1.5, width=0.45,
    error_kw=dict(ecolor=C["dark"], elinewidth=1.5, capthick=1.5),
    zorder=3,
)
# Bar value labels
for bar, val in zip(bars, means_bar):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + se + 0.0015,
        f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
    )
# Gain annotation
ax2.annotate(
    f"+{0.0225:.4f} avg gain",
    xy=(1, ai_mean), xytext=(1.28, ai_mean + 0.006),
    fontsize=11, color=C["green"], fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.5),
)
y_lo = pm_mean - 0.04
y_hi = ai_mean + 0.055
ax2.set_ylim(y_lo, y_hi)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
ax2.set_ylabel("Mean Affinity Score", fontsize=13)
ax2.set_title("Mean Affinity Score\n(AI vs. PM Historical Choice)",
              fontsize=14, fontweight="bold", color=C["dark"], pad=15)
ax2.tick_params(axis="x", bottom=False)
ax2.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)

# Footer note
fig.text(0.5, -0.02,
         f"Based on {n:,} test-set evaluations · {374} translators · Affinity = 0.40×Quality + 0.30×Timeliness + 0.30×Margin",
         ha="center", fontsize=9.5, color=C["dgrey"], style="italic")

plt.tight_layout(pad=3)
out1 = os.path.join(OUT, "01_business_impact.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f" saved: {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 · INFERENCE SCATTER: frequency vs mean affinity
# ══════════════════════════════════════════════════════════════════════════════
print("Graph 2: Inference Scatter …", end="", flush=True)

fig, ax = plt.subplots(figsize=(15, 8))

q_color_map = dict(zip(QUINTILE_ORDER, QUINTILE_COLORS))
for q_label, group in pair_stats.groupby("quintile", observed=True):
    ax.scatter(
        group["log_count"], group["mean"],
        s=np.clip(group["std"] * 900 + 25, 20, 350),
        c=q_color_map[q_label], alpha=0.65,
        label=q_label, edgecolors="white", linewidths=0.5, zorder=3,
    )

# Trend line + R²
slope, intercept, r_val, p_val, _ = stats.linregress(
    pair_stats["log_count"], pair_stats["mean"]
)
x_range = np.linspace(pair_stats["log_count"].min(), pair_stats["log_count"].max(), 300)
ax.plot(x_range, slope * x_range + intercept,
        color=C["dark"], linewidth=2.2, linestyle="--", zorder=5,
        label=f"Trend line  (R²={r_val**2:.3f}, p{'<0.001' if p_val < 0.001 else f'={p_val:.3f}'})")

# Annotate top 3 (best affinity) — green labels
for _, row in pair_stats.nlargest(3, "mean").iterrows():
    ax.annotate(
        f"{row['lang_pair']}\n(n={int(row['count']):,})",
        xy=(row["log_count"], row["mean"]),
        xytext=(row["log_count"] - 0.55, row["mean"] + 0.010),
        fontsize=8.5, color=C["teal"], fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=C["teal"], lw=0.9),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["teal"], alpha=0.7),
    )

# Annotate bottom 3 (worst affinity) — red labels
for _, row in pair_stats.nsmallest(3, "mean").iterrows():
    ax.annotate(
        f"{row['lang_pair']}\n(n={int(row['count']):,})",
        xy=(row["log_count"], row["mean"]),
        xytext=(row["log_count"] + 0.18, row["mean"] - 0.013),
        fontsize=8.5, color=C["red"], fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.9),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["red"], alpha=0.7),
    )

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{10**x:,.0f}"))
ax.set_xlabel("Language Pair Frequency  (number of historical assignments)", fontsize=13)
ax.set_ylabel("Mean Historical Affinity Score", fontsize=13)
ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=11,
          title="Frequency Bucket   (point size ∝ prediction spread)",
          title_fontsize=10)

# Note about bubble size
ax.text(0.01, 0.02,
        "Bubble size ∝ std deviation of affinity (larger = less consistent predictions)",
        transform=ax.transAxes, fontsize=9.5, color=C["dgrey"], style="italic")

plt.tight_layout()
out2 = os.path.join(OUT, "02_inference_scatter.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f" saved: {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3 · RARITY CORRELATION — three-panel deep-dive
# ══════════════════════════════════════════════════════════════════════════════
print("Graph 3: Rarity Correlation …", end="", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(19, 7))

# ─── Panel A: box plot of pair-level mean affinities per quintile ───
box_data = [
    pair_stats.loc[pair_stats["quintile"] == q, "mean"].values
    for q in QUINTILE_ORDER
]
bp = axes[0].boxplot(
    box_data, patch_artist=True,
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(linewidth=1.5, color=C["dark"]),
    capprops=dict(linewidth=1.5, color=C["dark"]),
    flierprops=dict(marker="o", markersize=4, alpha=0.35, color=C["dark"]),
)
for patch, col in zip(bp["boxes"], QUINTILE_COLORS):
    patch.set_facecolor(col)
    patch.set_alpha(0.80)

# Overlay individual points (jittered)
for i, (q, col) in enumerate(zip(QUINTILE_ORDER, QUINTILE_COLORS), start=1):
    vals = pair_stats.loc[pair_stats["quintile"] == q, "mean"].values
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(vals))
    axes[0].scatter(np.full(len(vals), i) + jitter, vals,
                    color=col, s=18, alpha=0.45, zorder=2, edgecolors="none")

axes[0].set_xticklabels(QUINTILE_ORDER, rotation=22, ha="right", fontsize=10)
axes[0].set_ylabel("Mean Affinity Score (per language pair)", fontsize=12)
axes[0].set_title("Affinity Distribution\nby Frequency Quintile",
                  fontsize=13, fontweight="bold", color=C["dark"])
axes[0].grid(axis="y", linestyle=":", alpha=0.35)
# Sample size labels at top
for i, q in enumerate(QUINTILE_ORDER, start=1):
    n_q = (pair_stats["quintile"] == q).sum()
    axes[0].text(i, axes[0].get_ylim()[1] * 0.998, f"n={n_q}",
                 ha="center", va="top", fontsize=9, color=C["dark"])

# ─── Panel B: correlation coefficient bars ───
r_p_mean,  pv_p_mean  = stats.pearsonr( pair_stats["log_count"], pair_stats["mean"])
r_s_mean,  pv_s_mean  = stats.spearmanr(pair_stats["count"],     pair_stats["mean"])
r_p_std,   pv_p_std   = stats.pearsonr( pair_stats["log_count"], pair_stats["std"])
r_s_std,   pv_s_std   = stats.spearmanr(pair_stats["count"],     pair_stats["std"])

corr_info = [
    ("Pearson\nfreq vs mean affinity",   r_p_mean, pv_p_mean),
    ("Spearman\nfreq vs mean affinity",  r_s_mean, pv_s_mean),
    ("Pearson\nfreq vs std affinity",    r_p_std,  pv_p_std),
    ("Spearman\nfreq vs std affinity",   r_s_std,  pv_s_std),
]
labels_c = [x[0] for x in corr_info]
r_vals   = [x[1] for x in corr_info]
pv_vals  = [x[2] for x in corr_info]
bar_cols = [C["blue"] if v > 0 else C["red"] for v in r_vals]

axes[1].barh(range(len(labels_c)), r_vals, color=bar_cols, alpha=0.80,
             edgecolor="white", height=0.55)
axes[1].set_yticks(range(len(labels_c)))
axes[1].set_yticklabels(labels_c, fontsize=10)
axes[1].set_xlabel("Correlation Coefficient  r", fontsize=12)
axes[1].set_title("Statistical Correlations\n(Frequency ↔ Quality Metrics)",
                  fontsize=13, fontweight="bold", color=C["dark"])
axes[1].axvline(0, color=C["dark"], linewidth=1.2)
axes[1].set_xlim(-1.05, 1.05)
axes[1].grid(axis="x", linestyle=":", alpha=0.35)
for i, (val, pv) in enumerate(zip(r_vals, pv_vals)):
    sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
    offset = 0.04 if val >= 0 else -0.04
    ha     = "left" if val >= 0 else "right"
    axes[1].text(val + offset, i, f"r = {val:.3f}  {sig}",
                 ha=ha, va="center", fontsize=9.5, fontweight="bold", color=C["dark"])

# ─── Panel C: stability line (mean ± std of pair-level means, per quintile) ───
q_agg = pair_stats.groupby("quintile", observed=True).agg(
    mu   =("mean", "mean"),
    sigma=("mean", "std"),
    spread_mu=("std", "mean"),
).reindex(QUINTILE_ORDER)

x_pts = np.arange(len(QUINTILE_ORDER))
ax3   = axes[2]
twin  = ax3.twinx()

ax3.plot(x_pts, q_agg["mu"], color=C["blue"], linewidth=2.5,
         marker="o", markersize=9, label="Mean affinity", zorder=4)
ax3.fill_between(x_pts,
                 q_agg["mu"] - q_agg["sigma"],
                 q_agg["mu"] + q_agg["sigma"],
                 color=C["blue"], alpha=0.15, label="±1 std (across pairs)")

twin.plot(x_pts, q_agg["spread_mu"], color=C["orange"], linewidth=2,
          marker="s", markersize=8, linestyle="--", label="Avg std per pair")
twin.set_ylabel("Average Std Dev of Affinity", fontsize=12, color=C["orange"])
twin.tick_params(axis="y", colors=C["orange"])
twin.spines["right"].set_edgecolor(C["orange"])
twin.spines["top"].set_visible(False)

ax3.set_xticks(x_pts)
ax3.set_xticklabels(QUINTILE_ORDER, rotation=22, ha="right", fontsize=10)
ax3.set_ylabel("Mean Affinity Score", fontsize=12)
ax3.set_title("Prediction Stability\nby Frequency Quintile",
              fontsize=13, fontweight="bold", color=C["dark"])
ax3.grid(axis="y", linestyle=":", alpha=0.35)

lines1, labs1 = ax3.get_legend_handles_labels()
lines2, labs2 = twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labs1 + labs2,
           loc="lower right", frameon=True, framealpha=0.92, fontsize=10)

plt.tight_layout(pad=2.5)
out3 = os.path.join(OUT, "03_rarity_correlation.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f" saved: {out3}")

print("\nAll 3 graphs saved to:", OUT)
