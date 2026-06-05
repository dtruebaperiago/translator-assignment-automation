"""
Render the per-quintile MAE results as a clean PNG table.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

C = {
    "green":  "#27AE60",
    "red":    "#E74C3C",
    "blue":   "#2980B9",
    "lgrey":  "#ECF0F1",
    "dgrey":  "#85929E",
    "dark":   "#2C3E50",
    "orange": "#E67E22",
    "teal":   "#16A085",
    "white":  "#FFFFFF",
}

QUINTILE_COLORS = {
    "Very Rare":    "#E74C3C",
    "Rare":         "#E67E22",
    "Medium":       "#85929E",
    "Common":       "#16A085",
    "Very Common":  "#2980B9",
}

# ── Table data ────────────────────────────────────────────────────────────────
headers = ["Quintile", "Test Pairs (n)", "% of Test Set", "MAE", "vs. Very Common"]
rows = [
    ("Very Rare",    "12",        "0.008%",  "0.1472", "+69%"),
    ("Rare",         "29",        "0.019%",  "0.1140", "+31%"),
    ("Medium",       "86",        "0.057%",  "0.1200", "+38%"),
    ("Common",       "420",       "0.280%",  "0.1236", "+42%"),
    ("Very Common",  "149,143",   "99.636%", "0.0870", "—  (baseline)"),
]
footer = "Model: m1  |  Overall MAE = 0.0871  |  Test set: 149,690 pairs"

# ── Layout ────────────────────────────────────────────────────────────────────
n_rows = len(rows)
n_cols = len(headers)
fig_w, fig_h = 13, 4.2
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")
fig.patch.set_facecolor(C["white"])

col_widths = [2.6, 2.1, 2.1, 1.6, 2.6]   # must sum to fig_w
col_x = [sum(col_widths[:i]) for i in range(n_cols)]

row_h     = 0.56
header_h  = 0.62
y_header  = fig_h - 0.55
y_rows    = [y_header - header_h - i * row_h for i in range(n_rows)]
text_pad  = 0.14

# ── Header row ────────────────────────────────────────────────────────────────
for j, (hdr, cx, cw) in enumerate(zip(headers, col_x, col_widths)):
    ax.add_patch(FancyBboxPatch(
        (cx + 0.04, y_header - header_h + 0.04),
        cw - 0.08, header_h - 0.04,
        boxstyle="round,pad=0.0", linewidth=0,
        facecolor=C["dark"], zorder=2,
    ))
    ax.text(cx + cw / 2, y_header - header_h / 2,
            hdr, ha="center", va="center",
            fontsize=12, fontweight="bold", color=C["white"], zorder=3)

# ── Data rows ─────────────────────────────────────────────────────────────────
for i, (row, y_r) in enumerate(zip(rows, y_rows)):
    q_name   = row[0]
    q_color  = QUINTILE_COLORS[q_name]
    row_bg   = C["lgrey"] if i % 2 == 0 else C["white"]

    for j, (cell, cx, cw) in enumerate(zip(row, col_x, col_widths)):
        # Row background
        ax.add_patch(FancyBboxPatch(
            (cx + 0.04, y_r - row_h + 0.04),
            cw - 0.08, row_h - 0.04,
            boxstyle="round,pad=0.0", linewidth=0,
            facecolor=row_bg, zorder=1,
        ))

        # Quintile name cell: colored left accent + badge
        if j == 0:
            ax.add_patch(plt.Rectangle(
                (cx + 0.04, y_r - row_h + 0.04), 0.12, row_h - 0.04,
                color=q_color, zorder=2,
            ))
            ax.text(cx + cw / 2 + 0.06, y_r - row_h / 2,
                    cell, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=C["dark"], zorder=3)
        # "vs. Very Common" cell: colour-code the delta
        elif j == 4:
            pct_color = C["red"] if cell.startswith("+") else C["teal"]
            ax.text(cx + cw / 2, y_r - row_h / 2,
                    cell, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=pct_color, zorder=3)
        # MAE cell: bold
        elif j == 3:
            ax.text(cx + cw / 2, y_r - row_h / 2,
                    cell, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=C["dark"], zorder=3)
        else:
            ax.text(cx + cw / 2, y_r - row_h / 2,
                    cell, ha="center", va="center",
                    fontsize=11, color=C["dark"], zorder=3)

# ── Footer ────────────────────────────────────────────────────────────────────
y_footer = y_rows[-1] - row_h
ax.text(fig_w / 2, y_footer + 0.18,
        footer, ha="center", va="center",
        fontsize=9.5, color=C["dgrey"], style="italic")

# ── Outer border ──────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (0.02, y_footer + 0.02), fig_w - 0.04, fig_h - y_footer - 0.04,
    boxstyle="round,pad=0.0", linewidth=1.5,
    edgecolor=C["dgrey"], facecolor="none", zorder=4,
))

plt.tight_layout(pad=0)
out = os.path.join(OUT_DIR, "05_mae_table.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C["white"])
plt.close()
print(f"saved: {out}")
