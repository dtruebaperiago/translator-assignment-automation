"""
generate_translator_data.py
============================
Reads the 'Data' sheet from the Initial Dataset Excel and computes
rolling historical features per translator. Outputs a single
``Translators_Data.csv`` with one row per translator (their most-recent
state) plus a comma-separated list of all task types they have worked.

Usage:
    python DATA/Implementation/generate_translator_data.py

Output:
    DATA/Translators_Data.csv
"""

import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(BASE_DIR, "Initial Dataset.xlsx")
OUTPUT_PATH = os.path.join(BASE_DIR, "Processed", "Translators_Data.csv")

# ── STEP 1: LOAD & CLEAN ────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading Initial Dataset (Data sheet)...")
print("=" * 70)

data = pd.read_excel(EXCEL_PATH, sheet_name="Data")
print(f"  Raw shape: {data.shape}")

# Parse datetime columns
datetime_cols = ["START", "END", "ASSIGNED", "READY", "WORKING", "DELIVERED",
                 "RECEIVED", "CLOSE"]
for col in datetime_cols:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors="coerce")

# Drop rows missing critical fields
critical = ["TASK_ID", "TRANSLATOR", "START"]
before = len(data)
data.dropna(subset=critical, inplace=True)
print(f"  Dropped {before - len(data)} rows with missing critical fields.")

# Coerce and clean numeric columns
for col in ["HOURS", "HOURLY_RATE", "COST", "QUALITY_EVALUATION"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data["HOURS"] = data["HOURS"].clip(lower=0)
data["QUALITY_EVALUATION"] = data["QUALITY_EVALUATION"].clip(lower=0, upper=10)

# Impute missing numerics with median
for col in ["HOURS", "HOURLY_RATE", "QUALITY_EVALUATION"]:
    if data[col].isna().any():
        med = data[col].median()
        data[col].fillna(med, inplace=True)
        print(f"  Imputed {col} NaNs with median={med:.2f}")

# Impute missing categoricals with mode
for col in ["TASK_TYPE", "MANUFACTURER_INDUSTRY"]:
    if col in data.columns and data[col].isna().any():
        mode_val = data[col].mode()[0]
        data[col].fillna(mode_val, inplace=True)
        print(f"  Imputed {col} NaNs with mode='{mode_val}'")

# Outlier capping (1st–99th percentile)
for col in ["HOURS", "HOURLY_RATE"]:
    p01, p99 = data[col].quantile(0.01), data[col].quantile(0.99)
    data[col] = data[col].clip(lower=p01, upper=p99)
    print(f"  Capped {col} to [{p01:.2f}, {p99:.2f}]")

# Compute historical_actual_hours = (DELIVERED - WORKING) in hours
data["historical_actual_hours"] = (
    (data["DELIVERED"] - data["WORKING"]).dt.total_seconds() / 3600.0
)
data["historical_actual_hours"] = data["historical_actual_hours"].clip(lower=0)

print(f"  Cleaned shape: {data.shape}")

# ── STEP 2: ROLLING FEATURES ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Computing rolling features per translator...")
print("=" * 70)

# Sort chronologically
data.sort_values("START", inplace=True)
data.reset_index(drop=True, inplace=True)

global_median_rate = data["HOURLY_RATE"].median()
print(f"  Global median hourly rate: {global_median_rate:.2f}")
print(f"  IS_SPECIALIST threshold: > {1.5 * global_median_rate:.2f}")


def compute_rolling_features(group):
    """Compute expanding-window features for a single translator group."""
    g = group.sort_values("START").copy()

    # --- Experience Counters ---
    # Overall task count (0-indexed = count of prior tasks)
    g["rolling_task_count"] = range(len(g))

    # Domain Experience: rolling count of tasks in same MANUFACTURER_INDUSTRY
    g["domain_experience"] = g.groupby("MANUFACTURER_INDUSTRY").cumcount()

    # Task Type Experience: rolling count of tasks of same TASK_TYPE
    g["task_type_experience"] = g.groupby("TASK_TYPE").cumcount()

    # --- Rolling Recent Quality Score (EMA, span=10) ---
    shifted_quality = g["QUALITY_EVALUATION"].shift(1)
    g["rolling_quality_ema"] = shifted_quality.ewm(span=10, min_periods=1).mean()
    g["rolling_quality_ema"].fillna(g["QUALITY_EVALUATION"].median(), inplace=True)

    # --- Rolling Avg Task Time ---
    shifted_hours = g["HOURS"].shift(1)
    g["rolling_avg_task_time"] = shifted_hours.expanding(min_periods=1).mean()
    g["rolling_avg_task_time"].fillna(g["HOURS"].median(), inplace=True)

    # --- Rolling Punctuality Score ---
    shifted_actual = g["historical_actual_hours"].shift(1)
    shifted_forecast = g["HOURS"].shift(1)
    late_flag = (shifted_actual > shifted_forecast).astype(float)
    rolling_late_rate = late_flag.expanding(min_periods=1).mean()
    g["rolling_punctuality_score"] = 1.0 - rolling_late_rate
    g["rolling_punctuality_score"].fillna(0.5, inplace=True)

    # --- Rolling Efficiency Ratio ---
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = shifted_actual / shifted_forecast.replace(0, np.nan)
    g["rolling_efficiency_ratio"] = ratio.expanding(min_periods=1).mean()
    g["rolling_efficiency_ratio"].fillna(1.0, inplace=True)

    # --- IS_NEW_EMPLOYEE flag ---
    g["IS_NEW_EMPLOYEE"] = (g["rolling_task_count"] < 10).astype(int)

    # --- IS_SPECIALIST flag ---
    g["IS_SPECIALIST"] = (
        (g["HOURLY_RATE"] > 1.5 * global_median_rate)
        & (g["rolling_task_count"] < 20)
    ).astype(int)

    return g


print("  Applying per-translator rolling computations (this may take a minute)...")
data = data.groupby("TRANSLATOR", group_keys=False).apply(compute_rolling_features)
data.sort_values("START", inplace=True)
data.reset_index(drop=True, inplace=True)

print(f"  Features computed for {data['TRANSLATOR'].nunique()} translators.")

# ── STEP 3: AGGREGATE TO ONE ROW PER TRANSLATOR ─────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Aggregating to one row per translator (latest state)...")
print("=" * 70)

# Take the last row per translator (= most recent rolling state)
latest = (
    data.sort_values("START")
    .drop_duplicates(subset=["TRANSLATOR"], keep="last")
    .copy()
)

# Build task_types_worked: comma-separated list of unique task types per translator
task_types_per_translator = (
    data.groupby("TRANSLATOR")["TASK_TYPE"]
    .apply(lambda x: ",".join(sorted(x.unique())))
    .reset_index()
    .rename(columns={"TASK_TYPE": "task_types_worked"})
)

latest = latest.merge(task_types_per_translator, on="TRANSLATOR", how="left")

# Select output columns
output_cols = [
    "TRANSLATOR",
    "task_types_worked",
    "domain_experience",
    "task_type_experience",
    "rolling_quality_ema",
    "rolling_avg_task_time",
    "rolling_punctuality_score",
    "rolling_efficiency_ratio",
    "IS_NEW_EMPLOYEE",
    "IS_SPECIALIST",
    "rolling_task_count",
]

result = latest[output_cols].copy()
result = result.sort_values("TRANSLATOR").reset_index(drop=True)

# ── STEP 4: SAVE ────────────────────────────────────────────────────────────
print(f"\n  Output shape: {result.shape}")
print(f"  Saving to: {OUTPUT_PATH}")
result.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 70)
print("DONE — Translators_Data.csv generated successfully.")
print("=" * 70)
print(f"\n  Preview (first 5 rows):")
print(result.head().to_string())
