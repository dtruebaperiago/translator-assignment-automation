"""
Data Preprocessing Pipeline for Dual-Tower MLP Recommender System
=================================================================
Ingests 4 raw CSVs, performs cleaning, feature engineering, affinity scoring,
and outputs Dual-Tower formatted data with chronological train/val/test split.
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "Initial Dataset", "CSV")
OUTPUT_DIR = os.path.join(BASE_DIR, "Processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0: LOAD RAW DATA
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 0: Loading raw datasets...")
print("=" * 70)

data = pd.read_csv(
    os.path.join(INPUT_DIR, "Data.csv"),
    sep=";", decimal=",", encoding="utf-8", low_memory=False,
)
# Drop trailing empty columns (artifacts of semicolons)
data = data.loc[:, ~data.columns.str.startswith("Unnamed")]

schedules = pd.read_csv(
    os.path.join(INPUT_DIR, "Schedules.csv"),
    sep=";", decimal=",", encoding="utf-8",
)

clients = pd.read_csv(
    os.path.join(INPUT_DIR, "Clients.csv"),
    sep=";", decimal=",", encoding="utf-8",
)
clients = clients.loc[:, ~clients.columns.str.startswith("Unnamed")]

translators_costs = pd.read_csv(
    os.path.join(INPUT_DIR, "Translators Costs+Pairs.csv"),
    sep=";", decimal=",", encoding="utf-8",
)

print(f"  data:              {data.shape}")
print(f"  schedules:         {schedules.shape}")
print(f"  clients:           {clients.shape}")
print(f"  translators_costs: {translators_costs.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA CLEANING & FEATURE SELECTION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1: Data Cleaning & Feature Selection")
print("=" * 70)

# --- 1a. Parse all datetime columns ---
datetime_cols = ["START", "END", "ASSIGNED", "READY", "WORKING", "DELIVERED",
                 "RECEIVED", "CLOSE"]
for col in datetime_cols:
    data[col] = pd.to_datetime(data[col], format="%d/%m/%Y %H:%M:%S", errors="coerce")

print(f"  Parsed {len(datetime_cols)} datetime columns.")

# --- 1b. Drop rows where critical columns are entirely missing ---
critical_cols = ["TASK_ID", "TRANSLATOR", "START"]
before = len(data)
data.dropna(subset=critical_cols, inplace=True)
print(f"  Dropped {before - len(data)} rows with missing critical fields.")

# --- 1c. Coerce numeric columns and fix logical incongruencies ---
for col in ["HOURS", "HOURLY_RATE", "COST", "QUALITY_EVALUATION"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Ensure HOURS >= 0
data["HOURS"] = data["HOURS"].clip(lower=0)
# Ensure COST >= 0
data["COST"] = data["COST"].clip(lower=0)
# Ensure QUALITY_EVALUATION is within plausible range [0, 10]
data["QUALITY_EVALUATION"] = data["QUALITY_EVALUATION"].clip(lower=0, upper=10)

# --- 1d. Impute missing values ---
# Numeric: median imputation
numeric_cols_to_impute = ["HOURS", "HOURLY_RATE", "COST", "QUALITY_EVALUATION"]
for col in numeric_cols_to_impute:
    if data[col].isna().any():
        median_val = data[col].median()
        data[col].fillna(median_val, inplace=True)
        print(f"  Imputed {col} missing values with median={median_val:.2f}")

# Categorical: mode imputation
cat_cols_to_impute = ["TASK_TYPE", "SOURCE_LANG", "TARGET_LANG", "PM",
                      "MANUFACTURER", "MANUFACTURER_INDUSTRY"]
for col in cat_cols_to_impute:
    if col in data.columns and data[col].isna().any():
        mode_val = data[col].mode()[0]
        data[col].fillna(mode_val, inplace=True)
        print(f"  Imputed {col} missing values with mode='{mode_val}'")

# --- 1e. Outlier handling (IQR / 1st-99th percentile capping) ---
outlier_cols = ["HOURS", "HOURLY_RATE", "COST"]
for col in outlier_cols:
    p01 = data[col].quantile(0.01)
    p99 = data[col].quantile(0.99)
    before_clip = data[col].describe()
    data[col] = data[col].clip(lower=p01, upper=p99)
    print(f"  Capped {col} to [{p01:.2f}, {p99:.2f}]")

# --- 1f. Drop redundant categorical hierarchies ---
drop_hierarchy = ["MANUFACTURER_SECTOR", "MANUFACTURER_INDUSTRY_GROUP",
                  "MANUFACTURER_SUBINDUSTRY"]
data.drop(columns=[c for c in drop_hierarchy if c in data.columns],
          inplace=True, errors="ignore")
print(f"  Dropped redundant hierarchy columns: {drop_hierarchy}")

# --- 1g. Handle non-numeric HOURLY_RATE in translators_costs ---
# Some entries have '#¡VALOR!' — coerce to NaN and drop
translators_costs["HOURLY_RATE"] = pd.to_numeric(
    translators_costs["HOURLY_RATE"], errors="coerce"
)
translators_costs.dropna(subset=["HOURLY_RATE"], inplace=True)

print(f"\n  Cleaned data shape: {data.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1h: SCHEDULE MODELING (from schedules data)
# ──────────────────────────────────────────────────────────────────────────────
print("\n  Computing schedule features...")

# Parse schedule START/END as time objects to compute shift length
sched_start = pd.to_datetime(schedules["START"], format="%H:%M:%S", errors="coerce")
sched_end = pd.to_datetime(schedules["END"], format="%H:%M:%S", errors="coerce")

# Daily_Shift_Length in hours — handle overnight shifts
shift_seconds = (sched_end - sched_start).dt.total_seconds()
# If negative, the shift crosses midnight: add 24h
shift_seconds = shift_seconds.where(shift_seconds > 0, shift_seconds + 24 * 3600)
schedules["Daily_Shift_Length"] = shift_seconds / 3600.0

# Count active weekdays
day_cols = ["MON", "TUES", "WED", "THURS", "FRI", "SAT", "SUN"]
schedules["Active_Days"] = schedules[day_cols].sum(axis=1)

# Weekly_Availability_Hours = Daily_Shift_Length × number of active days
schedules["Weekly_Availability_Hours"] = (
    schedules["Daily_Shift_Length"] * schedules["Active_Days"]
)

# Works_Weekends flag (1 if SAT or SUN is active)
schedules["Works_Weekends"] = (
    (schedules["SAT"] == 1) | (schedules["SUN"] == 1)
).astype(int)

# Keep only needed columns for merge
schedule_features = schedules[["NAME", "Daily_Shift_Length",
                                "Weekly_Availability_Hours", "Works_Weekends"]]
schedule_features = schedule_features.rename(columns={"NAME": "TRANSLATOR"})

print(f"  Schedule features computed for {len(schedule_features)} translators.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: SYNTHETIC FEATURE ENGINEERING (Time-Series & Rolling)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Synthetic Feature Engineering")
print("=" * 70)

# Sort chronologically by task START date for proper expanding window logic
data.sort_values("START", inplace=True)
data.reset_index(drop=True, inplace=True)

# --- 2a. Compute historical_actual_hours = (DELIVERED - WORKING) in hours ---
data["historical_actual_hours"] = (
    (data["DELIVERED"] - data["WORKING"]).dt.total_seconds() / 3600.0
)
# Clip negative values to 0 (data errors)
data["historical_actual_hours"] = data["historical_actual_hours"].clip(lower=0)

# --- 2b. Rolling features grouped by TRANSLATOR, expanding window ---
# CRITICAL: Each row's features must only use data PRIOR to its START date.
# We use shift(1) after sorting within each translator group to exclude current row.

print("  Computing rolling features (this may take a few minutes)...")

def compute_rolling_features(group):
    """Compute expanding-window features for a single translator group."""
    g = group.sort_values("START").copy()

    # --- Experience Counters ---
    # Overall task count (shifted to exclude current)
    g["rolling_task_count"] = range(len(g))  # 0-indexed = count of prior tasks

    # Domain Experience: rolling count of tasks in same MANUFACTURER_INDUSTRY
    g["domain_experience"] = (
        g.groupby("MANUFACTURER_INDUSTRY").cumcount()
    )

    # Task Type Experience: rolling count of tasks of same TASK_TYPE
    g["task_type_experience"] = (
        g.groupby("TASK_TYPE").cumcount()
    )

    # --- Rolling Recent Quality Score (EMA) ---
    # Use EMA with span=10 on shifted quality (exclude current task)
    shifted_quality = g["QUALITY_EVALUATION"].shift(1)
    g["rolling_quality_ema"] = shifted_quality.ewm(span=10, min_periods=1).mean()
    # Fill first row (no history) with global median
    g["rolling_quality_ema"].fillna(g["QUALITY_EVALUATION"].median(), inplace=True)

    # --- Rolling Avg Task Time ---
    shifted_hours = g["HOURS"].shift(1)
    g["rolling_avg_task_time"] = shifted_hours.expanding(min_periods=1).mean()
    g["rolling_avg_task_time"].fillna(g["HOURS"].median(), inplace=True)

    # --- Rolling Punctuality Score ---
    # Late if actual > forecast, else on-time
    shifted_actual = g["historical_actual_hours"].shift(1)
    shifted_forecast = g["HOURS"].shift(1)
    late_flag = (shifted_actual > shifted_forecast).astype(float)
    rolling_late_rate = late_flag.expanding(min_periods=1).mean()
    g["rolling_punctuality_score"] = 1.0 - rolling_late_rate
    g["rolling_punctuality_score"].fillna(0.5, inplace=True)  # neutral default

    # --- Rolling Efficiency Ratio ---
    # actual_hours / forecasted_hours (expanding mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = shifted_actual / shifted_forecast.replace(0, np.nan)
    g["rolling_efficiency_ratio"] = ratio.expanding(min_periods=1).mean()
    g["rolling_efficiency_ratio"].fillna(1.0, inplace=True)  # neutral default

    # --- IS_NEW_EMPLOYEE flag ---
    g["IS_NEW_EMPLOYEE"] = (g["rolling_task_count"] < 10).astype(int)

    return g

# Apply per translator
data = data.groupby("TRANSLATOR", group_keys=False).apply(compute_rolling_features)
data.sort_values("START", inplace=True)
data.reset_index(drop=True, inplace=True)

# --- 2c. IS_SPECIALIST flag ---
# Base hourly rate > 1.5× global median AND rolling task count < 20
global_median_rate = data["HOURLY_RATE"].median()
data["IS_SPECIALIST"] = (
    (data["HOURLY_RATE"] > 1.5 * global_median_rate) &
    (data["rolling_task_count"] < 20)
).astype(int)

print(f"  Global median hourly rate: {global_median_rate:.2f}")
print(f"  IS_SPECIALIST threshold: > {1.5 * global_median_rate:.2f}")
print(f"  Rolling features computed for {data['TRANSLATOR'].nunique()} translators.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: CALCULATE CONTINUOUS AFFINITY SCORE (Target Variable Y)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Affinity Score Calculation")
print("=" * 70)

# --- 3a. Join Clients data (MANUFACTURER → CLIENT_NAME) ---
clients_renamed = clients.rename(columns={"CLIENT_NAME": "MANUFACTURER"})
data = data.merge(
    clients_renamed[["MANUFACTURER", "SELLING_HOURLY_PRICE", "MIN_QUALITY", "WILDCARD"]],
    on="MANUFACTURER", how="left"
)

# Impute any missing client data
data["SELLING_HOURLY_PRICE"].fillna(data["SELLING_HOURLY_PRICE"].median(), inplace=True)
data["MIN_QUALITY"].fillna(0, inplace=True)
data["WILDCARD"].fillna(data["WILDCARD"].mode()[0], inplace=True)

# --- 3b. Quality component (0 to 1) ---
# Normalize QUALITY_EVALUATION to 0-1 scale (original is 0-10)
data["quality_norm"] = data["QUALITY_EVALUATION"] / 10.0
# Apply strict penalty if QUALITY_EVALUATION < MIN_QUALITY
data["quality_score"] = data["quality_norm"].copy()
penalty_mask = data["QUALITY_EVALUATION"] < data["MIN_QUALITY"]
data.loc[penalty_mask, "quality_score"] = (
    data.loc[penalty_mask, "quality_norm"] - 0.3
)
data["quality_score"] = data["quality_score"].clip(0.0, 1.0)

print(f"  Quality penalties applied to {penalty_mask.sum()} rows.")

# --- 3c. Time Efficiency component (0 to 1) ---
# actual_hours_worked = (DELIVERED - WORKING) in hours (already computed)
data["delay_hours"] = data["historical_actual_hours"] - data["HOURS"]
# If delay <= 0 → score 1.0; if > 0 → decay linearly
data["time_efficiency_score"] = np.where(
    data["delay_hours"] <= 0,
    1.0,
    np.maximum(0.0, 1.0 - (data["delay_hours"] / 48.0))
)

# --- 3d. Economic / Profit component (0 to 1) ---
# Profit Margin = (SELLING_HOURLY_PRICE - HOURLY_RATE) / SELLING_HOURLY_PRICE
data["profit_margin"] = (
    (data["SELLING_HOURLY_PRICE"] - data["HOURLY_RATE"]) /
    data["SELLING_HOURLY_PRICE"].replace(0, np.nan)
)
data["profit_margin"].fillna(0, inplace=True)

# Min-Max scaling to [0, 1]
pm_min = data["profit_margin"].min()
pm_max = data["profit_margin"].max()
if pm_max > pm_min:
    data["economic_score"] = (data["profit_margin"] - pm_min) / (pm_max - pm_min)
else:
    data["economic_score"] = 0.5
data["economic_score"] = data["economic_score"].clip(0.0, 1.0)

# --- 3e. Final Affinity Label ---
data["AFFINITY_LABEL"] = (
    0.40 * data["quality_score"] +
    0.30 * data["time_efficiency_score"] +
    0.30 * data["economic_score"]
)
data["AFFINITY_LABEL"] = data["AFFINITY_LABEL"].clip(0.0, 1.0)

print(f"  AFFINITY_LABEL stats:")
print(f"    mean={data['AFFINITY_LABEL'].mean():.4f}  "
      f"std={data['AFFINITY_LABEL'].std():.4f}  "
      f"min={data['AFFINITY_LABEL'].min():.4f}  "
      f"max={data['AFFINITY_LABEL'].max():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 (continued): ONE-HOT ENCODING
# ──────────────────────────────────────────────────────────────────────────────
print("\n  Applying One-Hot Encoding...")

# One-Hot Encode TASK_TYPE, SOURCE_LANG, TARGET_LANG
ohe_cols = ["TASK_TYPE", "SOURCE_LANG", "TARGET_LANG"]
data = pd.get_dummies(data, columns=ohe_cols, prefix=ohe_cols, dtype=int)

print(f"  Shape after OHE: {data.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: SPLIT FEATURES INTO DUAL-TOWER CATEGORIES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Dual-Tower Feature Split")
print("=" * 70)

# --- Drop all leakage / post-assignment columns ---
leakage_cols = [
    "START", "END", "ASSIGNED", "READY", "WORKING", "DELIVERED",
    "RECEIVED", "CLOSE", "COST", "QUALITY_EVALUATION",
    "historical_actual_hours", "delay_hours", "quality_norm",
    "quality_score", "time_efficiency_score", "profit_margin",
    "economic_score", "PM",
]
data.drop(columns=[c for c in leakage_cols if c in data.columns],
          inplace=True, errors="ignore")
print(f"  Dropped leakage columns. Shape: {data.shape}")

# --- Merge schedule features into data ---
data = data.merge(schedule_features, on="TRANSLATOR", how="left")
# Fill missing schedule features with median
for col in ["Daily_Shift_Length", "Weekly_Availability_Hours", "Works_Weekends"]:
    data[col].fillna(data[col].median(), inplace=True)

# --- Get translator base cost from Translators Costs+Pairs ---
# Use the median hourly rate per translator as their "base cost"
translator_base_cost = (
    translators_costs.groupby("TRANSLATOR")["HOURLY_RATE"]
    .median()
    .reset_index()
    .rename(columns={"HOURLY_RATE": "translator_base_cost"})
)
data = data.merge(translator_base_cost, on="TRANSLATOR", how="left")
data["translator_base_cost"].fillna(data["translator_base_cost"].median(), inplace=True)

# --- Identify column groups ---
# OHE language columns (shared between towers)
source_lang_cols = [c for c in data.columns if c.startswith("SOURCE_LANG_")]
target_lang_cols = [c for c in data.columns if c.startswith("TARGET_LANG_")]
task_type_cols = [c for c in data.columns if c.startswith("TASK_TYPE_")]
lang_cols = source_lang_cols + target_lang_cols

# --- Label-encode MANUFACTURER and MANUFACTURER_INDUSTRY for Tower B ---
from sklearn.preprocessing import LabelEncoder

le_mfr = LabelEncoder()
data["MANUFACTURER_enc"] = le_mfr.fit_transform(data["MANUFACTURER"].astype(str))

le_ind = LabelEncoder()
data["MANUFACTURER_INDUSTRY_enc"] = le_ind.fit_transform(
    data["MANUFACTURER_INDUSTRY"].astype(str)
)

# --- Label-encode WILDCARD ---
le_wc = LabelEncoder()
data["WILDCARD_enc"] = le_wc.fit_transform(data["WILDCARD"].astype(str))

# --- TOWER A: Employee Features ---
tower_a_cols = (
    ["TASK_ID", "TRANSLATOR",  # tracking keys (not for tensor)
     "translator_base_cost",
     "Daily_Shift_Length", "Weekly_Availability_Hours", "Works_Weekends",
     "rolling_quality_ema", "rolling_avg_task_time",
     "rolling_punctuality_score", "rolling_efficiency_ratio",
     "domain_experience", "task_type_experience",
     "IS_NEW_EMPLOYEE", "IS_SPECIALIST"]
    + lang_cols
)

# --- TOWER B: Task Features ---
tower_b_cols = (
    ["TASK_ID", "TRANSLATOR",  # tracking keys
     "HOURS",
     "MANUFACTURER_enc", "MANUFACTURER_INDUSTRY_enc",
     "MIN_QUALITY", "WILDCARD_enc", "SELLING_HOURLY_PRICE"]
    + task_type_cols + lang_cols
)

# --- TARGET LABELS ---
target_cols = ["TASK_ID", "TRANSLATOR", "AFFINITY_LABEL"]

tower_a = data[[c for c in tower_a_cols if c in data.columns]].copy()
tower_b = data[[c for c in tower_b_cols if c in data.columns]].copy()
target_labels = data[target_cols].copy()

print(f"  Tower A (Employee) shape: {tower_a.shape}")
print(f"  Tower B (Task) shape:     {tower_b.shape}")
print(f"  Target Labels shape:      {target_labels.shape}")

# Save tower files
tower_a.to_csv(os.path.join(OUTPUT_DIR, "tower_a_employee_features.csv"), index=False)
tower_b.to_csv(os.path.join(OUTPUT_DIR, "tower_b_task_features.csv"), index=False)
target_labels.to_csv(os.path.join(OUTPUT_DIR, "target_labels.csv"), index=False)
print("  Saved tower files.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: CHRONOLOGICAL DATA SPLIT
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Chronological Data Split (70/15/15)")
print("=" * 70)

# Data is already sorted by START date (which we used for feature engineering)
n = len(data)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Merge all features + label for train/val
feature_cols_for_merge = list(
    set(tower_a.columns) | set(tower_b.columns) | set(target_cols)
)
feature_cols_for_merge = [c for c in feature_cols_for_merge if c in data.columns]
merged = data[feature_cols_for_merge].copy()

train_df = merged.iloc[:train_end]
val_df = merged.iloc[train_end:val_end]
test_df = merged.iloc[val_end:]

# Train/Val: merged historical interactions
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_merged.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_merged.csv"), index=False)

# Test: separate Tasks and Translators for ranking evaluation (Hit Rate@K)
test_task_cols = [c for c in tower_b.columns if c in data.columns]
test_translator_cols = [c for c in tower_a.columns if c in data.columns]

test_tasks = test_df[test_task_cols].copy()
test_translators = test_df[test_translator_cols].copy()

test_tasks.to_csv(os.path.join(OUTPUT_DIR, "test_tasks.csv"), index=False)
test_translators.to_csv(os.path.join(OUTPUT_DIR, "test_translators.csv"), index=False)

# Also save test labels for evaluation
test_labels = test_df[["TASK_ID", "TRANSLATOR", "AFFINITY_LABEL"]].copy()
test_labels.to_csv(os.path.join(OUTPUT_DIR, "test_labels.csv"), index=False)

print(f"  Train: {len(train_df)} rows ({len(train_df)/n*100:.1f}%)")
print(f"  Val:   {len(val_df)} rows ({len(val_df)/n*100:.1f}%)")
print(f"  Test:  {len(test_df)} rows ({len(test_df)/n*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"\nOutput files saved to: {OUTPUT_DIR}")
for f in os.listdir(OUTPUT_DIR):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  {f:40s} {size_mb:8.2f} MB")
