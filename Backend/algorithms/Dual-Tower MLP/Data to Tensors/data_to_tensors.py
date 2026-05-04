"""
Data-to-Tensors Pipeline for the Two-Tower Recommender
=======================================================
Reads the preprocessed CSVs produced by preprocessing_pipeline.py and
creates PyTorch float32 tensors ready for the Two-Tower model.

Outputs per split:
  - tensor_tower_a   (N x F_a)  Translator / Employee features
  - tensor_tower_b   (N x F_b)  Task features
  - tensor_target    (N,)       AFFINITY_LABEL  (continuous 0-1)
"""

import os
import warnings

import pandas as pd
import torch

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "DATA", "Implementation", "Processed")

# Where to persist the .pt tensor files (optional, but handy)
TENSOR_DIR = os.path.join(BASE_DIR, "tensors")
os.makedirs(TENSOR_DIR, exist_ok=True)

# Identifiers that must NOT be fed as trainable features
ID_COLUMNS = {"TASK_ID", "TRANSLATOR"}

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Read tower CSVs (header only) to get column lists
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Extracting feature column lists from tower files")
print("=" * 70)

tower_a_all_cols = list(
    pd.read_csv(os.path.join(DATA_DIR, "tower_a_employee_features.csv"), nrows=0).columns
)
tower_b_all_cols = list(
    pd.read_csv(os.path.join(DATA_DIR, "tower_b_task_features.csv"), nrows=0).columns
)

print(f"  Tower A raw columns: {len(tower_a_all_cols)}")
print(f"  Tower B raw columns: {len(tower_b_all_cols)}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Remove identifiers from the feature lists
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Removing identifier columns")
print("=" * 70)

tower_a_feature_cols = [c for c in tower_a_all_cols if c not in ID_COLUMNS]
tower_b_feature_cols = [c for c in tower_b_all_cols if c not in ID_COLUMNS]

print(f"  Tower A feature columns: {len(tower_a_feature_cols)}")
print(f"  Tower B feature columns: {len(tower_b_feature_cols)}")

# Shared columns (language OHE vectors appear in both towers)
shared = set(tower_a_feature_cols) & set(tower_b_feature_cols)
print(f"  Shared columns (present in both towers): {len(shared)}")


# ──────────────────────────────────────────────────────────────────────────────
#  Build tensors from a single merged DataFrame
# ──────────────────────────────────────────────────────────────────────────────
def build_tensors(df: pd.DataFrame, split_name: str):
    """
    Given a merged DataFrame that contains columns from both towers plus
    AFFINITY_LABEL, extract and return three float32 tensors.

    Returns
    -------
    tensor_tower_a : torch.Tensor  (N x len(tower_a_feature_cols))
    tensor_tower_b : torch.Tensor  (N x len(tower_b_feature_cols))
    tensor_target  : torch.Tensor  (N,)
    """
    # --- Align columns: fill any missing tower column with 0 ---------------
    for col in tower_a_feature_cols + tower_b_feature_cols:
        if col not in df.columns:
            print(f"    WARNING: Column '{col}' missing in {split_name} -- filling with 0")
            df[col] = 0

    # Defragment after potentially adding many columns one-by-one
    df = df.copy()

    # --- Extract sub-DataFrames --------------------------------------------
    df_a = df[tower_a_feature_cols].copy()
    df_b = df[tower_b_feature_cols].copy()

    # --- Extract target (only for splits that carry AFFINITY_LABEL) --------
    if "AFFINITY_LABEL" in df.columns:
        target_series = df["AFFINITY_LABEL"].copy()
    else:
        target_series = None

    # --- Safety: replace any remaining NaN with 0 --------------------------
    df_a = df_a.fillna(0)
    df_b = df_b.fillna(0)

    # --- Convert to float32 tensors ----------------------------------------
    tensor_tower_a = torch.tensor(df_a.values, dtype=torch.float32)
    tensor_tower_b = torch.tensor(df_b.values, dtype=torch.float32)

    if target_series is not None:
        tensor_target = torch.tensor(target_series.values, dtype=torch.float32)
    else:
        tensor_target = None

    # --- Report -------------------------------------------------------------
    print(f"  {split_name} tensor_tower_a : {tensor_tower_a.shape}")
    print(f"  {split_name} tensor_tower_b : {tensor_tower_b.shape}")
    if tensor_target is not None:
        print(f"  {split_name} tensor_target  : {tensor_target.shape}")

    return tensor_tower_a, tensor_tower_b, tensor_target


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3-6 (TRAIN): Read train_merged.csv -> extract -> tensorise
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3-6: Building TRAIN tensors")
print("=" * 70)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_merged.csv"), low_memory=False)
print(f"  Loaded train_merged.csv: {train_df.shape}")

train_tower_a, train_tower_b, train_target = build_tensors(train_df, "train")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3-6 (VAL): Read val_merged.csv -> extract -> tensorise
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3-6: Building VAL tensors")
print("=" * 70)

val_df = pd.read_csv(os.path.join(DATA_DIR, "val_merged.csv"), low_memory=False)
print(f"  Loaded val_merged.csv: {val_df.shape}")

val_tower_a, val_tower_b, val_target = build_tensors(val_df, "val")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3-6 (TEST): Reconstruct merged test from tasks + translators + labels
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3-6: Building TEST tensors")
print("=" * 70)

test_tasks = pd.read_csv(os.path.join(DATA_DIR, "test_tasks.csv"), low_memory=False)
test_translators = pd.read_csv(
    os.path.join(DATA_DIR, "test_translators.csv"), low_memory=False
)
test_labels = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv"))

# Combine column-wise (rows are already aligned from the preprocessing pipeline).
# test_translators and test_tasks share ID cols and language OHE cols -- keep only
# one copy of each shared column to avoid duplicated-column tensor bloat.
cols_from_tasks = [
    c for c in test_tasks.columns
    if c not in test_translators.columns  # skip IDs + shared lang cols already in translators
]
test_df = pd.concat(
    [test_translators, test_tasks[cols_from_tasks]],
    axis=1,
)
test_df = test_df.copy()  # defragment before adding a new column
with warnings.catch_warnings():
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    test_df["AFFINITY_LABEL"] = test_labels["AFFINITY_LABEL"].values

print(f"  Loaded test set: {test_df.shape}")

test_tower_a, test_tower_b, test_target = build_tensors(test_df, "test")

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: Persist tensors to disk
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Saving tensors to disk")
print("=" * 70)

torch.save(train_tower_a, os.path.join(TENSOR_DIR, "train_tower_a.pt"))
torch.save(train_tower_b, os.path.join(TENSOR_DIR, "train_tower_b.pt"))
torch.save(train_target,  os.path.join(TENSOR_DIR, "train_target.pt"))

torch.save(val_tower_a,   os.path.join(TENSOR_DIR, "val_tower_a.pt"))
torch.save(val_tower_b,   os.path.join(TENSOR_DIR, "val_tower_b.pt"))
torch.save(val_target,    os.path.join(TENSOR_DIR, "val_target.pt"))

torch.save(test_tower_a,  os.path.join(TENSOR_DIR, "test_tower_a.pt"))
torch.save(test_tower_b,  os.path.join(TENSOR_DIR, "test_tower_b.pt"))
torch.save(test_target,   os.path.join(TENSOR_DIR, "test_target.pt"))

print(f"\n  All tensors saved to: {TENSOR_DIR}")
for f in sorted(os.listdir(TENSOR_DIR)):
    fpath = os.path.join(TENSOR_DIR, f)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"    {f:30s} {size_mb:8.2f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TENSOR CREATION COMPLETE")
print("=" * 70)
print(f"""
  Tower A features : {len(tower_a_feature_cols)} columns  (Translator / Employee)
  Tower B features : {len(tower_b_feature_cols)} columns  (Task)
  Shared columns   : {len(shared)} columns  (language OHE vectors in both)

  TRAIN  -> tower_a {train_tower_a.shape}, tower_b {train_tower_b.shape}, target {train_target.shape}
  VAL    -> tower_a {val_tower_a.shape}, tower_b {val_tower_b.shape}, target {val_target.shape}
  TEST   -> tower_a {test_tower_a.shape}, tower_b {test_tower_b.shape}, target {test_target.shape}
""")
