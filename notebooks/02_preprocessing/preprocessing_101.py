
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_INPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "Data_enriched.csv"
)

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed"
)

RANDOM_SEED = 42

# Split ratios: 70% train, 15% validation, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Columns to drop -- administrative IDs, Kanban timestamps, and redundant
COLUMNS_TO_DROP: List[str] = [
    "PROJECT_ID",           
    "TASK_ID",            
    "PM",           
    "START",                
    "END",                  
    "ASSIGNED",             
    "READY",                
    "WORKING",              
    "DELIVERED",            
    "RECEIVED",             
    "CLOSE",                
    "_LANG_PAIR",           
]

# Language encoding convention:  English = 1, Spanish = 0

LANGUAGE_ENCODING = {
    "English": 1,
    "English (UK)": 1,
    "English (US)": 1,
    "Spanish (Iberian)": 0,
    "Spanish (LA)": 0,
    "Spanish (Global)": 0,
    "Spanish (Mexico)": 0,
    "Spanish (US)": 0,
    "Spanish (Argentina)": 0,
    "Spanish (Chile)": 0,
    "Spanish (SOURCE)": 0,
}


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are not useful for the assignment model.

    Only drops columns that actually exist in the dataframe so the
    function is safe to call even if the schema has changed.

    Parameters
    ----------
    df : pd.DataFrame
        The raw (or enriched) dataset.

    Returns
    -------
    pd.DataFrame
        A copy of *df* without the irrelevant columns.
    """
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    cols_missing = [c for c in COLUMNS_TO_DROP if c not in df.columns]

    df_clean = df.drop(columns=cols_present)

    print(f"[OK] Step 1 -- Dropped {len(cols_present)} columns: {cols_present}")
    if cols_missing:
        print(f"   [WARN] Columns not found (skipped): {cols_missing}")
    print(f"   Remaining columns: {len(df_clean.columns)}")

    return df_clean


#  One-Hot / Label Encoding


def _build_language_map(series: pd.Series) -> dict:
    """Build a complete language -> integer mapping.

    English variants -> 1, Spanish variants -> 0 (as specified).
    All other languages get a unique integer starting from 2.
    """
    lang_map = dict(LANGUAGE_ENCODING)  # start with the predefined ones
    next_code = 2

    for lang in sorted(series.dropna().unique()):
        if lang not in lang_map:
            lang_map[lang] = next_code
            next_code += 1

    return lang_map


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply encoding to categorical columns.

    - **SOURCE_LANG / TARGET_LANG**: Label-encoded with English = 1,
      Spanish = 0, other languages = 2, 3, 4 ...
    - **TASK_TYPE**: One-hot encoded (creates one binary column per task type).
    - **MANUFACTURER_SECTOR / MANUFACTURER_INDUSTRY_GROUP /
      MANUFACTURER_INDUSTRY / MANUFACTURER_SUBINDUSTRY / MANUFACTURER /
      TRANSLATOR**: Label-encoded (integer codes).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after irrelevant columns have been dropped.

    Returns
    -------
    pd.DataFrame
        Dataset with all categorical columns replaced by numeric encodings.
    """
    df = df.copy()

    for col in ["SOURCE_LANG", "TARGET_LANG"]:
        if col in df.columns:
            lang_map = _build_language_map(df[col])
            df[col] = df[col].map(lang_map)
            unmapped = df[col].isna().sum()
            if unmapped > 0:
                print(f"   [WARN] {unmapped} unmapped values in {col}")
            print(f"   {col}: encoded {len(lang_map)} languages "
                  f"(English=1, Spanish=0, others>=2)")

    # --- TASK_TYPE -> one-hot -----------------------------------------------
    if "TASK_TYPE" in df.columns:
        dummies = pd.get_dummies(df["TASK_TYPE"], prefix="TASK", dtype=int)
        df = pd.concat([df.drop(columns=["TASK_TYPE"]), dummies], axis=1)
        print(f"   TASK_TYPE: one-hot encoded -> {list(dummies.columns)}")

    label_encode_cols = [
        "TRANSLATOR",
        "MANUFACTURER",
        "MANUFACTURER_SECTOR",
        "MANUFACTURER_INDUSTRY_GROUP",
        "MANUFACTURER_INDUSTRY",
        "MANUFACTURER_SUBINDUSTRY",
    ]

    for col in label_encode_cols:
        if col in df.columns:
            codes, uniques = pd.factorize(df[col], sort=True)
            df[col] = codes
            print(f"   {col}: label-encoded -> {len(uniques)} unique values")

    print(f"[OK] Step 2 -- Encoding complete. Shape: {df.shape}")
    return df


# Handle missing values
def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Delete every row that contains at least one missing value.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after encoding.

    Returns
    -------
    pd.DataFrame
        Dataset with no missing values.
    """
    rows_before = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df_clean)

    print(f"[OK] Step 3 -- Dropped {rows_dropped:,} rows with missing values "
          f"({rows_before:,} -> {len(df_clean):,})")

    return df_clean


#  Train / Validation / Test split

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training, validation, and test subsets.

    Uses a two-stage stratified-random split:
        1. Train vs. (Validation + Test)
        2. Validation vs. Test

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed dataset (no missing values, all numeric).
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Split ratios must sum to 1.0"

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        shuffle=True,
    )

    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=random_seed,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[OK] Step 4 -- Split complete (seed={random_seed}):")
    print(f"   Train      : {len(train_df):>8,} rows  ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation : {len(val_df):>8,} rows  ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test       : {len(test_df):>8,} rows  ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def run_pipeline(
    input_path: str | os.PathLike = DEFAULT_INPUT_PATH,
    output_dir: str | os.PathLike = DEFAULT_OUTPUT_DIR,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full preprocessing pipeline.

    1. Load CSV
    2. Drop irrelevant columns
    3. Encode categorical columns
    4. Drop rows with missing values
    5. Split into train / val / test
    6. (Optionally) save to CSV

    Parameters
    ----------
    input_path : str or path-like
        Path to the input CSV file.
    output_dir : str or path-like
        Directory where the processed CSVs will be saved.
    save : bool
        Whether to persist the splits as CSV files.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()

    print("=" * 60)
    print("  iDISC -- Dataset Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_dir}")
    print("=" * 60)

    print("\n>> Loading dataset ...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"   Loaded {len(df):,} rows x {len(df.columns)} columns\n")

    df = drop_irrelevant_columns(df)
    print()

    df = encode_categorical_columns(df)
    print()

    df = drop_missing_rows(df)
    print()

    train_df, val_df, test_df = split_dataset(df)
    print()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train.csv"
        val_path   = output_dir / "validation.csv"
        test_path  = output_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f">> Saved splits to {output_dir}:")
        print(f"   - {train_path.name}       ({len(train_df):,} rows)")
        print(f"   - {val_path.name}  ({len(val_df):,} rows)")
        print(f"   - {test_path.name}        ({len(test_df):,} rows)")

    print("\n[DONE] Preprocessing pipeline complete!")
    return train_df, val_df, test_df



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="iDISC dataset preprocessing pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to the input CSV file (default: data/Data_enriched.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the processed splits (default: data/processed/)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the pipeline without saving files (dry run)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        save=not args.no_save,
    )
