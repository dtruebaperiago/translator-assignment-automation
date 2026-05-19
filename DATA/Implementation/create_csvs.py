"""
create_csvs.py
==============
Run this ONCE before running preprocessing_pipeline.py.

Reads 260319_GrauIAI_data.zip from DATA/Initial Dataset/,
extracts the xlsx inside it, and saves each of the 4 sheets as:
  - DATA/Initial Dataset/CSV/<sheet>.csv   (semicolon-separated)
  - DATA/Initial Dataset/XLSX/<sheet>.xlsx (one file per sheet)
"""

import os
import zipfile
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Implementation/
DATA_DIR   = os.path.dirname(SCRIPT_DIR)                          # .../DATA/
INIT_DIR   = os.path.join(DATA_DIR, "Initial Dataset")
CSV_DIR    = os.path.join(INIT_DIR, "CSV")
XLSX_DIR   = os.path.join(INIT_DIR, "XLSX")
ZIP_PATH   = os.path.join(INIT_DIR, "260319_GrauIAI_data.zip")
XLSX_PATH  = os.path.join(INIT_DIR, "260319_GrauIAI_data.xlsx")

os.makedirs(CSV_DIR,  exist_ok=True)
os.makedirs(XLSX_DIR, exist_ok=True)

# ── Sheet → output filename mapping ──────────────────────────────────────────
SHEETS = {
    "Data":                  "Data",
    "Schedules":             "Schedules",
    "Clients":               "Clients",
    "TranslatorsCost+Pairs": "TranslatorsCost+Pairs",
}

# ── Step 1: unzip the xlsx to disk ───────────────────────────────────────────
if not os.path.isfile(XLSX_PATH):
    print(f"Unzipping {os.path.basename(ZIP_PATH)} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # find the xlsx entry inside the zip
        entries = [e for e in zf.namelist() if e.lower().endswith(".xlsx")]
        if not entries:
            raise FileNotFoundError("No .xlsx found inside the zip.")
        zf.extract(entries[0], INIT_DIR)
        extracted = os.path.join(INIT_DIR, entries[0])
        if os.path.abspath(extracted) != os.path.abspath(XLSX_PATH):
            os.rename(extracted, XLSX_PATH)
    print(f"  Done. xlsx saved to: {XLSX_PATH}")
else:
    print(f"xlsx already on disk: {XLSX_PATH}")

# ── Step 2: read each sheet and write CSV + XLSX ──────────────────────────────
print("\nReading sheets and writing CSV / XLSX files ...")
print("(This may take several minutes for a 150 MB file)\n")

for sheet_name, base_name in SHEETS.items():
    print(f"  [{sheet_name}] reading ...", end=" ", flush=True)
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet_name, engine="openpyxl")

    csv_out  = os.path.join(CSV_DIR,  f"{base_name}.csv")
    xlsx_out = os.path.join(XLSX_DIR, f"{base_name}.xlsx")

    df.to_csv(csv_out,  sep=";", decimal=",", index=False, encoding="utf-8")
    df.to_excel(xlsx_out, index=False, engine="openpyxl")

    print(f"done  ({len(df):,} rows)  →  CSV + XLSX saved")

print("\nAll done!")
print(f"  CSVs  : {CSV_DIR}")
print(f"  XLSXs : {XLSX_DIR}")
print("\nYou can now run preprocessing_pipeline.py")
