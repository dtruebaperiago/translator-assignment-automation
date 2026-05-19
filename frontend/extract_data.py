"""
extract_data.py — Generate JSON data files for the IDISC HUD frontend.
Reads Data.csv (semicolon-separated, comma decimal) and produces:
  - translators.json  (~900 unique translators with aggregated stats)
  - clients.json      (unique clients/manufacturers with industry info)
"""
import pandas as pd
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(SCRIPT_DIR, "..", "DATA", "Initial Dataset", "CSV", "Data.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Reading {DATA_CSV} ...")
df = pd.read_csv(DATA_CSV, sep=";", decimal=",", encoding="utf-8", low_memory=False)
print(f"  {len(df):,} rows loaded.")

# ── TRANSLATORS ──────────────────────────────────────────────────────────────
print("\nExtracting translators ...")
# Aggregate per translator
tg = df.groupby("TRANSLATOR").agg(
    task_count=("TASK_ID", "nunique"),
    avg_quality=("QUALITY_EVALUATION", "mean"),
    avg_rate=("HOURLY_RATE", "mean"),
    total_hours=("HOURS", "sum"),
    task_types=("TASK_TYPE", lambda x: list(x.dropna().unique())),
    source_langs=("SOURCE_LANG", lambda x: list(x.dropna().unique())),
    target_langs=("TARGET_LANG", lambda x: list(x.dropna().unique())),
    clients_worked=("MANUFACTURER", lambda x: list(x.dropna().unique())),
    last_task=("START", "max"),
).reset_index()

translators = []
for _, r in tg.iterrows():
    # Determine primary language pair (most common)
    tr_data = df[df["TRANSLATOR"] == r["TRANSLATOR"]]
    pair_counts = tr_data.groupby(["SOURCE_LANG", "TARGET_LANG"]).size().reset_index(name="count")
    top_pair = pair_counts.sort_values("count", ascending=False).iloc[0] if len(pair_counts) > 0 else None

    translators.append({
        "id": f"TR-{len(translators)+1:04d}",
        "name": r["TRANSLATOR"],
        "source": top_pair["SOURCE_LANG"] if top_pair is not None else "N/A",
        "target": top_pair["TARGET_LANG"] if top_pair is not None else "N/A",
        "rate": round(r["avg_rate"], 1) if pd.notna(r["avg_rate"]) else 0,
        "quality": round(r["avg_quality"], 1) if pd.notna(r["avg_quality"]) else 0,
        "taskTypes": r["task_types"],
        "taskCount": int(r["task_count"]),
        "totalHours": round(r["total_hours"], 1) if pd.notna(r["total_hours"]) else 0,
        "sourceLangs": r["source_langs"],
        "targetLangs": r["target_langs"],
        "clientsWorked": len(r["clients_worked"]),
        "status": "Available",
        "workerType": "Internal" if r["task_count"] > 20 else "Third-Party",
    })

out_path = os.path.join(OUT_DIR, "translators.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(translators, f, ensure_ascii=False, indent=None)
print(f"  {len(translators)} translators → {out_path}")

# ── CLIENTS ──────────────────────────────────────────────────────────────────
print("\nExtracting clients ...")
cg = df.groupby("MANUFACTURER").agg(
    industry=("MANUFACTURER_INDUSTRY", "first"),
    sector=("MANUFACTURER_SECTOR", "first"),
    industry_group=("MANUFACTURER_INDUSTRY_GROUP", "first"),
    subindustry=("MANUFACTURER_SUBINDUSTRY", "first"),
    task_count=("TASK_ID", "nunique"),
    avg_sell_rate=("HOURLY_RATE", "mean"),
    avg_quality=("QUALITY_EVALUATION", "mean"),
    translators_used=("TRANSLATOR", "nunique"),
    task_types=("TASK_TYPE", lambda x: list(x.dropna().unique())),
).reset_index()

clients = []
for _, r in cg.iterrows():
    clients.append({
        "id": f"CL-{len(clients)+1:04d}",
        "name": r["MANUFACTURER"],
        "industry": r["industry"] if pd.notna(r["industry"]) else "General",
        "sector": r["sector"] if pd.notna(r["sector"]) else "",
        "industryGroup": r["industry_group"] if pd.notna(r["industry_group"]) else "",
        "subindustry": r["subindustry"] if pd.notna(r["subindustry"]) else "",
        "taskCount": int(r["task_count"]),
        "avgRate": round(r["avg_sell_rate"], 1) if pd.notna(r["avg_sell_rate"]) else 0,
        "avgQuality": round(r["avg_quality"], 1) if pd.notna(r["avg_quality"]) else 0,
        "translatorsUsed": int(r["translators_used"]),
        "taskTypes": r["task_types"],
    })

out_path = os.path.join(OUT_DIR, "clients.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(clients, f, ensure_ascii=False, indent=None)
print(f"  {len(clients)} clients → {out_path}")

print("\nDone!")
