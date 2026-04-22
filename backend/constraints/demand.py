"""
demand.py
=========
Reads a client-demand CSV, enriches each demand with data from the
reference Excel workbook (Clients/Schedules sheet), and filters the full translator
pool to only those who satisfy the three hard constraints:

    1. Language-pair match  – translator has worked source → target before.
    2. Task-type match      – translator has worked this task type before.
    3. Schedule match       – translator is available on the day / at the time
                              given by the demand's START field.

Usage (standalone):
    python demand.py --demands demands.csv --excel data.xlsx --top 5

Import usage (from main.py):
    from backend.constraints.demand import load_demands, enrich_demands, filter_translators
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional
import pandas as pd


# Constants — column names expected in each sheet / CSV

# Columns that MUST be present in the client-demand CSV
DEMAND_CSV_COLS = [
    "START",
    "END",
    "TASK_TYPE",
    "SOURCE_LANG",
    "TARGET_LANG",
    "HOURS",
    "MANUFACTURER",
    "MANUFACTURER_SECTOR",
    "MANUFACTURER_INDUSTRY_GROUP",
    "MANUFACTURER_INDUSTRY",
    "MANUFACTURER_SUBINDUSTRY",
]

SCHEDULE_DOW_COLS = ["MON", "TUES", "WED", "THURS", "FRI", "SAT", "SUN"]
# Mapping from Python weekday() integer to Schedules column name
WEEKDAY_TO_COL = {
    0: "MON",
    1: "TUES",
    2: "WED",
    3: "THURS",
    4: "FRI",
    5: "SAT",
    6: "SUN",
}


# Dataclass
@dataclass
class Demand:
    """A single client task demand."""

    # From the Client Demand CSV
    start: datetime
    end: datetime
    task_type: str
    source_lang: str
    target_lang: str
    hours: float
    manufacturer: str
    manufacturer_sector: str
    manufacturer_industry_group: str
    manufacturer_industry: str
    manufacturer_subindustry: str

    # From the Clients excel sheet (filled in by enrich_demands, None until then)
    selling_hourly_price: Optional[float] = field(default=None)
    min_quality: Optional[float] = field(default=None)
    wildcard: Optional[str] = field(default=None)


    # Convenience helpers for times and prints
    @property
    def day_of_week_col(self) -> str:
        """Return the Schedules column name for the demand's START weekday."""
        return WEEKDAY_TO_COL[self.start.weekday()]

    @property
    def start_time(self) -> time:
        """Return just the time component of START."""
        return self.start.time()

    def __str__(self) -> str:
        return (
            f"Demand({self.task_type}, {self.source_lang}→{self.target_lang}, "
            f"start={self.start:%Y-%m-%d %H:%M}, hours={self.hours}h, "
            f"manufacturer={self.manufacturer})"
        )



# 1. Load demands from CSV to convert to a list of the demands
def load_demands(csv_path: str) -> list[Demand]:
    """
    Read a CSV file and return a list of class "Demand" objects.

    The CSV must contain (at minimum) the columns listed in
    "DEMAND_CSV_COLS". Extra columns are ignored.

    Input:
    csv_path : str / Path to the client demand CSV file.

    Output:
    list[Demand] / List of Demand objects.
    """
    df = pd.read_csv(csv_path, parse_dates=["START", "END"])

    # Validate required columns
    missing = [c for c in DEMAND_CSV_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The demand CSV is missing required columns: {missing}\n"
            f"Expected: {DEMAND_CSV_COLS}"
        )

    demands: list[Demand] = []
    for _, row in df.iterrows():
        demands.append(
            Demand(
                start=pd.to_datetime(row["START"]),
                end=pd.to_datetime(row["END"]),
                task_type=str(row["TASK_TYPE"]).strip(),
                source_lang=str(row["SOURCE_LANG"]).strip(),
                target_lang=str(row["TARGET_LANG"]).strip(),
                hours=float(row["HOURS"]),
                manufacturer=str(row["MANUFACTURER"]).strip(),
                manufacturer_sector=str(row["MANUFACTURER_SECTOR"]).strip(),
                manufacturer_industry_group=str(row["MANUFACTURER_INDUSTRY_GROUP"]).strip(),
                manufacturer_industry=str(row["MANUFACTURER_INDUSTRY"]).strip(),
                manufacturer_subindustry=str(row["MANUFACTURER_SUBINDUSTRY"]).strip(),
            )
        )

    return demands


# 2. Adds info to demands using Clients-sheet data
def enrich_demands(demands: list[Demand], clients_df: pd.DataFrame, manufacturer_col: str = "MANUFACTURER") -> list[Demand]:
    """
    Add ``SELLING_HOURLY_PRICE``, ``MIN_QUALITY``, and ``WILDCARD`` to each
    demand by looking up the demand's manufacturer in the Clients sheet.

    Input:
    demands : list[Demand] / Demands produced by :func:`load_demands`.
    clients_df : pd.DataFrame / The Clients sheet loaded from the reference Excel workbook.
    manufacturer_col : str / Name of the manufacturer-identifier column in the Clients sheet.

    Output:
    list[Demand] / The same list, mutated in place (also returned for convenience).
    """
    # Normalise column names to uppercase for robustness
    clients_df = clients_df.copy()
    clients_df.columns = [c.strip().upper() for c in clients_df.columns]
    manufacturer_col = manufacturer_col.upper()

    if manufacturer_col not in clients_df.columns:
        raise ValueError(
            f"Clients sheet does not contain a '{manufacturer_col}' column. "
            f"Available columns: {list(clients_df.columns)}"
        )

    # Build a lookup dict: manufacturer → {col: value}
    lookup = (
        clients_df.set_index(manufacturer_col)
        .to_dict(orient="index")
    )

    for demand in demands:
        client_row = lookup.get(demand.manufacturer, {})
        demand.selling_hourly_price = client_row.get("SELLING_HOURLY_PRICE")
        demand.min_quality = client_row.get("MIN_QUALITY")
        demand.wildcard = client_row.get("WILDCARD")
        

    return demands


# 3. Filter translators (hard constraints)
def filter_translators(demand: Demand, language_pairs_df: pd.DataFrame, task_types_df: pd.DataFrame, schedules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the subset of translators who satisfy all (three) hard
    constraints for the given demand.

    Hard constraints:
    1. Language-pair match — translator has previously worked on at least
       one task with the same SOURCE_LANG and TARGET_LANG.
    2. Task-type match — translator has previously worked on at least one
       task of the same TASK_TYPE.
    3. Schedule match — the translator works on the weekday of
       ``demand.start`` and their shift window [START, END] covers the
       demand's start time.

    Input:
    demand : Demand / The client demand to evaluate.
    language_pairs_df : pd.DataFrame / The company's historical data sheet with all languages pairs done in columns:
        ``TRANSLATOR``, ``SOURCE_LANG``, ``TARGET_LANG``.
    task_types_df : pd.DataFrame / The company's historical data sheet with all tasks and it's type done in columns:
        ``TRANSLATOR``, ``TASK_TYPE``.
    schedules_df : pd.DataFrame / The company's schedules sheet with columns:
        ``NAME``, ``START``, ``END``, ``MON``, ``TUES``, ``WED``,
        ``THURS``, ``FRI``, ``SAT``, ``SUN``.

    Output:
    pd.DataFrame / A filtered version of ``schedules_df`` (so each row represents one
        qualified translator)
    """

    # --- Normalise column names ---
    language_pairs_df = language_pairs_df.copy()
    language_pairs_df.columns = [c.strip().upper() for c in language_pairs_df.columns]

    task_types_df = task_types_df.copy()
    task_types_df.columns = [c.strip().upper() for c in task_types_df.columns]

    schedules_df = schedules_df.copy()
    schedules_df.columns = [c.strip().upper() for c in schedules_df.columns]

    # Convert schedule START / END to datetime.time objects
    schedules_df["_START_TIME"] = schedules_df["START"].apply(_parse_time)
    schedules_df["_END_TIME"] = schedules_df["END"].apply(_parse_time)

    # Constraints 1 & 2: language pair + task type (from the company's history)
    lang_pair_mask = (
        (language_pairs_df["SOURCE_LANG"].str.strip() == demand.source_lang)
        & (language_pairs_df["TARGET_LANG"].str.strip() == demand.target_lang)
    )
    task_type_mask = task_types_df["TASK_TYPE"].str.strip() == demand.task_type

    # Translators who satisfy BOTH constraints (intersection)
    translators_lang = set(language_pairs_df.loc[lang_pair_mask, "TRANSLATOR"].str.strip().unique())
    translators_task = set(task_types_df.loc[task_type_mask, "TRANSLATOR"].str.strip().unique())
    qualified_from_history = translators_lang & translators_task

    if not qualified_from_history:
        print(
            f"[filter_translators] No translators found for language pair "
            f"'{demand.source_lang}→{demand.target_lang}' and task type "
            f"'{demand.task_type}'."
        )
        return pd.DataFrame(columns=schedules_df.columns)

    # Constraint 3: schedule availability
    dow_col = demand.day_of_week_col
    demand_start_time = demand.start_time

    # Keep only translators from history that appear in Schedules
    sched_filtered = schedules_df[
        schedules_df["NAME"].str.strip().isin(qualified_from_history)
    ].copy()

    if sched_filtered.empty:
        print(
            "[filter_translators] History-qualified translators have no "
            "schedule entries — cannot verify availability."
        )
        return pd.DataFrame(columns=schedules_df.columns)

    # The translator must:
    #   a) Work on the demand's weekday (column value == 1)
    #   b) Demand start time falls within [shift start, shift end]
    if dow_col not in sched_filtered.columns:
        raise ValueError(
            f"Schedules sheet does not have a column '{dow_col}'. "
            f"Available columns: {list(sched_filtered.columns)}"
        )

    available_mask = (
        (sched_filtered[dow_col].astype(int) == 1)
        & (sched_filtered["_START_TIME"] <= demand_start_time)
        & (sched_filtered["_END_TIME"] >= demand_start_time)
    )

    result = sched_filtered[available_mask].copy()

    # Drop internal helper columns
    result = result.drop(columns=["_START_TIME", "_END_TIME"], errors="ignore")

    if result.empty:
        print("[filter_translators] No translators available at the demanded time.")
    else:
        print(
            f"[filter_translators] {len(result)} translator(s) passed all hard "
            f"constraints for demand: {demand}"
        )

    return result.reset_index(drop=True)


# Helpers

def _parse_time(value) -> time:
    """
    Convert a schedule start/end value to a :class:`datetime.time` object.

    Handles:
    - ``datetime.time`` objects (already correct)
    - ``datetime.datetime`` objects (extract ``.time()``)
    - ``str`` in "HH:MM" or "HH:MM:SS" format
    - ``pd.Timestamp``
    """
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, pd.Timestamp):
        return value.time()
    # String fallback
    value = str(value).strip()
    parts = value.split(":")
    if len(parts) >= 2:
        return time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    raise ValueError(f"Cannot parse time value: {value!r}")


# Convenience: load everything from an Excel workbook in one call
def load_excel_sheets(excel_path: str, data_sheet: str = "Data", schedules_sheet: str = "Schedules", clients_sheet: str = "Clients", translator_pairs_sheet: str = "TranslatorsCost+Pairs") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three main sheets from the iDISC Excel workbook.

    Returns
    -------
    (language_pairs_df, task_types_df, schedules_df, clients_df)
    """
    xl = pd.ExcelFile(excel_path)
    language_pairs_df = xl.parse(translator_pairs_sheet)
    task_types_df = xl.parse(data_sheet)
    schedules_df = xl.parse(schedules_sheet)
    clients_df = xl.parse(clients_sheet)
    return language_pairs_df, task_types_df, schedules_df, clients_df


# CLI entry point (for quick testing)
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter translators against a client demand CSV."
    )
    p.add_argument("--demands", required=True, help="Path to the client-demand CSV file.")
    p.add_argument("--excel", required=True, help="Path to the iDISC reference Excel workbook.")
    p.add_argument(
        "--data-sheet", default="Data", help="Name of the historical data sheet (default: Data)."
    )
    p.add_argument(
        "--schedules-sheet",
        default="Schedules",
        help="Name of the schedules sheet (default: Schedules).",
    )
    p.add_argument(
        "--clients-sheet",
        default="Clients",
        help="Name of the clients sheet (default: Clients).",
    )
    p.add_argument(
        "--manufacturer-col",
        default="MANUFACTURER",
        help="Column in the Clients sheet that identifies the manufacturer (default: MANUFACTURER).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    print(f"Loading demands from: {args.demands}")
    demands = load_demands(args.demands)
    print(f"  → {len(demands)} demand(s) loaded.\n")

    print(f"Loading reference data from: {args.excel}")
    history_df, schedules_df, clients_df = load_excel_sheets(
        args.excel,
        data_sheet=args.data_sheet,
        schedules_sheet=args.schedules_sheet,
        clients_sheet=args.clients_sheet,
    )
    print(
        f"  → History rows: {len(history_df)}, "
        f"Schedules rows: {len(schedules_df)}, "
        f"Clients rows: {len(clients_df)}\n"
    )

    # Adds info to demands using Clients-sheet data
    enrich_demands(demands, clients_df, manufacturer_col=args.manufacturer_col)

    # Filter translators for each demand
    for i, demand in enumerate(demands, start=1):
        print(f"=== Demand {i}/{len(demands)}: {demand} ===")
        qualified = filter_translators(demand, history_df, schedules_df)
        if qualified.empty:
            print("  No qualified translators.\n")
        else:
            print(f"  Qualified translators ({len(qualified)}):")
            print(qualified[["NAME"]].to_string(index=False))
            print()


if __name__ == "__main__":
    main()
