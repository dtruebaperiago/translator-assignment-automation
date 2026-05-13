"""
demand.py
=========
Reads a client-demand CSV, enriches each demand with data from the
raw reference CSVs (Clients, Schedules, TranslatorsCost+Pairs) and the
generated Translators_Data.csv, then filters the full translator pool
to only those who satisfy the three hard constraints:

    1. Language-pair match  – translator has worked source -> target before.
    2. Task-type match      – translator has worked this task type before.
    3. Schedule match       – translator has enough available hours in the
                              [START, END] demand window.

Usage (standalone):
    python demand.py --demands demands.csv --data-dir DATA

Import usage (from filters.py):
    from backend.constraints.demand import (
        load_demands, enrich_demands, filter_translators, load_csv_data,
    )
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

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

# Bias multiplier applied to the forecasted hours to give translators
# extra buffer for unexpected problems (e.g. 1.5 -> 50 % slack required).
AVAILABILITY_BIAS: float = 1.5


#Define Demand
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

    # From the Clients CSV (filled in by enrich_demands, None until then)
    selling_hourly_price: Optional[float] = field(default=None)
    min_quality: Optional[float] = field(default=None)
    wildcard: Optional[str] = field(default=None)

    # Convenience helpers
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
            f"Demand({self.task_type}, {self.source_lang}->{self.target_lang}, "
            f"start={self.start:%Y-%m-%d %H:%M}, hours={self.hours}h, "
            f"manufacturer={self.manufacturer})"
        )


# 1. Load demands from CSV
def load_demands(csv_path: str) -> list[Demand]:
    """
    Read a CSV file and return a list of Demand objects.

    The CSV must contain (at minimum) the columns listed in
    DEMAND_CSV_COLS. Extra columns are ignored.
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


# 2. Enrich demands with client data from Clients.csv
def enrich_demands(
    demands: list[Demand],
    clients_df: pd.DataFrame,
    manufacturer_col: str = "CLIENT_NAME",
) -> list[Demand]:
    """
    Add SELLING_HOURLY_PRICE, MIN_QUALITY, and WILDCARD to each demand
    by looking up the demand's manufacturer in the Clients CSV.

    The Clients CSV maps CLIENT_NAME -> {SELLING_HOURLY_PRICE, MIN_QUALITY, WILDCARD}.
    """
    df = clients_df.copy()
    df.columns = [c.strip() for c in df.columns]

    if manufacturer_col not in df.columns:
        raise ValueError(
            f"Clients CSV does not contain a '{manufacturer_col}' column. "
            f"Available columns: {list(df.columns)}"
        )

    # Build lookup: manufacturer name -> {col: value}
    lookup = df.set_index(manufacturer_col).to_dict(orient="index")

    for demand in demands:
        client_row = lookup.get(demand.manufacturer, {})
        demand.selling_hourly_price = client_row.get("SELLING_HOURLY_PRICE")
        demand.min_quality = client_row.get("MIN_QUALITY")
        demand.wildcard = client_row.get("WILDCARD")

    return demands


# 3. Filter translators (hard constraints)
def filter_translators(
    demand: Demand,
    pairs_df: pd.DataFrame,
    translators_data_df: pd.DataFrame,
    schedules_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the subset of translators who satisfy the hard constraints.

    Hard constraints:
    1. Language-pair match -- translator has a row in TranslatorsCost+Pairs
       with matching SOURCE_LANG and TARGET_LANG.
    2. Task-type match -- translator's task_types_worked column in
       Translators_Data.csv contains the demand's TASK_TYPE.
    3. Capacity check -- translator has enough available hours in the
       [demand.start, demand.end] window (using raw schedule shifts):
       available_hours >= demand.hours * AVAILABILITY_BIAS

    Parameters
    ----------
    demand : Demand
    pairs_df : pd.DataFrame
        TranslatorsCost+Pairs.csv with columns:
        TRANSLATOR, SOURCE_LANG, TARGET_LANG, HOURLY_RATE.
    translators_data_df : pd.DataFrame
        Translators_Data.csv with columns:
        TRANSLATOR, task_types_worked, rolling features, flags.
    schedules_df : pd.DataFrame
        Schedules.csv with columns:
        NAME, START, END, MON, TUES, WED, THURS, FRI, SAT, SUN.

    Returns
    -------
    (passed_all, passed_c1c2_only) : tuple[pd.DataFrame, pd.DataFrame]
        passed_all        : translators that passed all 3 constraints.
        passed_c1c2_only  : translators that passed C1+C2 but FAILED C3
                            (used by client.py to re-evaluate with relaxed
                            deadline bias when wildcard == 'Deadline').
    """
    empty = pd.DataFrame(columns=["TRANSLATOR"])

    # --- Constraint 1: Language-pair match (from TranslatorsCost+Pairs) ---
    lang_mask = (
        (pairs_df["SOURCE_LANG"].str.strip() == demand.source_lang)
        & (pairs_df["TARGET_LANG"].str.strip() == demand.target_lang)
    )
    translators_lang = set(
        pairs_df.loc[lang_mask, "TRANSLATOR"].str.strip().unique()
    )

    if not translators_lang:
        print(
            f"[filter] No translators found for language pair "
            f"'{demand.source_lang}->{demand.target_lang}'."
        )
        return empty, empty

    # --- Constraint 2: Task-type match (from Translators_Data.csv) ---
    task_mask = translators_data_df["task_types_worked"].apply(
        lambda x: demand.task_type in str(x).split(",")
    )
    translators_task = set(
        translators_data_df.loc[task_mask, "TRANSLATOR"].str.strip().unique()
    )

    if not translators_task:
        print(
            f"[filter] No translators found with task type "
            f"'{demand.task_type}'."
        )
        return empty, empty

    # Intersection: must satisfy BOTH constraints 1 & 2
    qualified_from_history = translators_lang & translators_task

    if not qualified_from_history:
        print(
            f"[filter] No translators found for "
            f"'{demand.source_lang}->{demand.target_lang}' AND "
            f"task type '{demand.task_type}'."
        )
        return empty, empty

    # --- Constraint 2.5: Not currently assigned ---
    unassigned_mask = translators_data_df.get("assigned", 0) == 0
    unassigned_translators = set(
        translators_data_df.loc[unassigned_mask, "TRANSLATOR"].str.strip().unique()
    )

    qualified_from_history = qualified_from_history & unassigned_translators

    if not qualified_from_history:
        print("[filter] No translators found that are currently unassigned.")
        return empty, empty

    print(
        f"  [C1+C2+C2.5] {len(qualified_from_history)} translator(s) match "
        f"lang pair + task type + unassigned."
    )

    # --- Constraint 3: Schedule capacity check ---
    required_hours = demand.hours * AVAILABILITY_BIAS

    # Filter schedules to only qualified translators
    sched_filtered = schedules_df[
        schedules_df["NAME"].str.strip().isin(qualified_from_history)
    ].copy()

    if sched_filtered.empty:
        print("[filter] No schedule entries for history-qualified translators.")
        return empty, empty

    # Parse shift START/END to time objects
    sched_filtered["_START_TIME"] = sched_filtered["START"].apply(parse_time)
    sched_filtered["_END_TIME"] = sched_filtered["END"].apply(parse_time)

    # Compute available hours for each translator in the demand window
    sched_filtered["_AVAILABLE_HOURS"] = sched_filtered.apply(
        lambda row: compute_available_hours(row, demand.start, demand.end),
        axis=1,
    )

    available_mask = sched_filtered["_AVAILABLE_HOURS"] >= required_hours
    passed_names = set(
        sched_filtered.loc[available_mask, "NAME"].str.strip().unique()
    )
    failed_names = set(
        sched_filtered.loc[~available_mask, "NAME"].str.strip().unique()
    )

    # Build result DataFrames
    passed_all = translators_data_df[
        translators_data_df["TRANSLATOR"].str.strip().isin(passed_names)
    ].copy().reset_index(drop=True)

    passed_c1c2_only = translators_data_df[
        translators_data_df["TRANSLATOR"].str.strip().isin(failed_names)
    ].copy().reset_index(drop=True)

    print(
        f"  [C3] {len(passed_all)} passed all | "
        f"{len(passed_c1c2_only)} passed C1+C2 only (failed capacity at bias {AVAILABILITY_BIAS})"
    )

    return passed_all, passed_c1c2_only


# Helpers
def compute_available_hours(
    row: pd.Series,
    demand_start: datetime,
    demand_end: datetime,
) -> float:
    """
    Sum the working hours a translator has available between
    demand_start and demand_end using raw schedule data.

    Uses the translator's weekly schedule (MON–SUN binary columns +
    shift START / END time columns). Handles partial first / last days:
    only the overlap between the translator's shift and the demand window
    on each date is counted.
    """
    shift_start: time = row["_START_TIME"]
    shift_end: time = row["_END_TIME"]

    total_hours = 0.0
    current_date = demand_start.date()
    end_date = demand_end.date()

    while current_date <= end_date:
        dow_col = WEEKDAY_TO_COL[current_date.weekday()]

        # Translator must work this weekday (column value == 1)
        if int(row.get(dow_col, 0)) == 1:
            # Build working window for this specific day
            day_work_start = datetime.combine(current_date, shift_start)
            day_work_end = datetime.combine(current_date, shift_end)

            # Clamp to the demand window
            overlap_start = max(day_work_start, demand_start)
            overlap_end = min(day_work_end, demand_end)

            if overlap_end > overlap_start:
                total_hours += (overlap_end - overlap_start).total_seconds() / 3600.0

        current_date += timedelta(days=1)

    return total_hours


def parse_time(value) -> time:
    """
    Convert a schedule start/end value to a datetime.time object.

    Handles: datetime.time, datetime.datetime, pd.Timestamp, str "HH:MM:SS".
    """
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, pd.Timestamp):
        return value.time()
    # String fallback
    value = str(value).strip()
    try:
        return pd.to_datetime(value).time()
    except Exception:
        pass
    parts = value.split(":")
    if len(parts) >= 2:
        return time(int(parts[0]), int(parts[1]),
                     int(parts[2]) if len(parts) > 2 else 0)
    raise ValueError(f"Cannot parse time value: {value!r}")


# Load all reference CSVs in one call
def load_csv_data(
    data_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the four reference CSV files from the DATA directory.

    Parameters
    ----------
    data_dir : str
        Path to the DATA/ directory containing the CSVs.

    Returns
    -------
    (clients_df, schedules_df, pairs_df, translators_data_df)
    """
    import os

    # Raw CSVs use semicolon separator, comma decimal
    clients_path = os.path.join(data_dir, "Clients.csv")
    schedules_path = os.path.join(data_dir, "Schedules.csv")
    pairs_path = os.path.join(data_dir, "TranslatorsCost+Pairs.csv")
    # Generated CSV uses standard comma separator
    translators_data_path = os.path.join(data_dir, "Processed", "Translators_Data.csv")

    print(f"  Loading Clients from:              {clients_path}")
    clients_df = pd.read_csv(clients_path, sep=";", decimal=",", encoding="utf-8")
    clients_df = clients_df.loc[:, ~clients_df.columns.str.startswith("Unnamed")]

    print(f"  Loading Schedules from:            {schedules_path}")
    schedules_df = pd.read_csv(schedules_path, sep=";", decimal=",", encoding="utf-8")

    print(f"  Loading TranslatorsCost+Pairs from:{pairs_path}")
    pairs_df = pd.read_csv(pairs_path, sep=";", decimal=",", encoding="utf-8")

    print(f"  Loading Translators_Data from:     {translators_data_path}")
    translators_data_df = pd.read_csv(translators_data_path)

    return clients_df, schedules_df, pairs_df, translators_data_df


# CLI entry point (for quick standalone testing)
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter translators against a client demand CSV (CSV mode)."
    )
    p.add_argument("--demands", required=True,
                   help="Path to the client-demand CSV file.")
    p.add_argument("--data-dir", default="DATA",
                   help="Path to the DATA/ directory with reference CSVs.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    print(f"Loading demands from: {args.demands}")
    demands = load_demands(args.demands)
    print(f"  -> {len(demands)} demand(s) loaded.\n")

    print("Loading reference CSVs...")
    clients_df, schedules_df, pairs_df, translators_data_df = load_csv_data(
        args.data_dir
    )

    enrich_demands(demands, clients_df)

    for i, demand in enumerate(demands, start=1):
        print(f"\n=== Demand {i}/{len(demands)}: {demand} ===")
        qualified = filter_translators(
            demand, pairs_df, translators_data_df, schedules_df
        )
        if qualified.empty:
            print("  No qualified translators.")
        else:
            print(f"  Qualified translators ({len(qualified)}):")
            print(qualified["TRANSLATOR"].to_string(index=False))


if __name__ == "__main__":
    main()
