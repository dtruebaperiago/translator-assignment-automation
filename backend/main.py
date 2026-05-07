"""
main.py — AssignMate entry point
=================================
Loads a client-demand CSV and the reference CSVs from the DATA/ directory,
then runs the hard-constraint filtering pipeline defined in
``backend/constraints/demand.py`` and the client-constraint filter
from ``backend/constraints/client.py``.

Quick start:
    python backend/main.py --demands sample_demand.csv

Optional arguments:
    --data-dir DIR   Path to the DATA/ folder (default: DATA).
    --top N          Show only the top-N results per demand (default: all).
    --output FILE    Write qualified translators to a CSV file.
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd

# Allow running from the repo root: `python backend/main.py ...`
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.constraints.demand import (
    Demand,
    load_demands,
    enrich_demands,
    filter_translators,
    load_csv_data,
)
from backend.constraints.client import filter_by_client_constraints


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AssignMate — hard-constraint translator filter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--demands",
        required=True,
        help="Path to the client-demand CSV file.",
    )
    p.add_argument(
        "--data-dir",
        default="DATA",
        help="Path to the DATA/ directory with reference CSVs.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=None,
        help="If set, show only the top-N qualified translators per demand.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="If set, write all qualified translators to this CSV file.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # 1. Load demands
    print(f"\n[1/4] Loading demands from: {args.demands}")
    demands: list[Demand] = load_demands(args.demands)
    print(f"      {len(demands)} demand(s) loaded.")

    # 2. Load reference CSVs
    print(f"\n[2/4] Loading reference CSVs from: {args.data_dir}")
    clients_df, schedules_df, pairs_df, translators_data_df = load_csv_data(
        args.data_dir
    )
    print(
        f"\tClients: {len(clients_df)} rows | "
        f"Schedules: {len(schedules_df)} rows | "
        f"Language Pairs: {len(pairs_df)} rows | "
        f"Translators Data: {len(translators_data_df)} rows"
    )

    # Enrich demands with client-level data
    enrich_demands(demands, clients_df)

    # 3. Hard-constraint filter (demand.py)
    # 4. Client-constraint filter (client.py)
    print(f"\n[3/4] Hard-constraint filter (demand.py)")
    print(f"[4/4] Client-constraint filter (client.py)")
    print(f"{'='*60}")

    all_results: list[pd.DataFrame] = []

    for idx, demand in enumerate(demands, start=1):
        print(f"\nDemand {idx}/{len(demands)}: {demand}")

        if demand.min_quality is not None:
            print(f"  Client min quality  : {demand.min_quality}")
        if demand.selling_hourly_price is not None:
            print(f"  Selling hourly price: {demand.selling_hourly_price}")
        if demand.wildcard is not None:
            print(f"  Wildcard            : {demand.wildcard}")

        # Step 3: Hard constraints (language pair, task type, capacity)
        passed_all, passed_c1c2_only = filter_translators(
            demand, pairs_df, translators_data_df, schedules_df
        )

        if passed_all.empty and passed_c1c2_only.empty:
            print("  -> No translators passed the hard constraints.")
            continue

        # Step 4: Client constraints (quality, price, deadline relaxation)
        qualified = filter_by_client_constraints(
            demand, passed_all, passed_c1c2_only,
            pairs_df, translators_data_df, schedules_df,
        )

        if qualified.empty:
            print("  -> No translators passed the client constraints.")
            continue

        if args.top is not None:
            qualified = qualified.head(args.top)

        print(f"  -> {len(qualified)} translator(s) qualify:")
        for name in qualified["TRANSLATOR"].tolist():
            print(f"       * {name}")

        # Tag with demand index for output CSV
        qualified = qualified.copy()
        qualified.insert(0, "DEMAND_IDX", idx)
        all_results.append(qualified)

    # 4. Optional: write to output CSV
    if args.output and all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nResults written to: {args.output}")
    elif args.output:
        print(f"\nNo results to write to {args.output}.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
