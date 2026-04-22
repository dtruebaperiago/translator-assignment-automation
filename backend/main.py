"""
main.py — AssignMate entry point
=================================
Loads a client-demand CSV plus the reference Excel workbook and runs the
hard-constraint filtering pipeline defined in
``backend/constraints/demand.py``.

Quick start:
    python backend/main.py --demands sample_demand.csv --excel data.xlsx

Optional arguments:
    --top N          Show only the top-N results per demand (default: all).
    --output FILE    Write qualified translators to a CSV file.
"""

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
    load_excel_sheets,
)


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
        "--excel",
        required=True,
        help="Path to the iDISC reference Excel workbook (.xlsx).",
    )
    p.add_argument(
        "--data-sheet",
        default="Data",
        help="Sheet name for the historical task data.",
    )
    p.add_argument(
        "--schedules-sheet",
        default="Schedules",
        help="Sheet name for translator schedules.",
    )
    p.add_argument(
        "--clients-sheet",
        default="Clients",
        help="Sheet name for client information.",
    )
    p.add_argument(
        "--translator-pairs-sheet",
        default="TranslatorsCost+Pairs",
        help="Sheet name for translator pairs.",
    )
    p.add_argument(
        "--manufacturer-col",
        default="CLIENT_NAME",
        help="Column in the Clients sheet that identifies the manufacturer.",
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
    print(f"\n[1/3] Loading demands from: {args.demands}")
    demands: list[Demand] = load_demands(args.demands)
    print(f"      {len(demands)} demand(s) loaded.")

    # 2. Load reference data
    print(f"\n[2/3] Loading reference workbook: {args.excel}")
    language_pairs_df, task_types_df, schedules_df, clients_df = load_excel_sheets(
        args.excel,
        data_sheet=args.data_sheet,
        schedules_sheet=args.schedules_sheet,
        clients_sheet=args.clients_sheet,
        translator_pairs_sheet=args.translator_pairs_sheet,
    )
    print(
        f"\tHistory Tasks: {len(task_types_df)} rows | "
        f"\tSchedules: {len(schedules_df)} rows | "
        f"\tClients: {len(clients_df)} rows | "
        f"\tTranslator Language Pairs: {len(language_pairs_df)} rows"
    )

    # Enrich demands with client-level data (selling price, min quality, wildcard)
    enrich_demands(demands, clients_df, manufacturer_col=args.manufacturer_col)

    # 3. Filter translators for each demand
    print(f"\n[3/3] Filtering translators (hard constraints) …\n{'='*60}")

    all_results: list[pd.DataFrame] = []

    for idx, demand in enumerate(demands, start=1):
        print(f"\nDemand {idx}/{len(demands)}: {demand}")

        if demand.min_quality is not None:
            print(f"  Client min quality : {demand.min_quality}")
        if demand.selling_hourly_price is not None:
            print(f"  Selling hourly price: {demand.selling_hourly_price}")
        if demand.wildcard is not None:
            print(f"  Wildcard           : {demand.wildcard}")

        qualified: pd.DataFrame = filter_translators(demand, language_pairs_df, task_types_df, schedules_df)

        if qualified.empty:
            print("  → No translators passed the hard constraints for this demand.")
            continue

        if args.top is not None:
            qualified = qualified.head(args.top)

        print(f"  → {len(qualified)} translator(s) qualify:")
        for name in qualified["NAME"].tolist():
            print(f"       • {name}")

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
