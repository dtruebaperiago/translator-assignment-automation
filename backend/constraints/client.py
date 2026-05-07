"""
client.py
=========
Second-pass filter that enforces client-specific constraints using
biased relaxation governed by the client's WILDCARD field.

Instead of completely skipping a check when it matches the wildcard,
each constraint is always evaluated but with a relaxed bias when the
client is flexible about that dimension.

Wildcard behaviour:
    "Quality"  -> quality threshold lowered  (min_quality * 0.7)
    "Price"    -> price ceiling raised        (selling_price * 2.0)
    "Deadline" -> capacity bias reduced       (0.75 for punctual, 1.0 for others)

Usage:
    from backend.constraints.client import filter_by_client_constraints
"""

from __future__ import annotations

import pandas as pd

from backend.constraints.demand import (
    Demand,
    WEEKDAY_TO_COL,
    AVAILABILITY_BIAS,
    compute_available_hours,
    parse_time,
)


# ── Bias constants ───────────────────────────────────────────────────────────

QUALITY_WILDCARD_BIAS: float = 0.7     # lower the quality bar by 30%
PRICE_WILDCARD_BIAS: float = 2.0       # raise the price ceiling by 100%
DEADLINE_BIAS_TRUSTED: float = 0.75    # for punctuality >= PUNCTUALITY_THRESHOLD
DEADLINE_BIAS_NORMAL: float = 1.0      # for punctuality < PUNCTUALITY_THRESHOLD
PUNCTUALITY_THRESHOLD: float = 0.8     # cutoff for "trusted" translators


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------

def filter_by_client_constraints(
    demand: Demand,
    passed_all: pd.DataFrame,
    passed_c1c2_only: pd.DataFrame,
    pairs_df: pd.DataFrame,
    translators_data_df: pd.DataFrame,
    schedules_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply client-specific constraints (quality, price, deadline) with
    biased relaxation based on the client's WILDCARD.

    Parameters
    ----------
    demand : Demand
        Enriched demand (must have selling_hourly_price, min_quality, wildcard).
    passed_all : pd.DataFrame
        Translators that passed ALL hard constraints from demand.py.
    passed_c1c2_only : pd.DataFrame
        Translators that passed C1+C2 but FAILED C3 (capacity) in demand.py.
        These get a second chance when wildcard == "Deadline".
    pairs_df : pd.DataFrame
        TranslatorsCost+Pairs.csv (TRANSLATOR, SOURCE_LANG, TARGET_LANG, HOURLY_RATE).
    translators_data_df : pd.DataFrame
        Translators_Data.csv (TRANSLATOR, rolling_quality_ema, ...).
    schedules_df : pd.DataFrame
        Schedules.csv (NAME, START, END, MON-SUN).

    Returns
    -------
    pd.DataFrame
        Final filtered translators that passed all client constraints.
    """
    wildcard = str(demand.wildcard).strip() if demand.wildcard else ""

    # Start with the translators that passed all hard constraints
    candidates = passed_all.copy()

    # --- Deadline wildcard: re-evaluate C3-failed translators ---
    if wildcard == "Deadline" and not passed_c1c2_only.empty:
        recovered = _recover_deadline_candidates(
            demand, passed_c1c2_only, schedules_df
        )
        if not recovered.empty:
            print(
                f"  [CLIENT-Deadline] Recovered {len(recovered)} translator(s) "
                f"with relaxed deadline bias."
            )
            candidates = pd.concat(
                [candidates, recovered], ignore_index=True
            ).drop_duplicates(subset=["TRANSLATOR"])

    if candidates.empty:
        print("  [CLIENT] No candidates to evaluate after deadline check.")
        return pd.DataFrame(columns=["TRANSLATOR"])

    # --- Quality filter (always applied) ---
    candidates = _apply_quality_filter(demand, candidates, wildcard)
    if candidates.empty:
        print("  [CLIENT] No translators passed quality filter.")
        return pd.DataFrame(columns=["TRANSLATOR"])

    # --- Price filter (always applied) ---
    candidates = _apply_price_filter(demand, candidates, pairs_df, wildcard)
    if candidates.empty:
        print("  [CLIENT] No translators passed price filter.")
        return pd.DataFrame(columns=["TRANSLATOR"])

    print(
        f"  [CLIENT] {len(candidates)} translator(s) passed all "
        f"client constraints (wildcard={wildcard})."
    )

    return candidates.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def _apply_quality_filter(
    demand: Demand,
    candidates: pd.DataFrame,
    wildcard: str,
) -> pd.DataFrame:
    """
    Keep translators whose rolling_quality_ema meets the client's
    minimum quality threshold.

    When wildcard == "Quality", the threshold is relaxed by
    QUALITY_WILDCARD_BIAS (e.g. 7.5 * 0.7 = 5.25).
    """
    min_quality = demand.min_quality
    if min_quality is None or min_quality == 0:
        # No quality requirement from this client
        return candidates

    if wildcard == "Quality":
        threshold = min_quality * QUALITY_WILDCARD_BIAS
        label = f"relaxed ({min_quality} * {QUALITY_WILDCARD_BIAS} = {threshold:.2f})"
    else:
        threshold = min_quality
        label = f"strict ({threshold})"

    before = len(candidates)
    mask = candidates["rolling_quality_ema"] >= threshold
    candidates = candidates[mask].copy()

    print(
        f"  [CLIENT-Quality] {label}: "
        f"{before} -> {len(candidates)} translator(s)"
    )

    return candidates


# ---------------------------------------------------------------------------
# Price filter
# ---------------------------------------------------------------------------

def _apply_price_filter(
    demand: Demand,
    candidates: pd.DataFrame,
    pairs_df: pd.DataFrame,
    wildcard: str,
) -> pd.DataFrame:
    """
    Keep translators whose HOURLY_RATE for the exact language pair
    does not exceed the client's selling hourly price.

    When wildcard == "Price", the ceiling is raised by
    PRICE_WILDCARD_BIAS (e.g. 35 * 2.0 = 70).
    """
    selling_price = demand.selling_hourly_price
    if selling_price is None or selling_price == 0:
        # No price constraint from this client
        return candidates

    if wildcard == "Price":
        ceiling = selling_price * PRICE_WILDCARD_BIAS
        label = f"relaxed ({selling_price} * {PRICE_WILDCARD_BIAS} = {ceiling:.2f})"
    else:
        ceiling = selling_price
        label = f"strict ({ceiling})"

    # Look up each candidate's hourly rate for THIS language pair
    pair_mask = (
        (pairs_df["SOURCE_LANG"].str.strip() == demand.source_lang)
        & (pairs_df["TARGET_LANG"].str.strip() == demand.target_lang)
    )
    pair_rates = pairs_df.loc[pair_mask, ["TRANSLATOR", "HOURLY_RATE"]].copy()
    pair_rates["TRANSLATOR"] = pair_rates["TRANSLATOR"].str.strip()
    pair_rates["HOURLY_RATE"] = pd.to_numeric(
        pair_rates["HOURLY_RATE"], errors="coerce"
    )

    # Merge rates into candidates
    merged = candidates.merge(
        pair_rates[["TRANSLATOR", "HOURLY_RATE"]],
        on="TRANSLATOR",
        how="left",
        suffixes=("", "_pair"),
    )

    before = len(merged)
    # Keep translators whose rate for this pair is within the ceiling
    mask = merged["HOURLY_RATE"].fillna(0) <= ceiling
    result = merged[mask].copy()

    # Drop the extra HOURLY_RATE column (it was just for the check)
    if "HOURLY_RATE" in result.columns and "HOURLY_RATE" not in candidates.columns:
        result = result.drop(columns=["HOURLY_RATE"])

    print(
        f"  [CLIENT-Price] {label}: "
        f"{before} -> {len(result)} translator(s)"
    )

    return result


# ---------------------------------------------------------------------------
# Deadline recovery (re-evaluate C3-failed with relaxed bias)
# ---------------------------------------------------------------------------

def _recover_deadline_candidates(
    demand: Demand,
    failed_c3: pd.DataFrame,
    schedules_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Re-evaluate translators that failed the strict capacity check (C3)
    using a relaxed deadline bias gated by punctuality:

    - rolling_punctuality_score >= 0.8  ->  bias = 0.75
    - rolling_punctuality_score <  0.8  ->  bias = 1.0

    Returns the subset that pass the relaxed check.
    """
    failed_names = set(failed_c3["TRANSLATOR"].str.strip().unique())

    sched_filtered = schedules_df[
        schedules_df["NAME"].str.strip().isin(failed_names)
    ].copy()

    if sched_filtered.empty:
        return pd.DataFrame(columns=["TRANSLATOR"])

    # Parse shift times
    sched_filtered["_START_TIME"] = sched_filtered["START"].apply(parse_time)
    sched_filtered["_END_TIME"] = sched_filtered["END"].apply(parse_time)

    # Compute available hours
    sched_filtered["_AVAILABLE_HOURS"] = sched_filtered.apply(
        lambda row: compute_available_hours(row, demand.start, demand.end),
        axis=1,
    )

    # Build a punctuality lookup from the failed_c3 dataframe
    punctuality_lookup = dict(
        zip(
            failed_c3["TRANSLATOR"].str.strip(),
            failed_c3["rolling_punctuality_score"],
        )
    )

    recovered_names = set()
    for _, row in sched_filtered.iterrows():
        name = str(row["NAME"]).strip()
        available = row["_AVAILABLE_HOURS"]
        punctuality = punctuality_lookup.get(name, 0.0)

        # Choose bias based on punctuality
        if punctuality >= PUNCTUALITY_THRESHOLD:
            bias = DEADLINE_BIAS_TRUSTED
        else:
            bias = DEADLINE_BIAS_NORMAL

        required = demand.hours * bias
        if available >= required:
            recovered_names.add(name)

    if not recovered_names:
        return pd.DataFrame(columns=["TRANSLATOR"])

    return failed_c3[
        failed_c3["TRANSLATOR"].str.strip().isin(recovered_names)
    ].copy()
