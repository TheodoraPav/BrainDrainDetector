"""
Structured stdout logging for the BrainDrainDetector pipeline.

Log lines use a fixed prefix so runs can be parsed from pasted output:
  [STEP 02 START] ...
  [STEP 02 STAT]  key=value
  [STEP 02 OK]    summary
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping


def stage_start(step: str, message: str = "") -> None:
    suffix = f" {message}" if message else ""
    print(f"[STEP {step} START]{suffix}")


def stage_ok(step: str, message: str = "") -> None:
    suffix = f" {message}" if message else ""
    print(f"[STEP {step} OK]{suffix}")


def stage_fail(step: str, message: str) -> None:
    print(f"[STEP {step} FAIL] {message}")


def log_stat(key: str, value) -> None:
    print(f"[STEP -- STAT] {key}={value}")


def log_stats(step: str, stats: Mapping[str, object]) -> None:
    for key, value in stats.items():
        print(f"[STEP {step} STAT] {key}={value}")


def summarize_counts(values: Iterable[int]) -> Dict[str, int | float]:
    values_list = list(values)
    if not values_list:
        return {"count": 0, "min": 0, "median": 0, "max": 0, "total": 0}

    sorted_values = sorted(values_list)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
    else:
        median = float(sorted_values[mid])

    return {
        "count": n,
        "min": sorted_values[0],
        "median": median,
        "max": sorted_values[-1],
        "total": sum(sorted_values),
    }


def format_count_summary(values: Iterable[int]) -> str:
    summary = summarize_counts(values)
    return (
        f"min={summary['min']} "
        f"median={summary['median']} "
        f"max={summary['max']} "
        f"total={summary['total']}"
    )


def log_participant_counts(step: str, counts: Mapping[str, int], limit: int = 0) -> None:
    """Print per-participant counts. Set limit>0 to print only the first N rows."""
    for idx, participant in enumerate(sorted(counts, key=lambda pid: int(pid[1:]))):
        if limit and idx >= limit:
            remaining = len(counts) - limit
            print(f"[STEP {step} STAT] participant_counts=... (+{remaining} more)")
            break
        print(f"[STEP {step} STAT] participant={participant} samples={counts[participant]}")
