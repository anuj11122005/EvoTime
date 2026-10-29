"""
quality_metrics.py — Timetable Quality Telemetry and Constraint Audit.

Evaluates timetables generated across a list of seeds for:
1. Status (SUCCESS or FAILED)
2. Theory subject same-day duplicates per batch: sum of (k - 1) for theory subjects appearing k >= 2 times on the same day.
3. Max consecutive periods taught by any teacher in a day (global).
"""

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from evotime.config import GAConfig, NUM_DAYS, PERIODS_PER_DAY
from evotime.ga import run_ga
from evotime.models import Timetable
from evotime.sample_data import get_feasible_dataset


def compute_theory_duplicates(timetable: Timetable, dataset: Any, batch_id: str) -> int:
    """
    Computes total same-day duplicate units for theory subjects in a batch:
    For each (day, theory_subject), if count k >= 2, adds (k - 1).
    Practicals (needs_lab=True) are exempt.
    """
    lecture_map = {l.lecture_id: l for l in dataset.lectures}
    counts = defaultdict(lambda: defaultdict(int))
    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if not lec or lec.batch_id != batch_id:
            continue
        sub = dataset.subjects.get(lec.subject_id)
        if sub and sub.type == "theory":
            counts[gene.day][lec.subject_id] += 1

    total_violations = 0
    for day, sub_counts in counts.items():
        for sub_id, count in sub_counts.items():
            if count >= 2:
                total_violations += (count - 1)
    return total_violations


def compute_max_teacher_consecutive(timetable: Timetable, dataset: Any) -> int:
    """
    Computes the maximum consecutive periods taught by any teacher on any day.
    Practicals (duration 2) occupy their consecutive periods.
    """
    lecture_map = {l.lecture_id: l for l in dataset.lectures}
    schedule = defaultdict(lambda: defaultdict(set))
    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if not lec:
            continue
        if 0 <= gene.day < NUM_DAYS:
            for p in range(lec.duration):
                period = gene.start_period + p
                if 0 <= period < PERIODS_PER_DAY:
                    schedule[lec.teacher_id][gene.day].add(period)

    max_consecutive = 0
    for t_id, days in schedule.items():
        for d, periods in days.items():
            if not periods:
                continue
            sorted_p = sorted(periods)
            run = 1
            cur_max = 1
            for i in range(1, len(sorted_p)):
                if sorted_p[i] == sorted_p[i - 1] + 1:
                    run += 1
                else:
                    run = 1
                if run > cur_max:
                    cur_max = run
            if cur_max > max_consecutive:
                max_consecutive = cur_max

    return max_consecutive


def run_quality_audit(
    seeds: List[int],
    output_path: Optional[str] = None,
    dataset: Any = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if dataset is None:
        dataset = get_feasible_dataset()

    records = []
    if verbose:
        print("\n" + "=" * 80)
        print(f"  EVOTIME QUALITY METRICS AUDIT — RUNNING {len(seeds)} SEEDS")
        print("=" * 80)
        print(f"  {'Seed':<6} | {'Status':<8} | {'CSE-A Dups':<12} | {'CSE-B Dups':<12} | {'Max Consec':<10}")
        print("  " + "-" * 76)

    for s in seeds:
        cfg = GAConfig(seed=s)
        res = run_ga(dataset, cfg)

        if res.status == "SUCCESS":
            dups_a = compute_theory_duplicates(res.final_best, dataset, "B_CSE_A")
            dups_b = compute_theory_duplicates(res.final_best, dataset, "B_CSE_B")
            max_consec = compute_max_teacher_consecutive(res.final_best, dataset)

            record = {
                "seed": s,
                "status": res.status,
                "cse_a_theory_duplicates": dups_a,
                "cse_b_theory_duplicates": dups_b,
                "max_consecutive_teacher_periods": max_consec,
            }
            records.append(record)
            if verbose:
                print(f"  {s:<6} | {res.status:<8} | {dups_a:<12} | {dups_b:<12} | {max_consec:<10}")
        else:
            record = {
                "seed": s,
                "status": res.status,
                "cse_a_theory_duplicates": None,
                "cse_b_theory_duplicates": None,
                "max_consecutive_teacher_periods": None,
            }
            records.append(record)
            if verbose:
                print(f"  {s:<6} | {res.status:<8} | {'N/A':<12} | {'N/A':<12} | {'N/A':<10}")

    df = pd.DataFrame(records)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print("  " + "-" * 76)
            print(f"[+] Saved quality metrics to: {output_path}\n")

    return df


def parse_seeds_arg(arg: str) -> List[int]:
    if arg == "test":
        return list(range(101, 121))
    elif arg == "tuning":
        return list(range(1, 11))
    elif "-" in arg:
        start, end = map(int, arg.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(s.strip()) for s in arg.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoTime Quality Metrics Runner")
    parser.add_argument(
        "--seeds",
        type=str,
        default="test",
        help="Seed list: 'test' (101-120), 'tuning' (1-10), '101-120', or comma-separated integers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/quality_metrics.csv",
        help="CSV destination path",
    )
    args = parser.parse_args()
    seeds = parse_seeds_arg(args.seeds)
    run_quality_audit(seeds, output_path=args.output)


if __name__ == "__main__":
    main()
