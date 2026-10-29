"""
summarize_results.py — Computes and displays summary statistics directly from results CSV files.

Reads:
- results/baseline_results.csv
- results/tuned_results.csv
- results/tuning_results.csv

Outputs:
- Success count / total (and percentage)
- Average and median generations for successful runs
- Average soft units for successful runs
- Average runtime (seconds)
- List of failed seeds (with their final hard violations)
"""

import os
import statistics
import pandas as pd


def analyze_test_seeds_csv(csv_path: str, label: str):
    print("=" * 78)
    print(f"  ANALYSIS FOR: {label} ({csv_path})")
    print("=" * 78)

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    total_runs = len(df)
    successes = df[df["status"] == "SUCCESS"]
    failures = df[df["status"] != "SUCCESS"]
    success_count = len(successes)
    success_rate = (success_count / total_runs * 100.0) if total_runs > 0 else 0.0

    avg_gens = successes["generations_run"].mean() if not successes.empty else 0.0
    median_gens = successes["generations_run"].median() if not successes.empty else 0.0
    avg_soft = successes["final_soft"].mean() if not successes.empty else 0.0
    avg_runtime = df["runtime_sec"].mean() if not df.empty else 0.0

    print(f"* Total Runs Evaluated:        {total_runs}")
    print(f"* Success Count / Total:       {success_count} / {total_runs} ({success_rate:.1f}%)")
    print(f"* Avg Gens (Successful Runs):   {avg_gens:.2f}")
    print(f"* Median Gens (Successful):    {median_gens:.1f}")
    print(f"* Avg Soft Units (Successful): {avg_soft:.2f}")
    print(f"* Avg Runtime per Run:         {avg_runtime:.2f}s")

    if not failures.empty:
        print(f"* Failed Seeds ({len(failures)}):")
        for _, row in failures.iterrows():
            print(f"    - Seed {int(row['seed'])}: {int(row['final_hard'])} hard violations (ran {int(row['generations_run'])} gens, stop: {row.get('stop_reason', 'N/A')})")
    else:
        print("* Failed Seeds: None (100% success)")
    print()


def analyze_tuning_csv(csv_path: str):
    print("=" * 78)
    print(f"  TUNING ABLATION SUMMARY ({csv_path})")
    print("=" * 78)

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        print(f"Variant: {row['variant']}")
        print(f"  - Parameters:                {row['parameters']}")
        print(f"  - Success Count / Rate:      {row['success_count']} ({row['success_rate_pct']}%)")
        print(f"  - Avg Final Hard Violations: {row['avg_final_hard_violations']}")
        print(f"  - Avg Gens for Successes:    {row['avg_generations_for_success']}")
        print(f"  - Total Runtime:             {row['runtime_sec']}s")
    print()


def main():
    analyze_test_seeds_csv("results/baseline_results.csv", "BASELINE CONFIGURATION (TEST SEEDS 101–120)")
    analyze_test_seeds_csv("results/tuned_results.csv", "TUNED CONFIGURATION (TEST SEEDS 101–120)")
    analyze_tuning_csv("results/tuning_results.csv")


if __name__ == "__main__":
    main()

