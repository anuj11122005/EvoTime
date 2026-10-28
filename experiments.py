"""
experiments.py — Multi-Seed Benchmark and Experimentation Suite for EvoTime.

Why multi-seed evaluation is necessary:
A Genetic Algorithm is stochastic. Evaluating a single run with a single seed (like seed 42)
proves very little: it could have encountered an unusually easy or unusually difficult
starting state. To scientifically assess algorithm efficacy and tuning, we test over a
suite of independent seeds.

Per guidelines:
- TUNING seeds (1–10): Used exclusively for parameter exploration and ablation.
- TEST seeds (101–120): Kept strictly hold-out for reporting final generalization performance.
"""

import argparse
import os
import statistics
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from evotime.config import GAConfig
from evotime.ga import run_ga
from evotime.sample_data import get_feasible_dataset


def run_experiment_suite(
    seeds: List[int],
    config_overrides: Optional[Dict[str, Any]] = None,
    csv_output_path: Optional[str] = None,
    dataset: Any = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Executes the GA over a specified list of random seeds, measures performance,
    prints a formatted telemetry table, and saves results to a CSV file.

    Args:
        seeds: List of integer seeds to run.
        config_overrides: Dictionary of GAConfig hyperparameter overrides.
        csv_output_path: Path where the resulting CSV table should be saved.
        dataset: Dataset instance (defaults to feasible dataset if None).
        verbose: Whether to print progress and ASCII tables to stdout.

    Returns:
        pandas DataFrame containing the run results.
    """
    if dataset is None:
        dataset = get_feasible_dataset()

    records: List[Dict[str, Any]] = []

    if verbose:
        print("\n" + "=" * 78)
        print(f"  EVOTIME BENCHMARK SUITE — RUNNING {len(seeds)} SEEDS")
        if config_overrides:
            print(f"  Overrides: {config_overrides}")
        print("=" * 78)
        print(f"  {'Seed':<6} | {'Status':<8} | {'Gens':<6} | {'Hard':<6} | {'Soft':<8} | {'Runtime (s)':<11}")
        print("  " + "-" * 74)

    for s in seeds:
        cfg = GAConfig(seed=s)
        if config_overrides:
            for k, v in config_overrides.items():
                setattr(cfg, k, v)

        t_start = time.perf_counter()
        result = run_ga(dataset, cfg)
        t_elapsed = time.perf_counter() - t_start

        record = {
            "seed": s,
            "status": result.status,
            "generations_run": result.generations_run,
            "final_hard": result.final_breakdown.hard_total,
            "final_soft": result.final_breakdown.soft_weighted,
            "runtime_sec": round(t_elapsed, 2),
            "stop_reason": result.stop_reason,
        }
        records.append(record)

        if verbose:
            print(
                f"  {s:<6} | {result.status:<8} | {result.generations_run:<6} | "
                f"{result.final_breakdown.hard_total:<6} | {result.final_breakdown.soft_weighted:<8.1f} | "
                f"{t_elapsed:<11.2f}"
            )

    df = pd.DataFrame(records)

    # Calculate summary metrics
    total_runs = len(df)
    successes = df[df["status"] == "SUCCESS"]
    success_count = len(successes)
    success_rate = (success_count / total_runs) * 100.0 if total_runs > 0 else 0.0

    avg_runtime = df["runtime_sec"].mean() if not df.empty else 0.0

    if not successes.empty:
        avg_gens_succ = successes["generations_run"].mean()
        med_gens_succ = successes["generations_run"].median()
        avg_soft_succ = successes["final_soft"].mean()
    else:
        avg_gens_succ = float("nan")
        med_gens_succ = float("nan")
        avg_soft_succ = float("nan")

    if verbose:
        print("  " + "-" * 74)
        print(f"  Summary Statistics:")
        print(f"  * Success Rate: {success_count}/{total_runs} ({success_rate:.1f}%)")
        print(f"  * Avg Generations (Successful Runs): {avg_gens_succ:.1f}")
        print(f"  * Median Generations (Successful Runs): {med_gens_succ:.1f}")
        print(f"  * Avg Soft Penalty Units (Successful Runs): {avg_soft_succ:.1f}")
        print(f"  * Avg Runtime per Run: {avg_runtime:.2f}s")
        print("=" * 78 + "\n")

    if csv_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_output_path)), exist_ok=True)
        df.to_csv(csv_output_path, index=False)
        if verbose:
            print(f"[+] Saved experiment results to: {csv_output_path}\n")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoTime Multi-Seed Benchmark Runner")
    parser.add_argument(
        "--seeds",
        choices=["test", "tuning", "custom"],
        default="test",
        help="Seed suite to run: test (101-120), tuning (1-10), or custom",
    )
    parser.add_argument(
        "--custom-seeds",
        type=int,
        nargs="+",
        help="List of custom integer seeds if --seeds custom is chosen",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as CSV (e.g. results/baseline_results.csv)",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=None,
        help="Override population size",
    )
    parser.add_argument(
        "--gens",
        type=int,
        default=None,
        help="Override max generations",
    )
    parser.add_argument(
        "--mut",
        type=float,
        default=None,
        help="Override mutation rate",
    )

    args = parser.parse_args()

    if args.seeds == "test":
        seed_list = list(range(101, 121))  # 20 test seeds: 101 to 120
    elif args.seeds == "tuning":
        seed_list = list(range(1, 11))     # 10 tuning seeds: 1 to 10
    else:
        seed_list = args.custom_seeds if args.custom_seeds else [42]

    overrides = {}
    if args.pop is not None:
        overrides["population_size"] = args.pop
    if args.gens is not None:
        overrides["max_generations"] = args.gens
    if args.mut is not None:
        overrides["mutation_rate"] = args.mut

    run_experiment_suite(
        seeds=seed_list,
        config_overrides=overrides if overrides else None,
        csv_output_path=args.output,
    )


if __name__ == "__main__":
    main()

