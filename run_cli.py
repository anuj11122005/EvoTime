"""
run_cli.py — Command-Line Runner for EvoTime Genetic Algorithm.

Allows executing the entire timetabling optimization pipeline directly from
the terminal. Performs pre-run feasibility analysis, initializes the GA, streams
generational convergence progress, and invokes the independent validator to
report the final verdict.

Usage:
    python run_cli.py
    python run_cli.py --dataset feasible --seed 42 --generations 200
    python run_cli.py --dataset infeasible --force
"""

import argparse
import sys
from typing import Any

from evotime.config import GAConfig
from evotime.feasibility import check_feasibility, format_feasibility_summary
from evotime.ga import run_ga
from evotime.sample_data import get_feasible_dataset, get_infeasible_dataset
from evotime.validator import format_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvoTime — Intelligent College Timetable Generator (CLI Runner)"
    )
    parser.add_argument(
        "--dataset",
        choices=["feasible", "infeasible"],
        default="feasible",
        help="Select dataset to run (default: feasible)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=50,
        help="Population size (default: 50)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Maximum generations to run (default: 200)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if feasibility pre-check fails",
    )

    args = parser.parse_args()

    # 1. Load Dataset
    if args.dataset == "feasible":
        dataset = get_feasible_dataset()
    else:
        dataset = get_infeasible_dataset()

    print("\n" + "=" * 70)
    print("  EVOTIME — INTELLIGENT COLLEGE TIMETABLE GENERATOR (CLI)")
    print("=" * 70)

    # 2. Feasibility Pre-Check
    print("\n>>> Step 1: Performing Pre-run Feasibility Analysis...")
    feasibility_report = check_feasibility(dataset)

    if not feasibility_report.ok:
        print("\n[!] FEASIBILITY PRE-CHECK DETECTED FATAL IMPOSSIBILITIES:")
        for err in feasibility_report.errors:
            print(f"    X {err}")

        if not args.force:
            print("\n[-] Pre-check failed. Aborting GA run.")
            print("    (Pass '--force' to execute anyway and demonstrate GA failure behavior).\n")
            sys.exit(1)
        else:
            print("\n[!] '--force' flag provided. Proceeding with GA despite known impossibility...\n")
    else:
        print("[+] Feasibility pre-check passed successfully.")

    # 3. Print Dataset Metrics
    total_lectures = len(dataset.lectures)
    total_periods = sum(l.duration for l in dataset.lectures)
    batch_info = list(dataset.batches.values())[0]
    batch_needed = sum(l.duration for l in dataset.lectures if l.batch_id == batch_info.id)
    utilization = (batch_needed / 25.0) * 100.0

    print(f"\n>>> Dataset: {dataset.name}")
    print(f"    * Total Lecture Sessions: {total_lectures} ({total_periods} periods)")
    print(f"    * Slots per Batch: 25 | Utilization: {utilization:.1f}%")
    print(f"    * GA Parameters: Population={args.population}, Max Generations={args.generations}, Seed={args.seed}")

    # 4. Configure and Run GA
    config = GAConfig(
        population_size=args.population,
        max_generations=args.generations,
        seed=args.seed,
    )

    print("\n>>> Step 2: Launching Genetic Algorithm Search...")

    def progress_callback(gen: int, best_breakdown: Any, avg_penalty: float) -> None:
        if gen == 1 or gen % 10 == 0 or gen == args.generations:
            print(
                f"    Gen {gen:3d} | Best Penalty: {best_breakdown.penalty:7.1f} "
                f"(Hard: {best_breakdown.hard_total:2d}, Soft: {best_breakdown.soft_weighted:5.1f}) | "
                f"Avg Penalty: {avg_penalty:7.1f}"
            )

    result = run_ga(dataset, config, callback=progress_callback)

    # 5. Output Summary
    print("\n" + "=" * 70)
    print("  GENETIC ALGORITHM EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  * Status: {result.status}")
    print(f"  * Generations Executed: {result.generations_run}")
    print(f"  * Termination Reason: {result.stop_reason}")
    print(
        f"  * Initial Best Penalty (Gen 0): {result.initial_breakdown.penalty:.1f} "
        f"(Hard Clashes: {result.initial_breakdown.hard_total}, Soft: {result.initial_breakdown.soft_weighted:.1f})"
    )
    print(
        f"  * Final Best Penalty:          {result.final_breakdown.penalty:.1f} "
        f"(Hard Clashes: {result.final_breakdown.hard_total}, Soft: {result.final_breakdown.soft_weighted:.1f})"
    )

    # 6. Independent Validator Audit
    print("\n>>> Step 3: Independent Validator Audit Report...")
    print(format_validation_report(result.validation_report))

    if result.status == "SUCCESS":
        print("[SUCCESS] Conflict-free timetable successfully synthesized!\n")
    else:
        print("[FAILED] Could not synthesize a conflict-free timetable within generation limits.\n")


if __name__ == "__main__":
    main()

