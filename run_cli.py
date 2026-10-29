"""
run_cli.py — Command-Line Interface Runner for EvoTime.

Runs the Genetic Algorithm engine with configurable datasets and random seeds,
demonstrates real-time generational logging, and outputs the final independent
validator audit report.
"""

import argparse
import sys
from typing import Any

from evotime.config import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_ELITISM_COUNT,
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_MUTATION_RATE,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TOURNAMENT_K,
    GAConfig,
)
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
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help=f"Population size (default: {DEFAULT_POPULATION_SIZE})",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_MAX_GENERATIONS,
        help=f"Maximum generations to run (default: {DEFAULT_MAX_GENERATIONS})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOURNAMENT_K,
        help=f"Tournament selection size k (default: {DEFAULT_TOURNAMENT_K})",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=DEFAULT_MUTATION_RATE,
        help=f"Mutation probability per gene (default: {DEFAULT_MUTATION_RATE})",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=DEFAULT_CROSSOVER_RATE,
        help=f"Crossover probability (default: {DEFAULT_CROSSOVER_RATE})",
    )
    parser.add_argument(
        "--elitism",
        type=int,
        default=DEFAULT_ELITISM_COUNT,
        help=f"Number of elite individuals preserved (default: {DEFAULT_ELITISM_COUNT})",
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

    # 4. Configure GA
    config = GAConfig(
        population_size=args.population,
        max_generations=args.generations,
        tournament_k=args.k,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elitism_count=args.elitism,
        seed=args.seed,
    )

    print(f"\n>>> Dataset: {dataset.name}")
    print(f"    * Total Lecture Sessions: {total_lectures} ({total_periods} periods)")
    print(f"    * Slots per Batch: 25 | Utilization: {utilization:.1f}%")
    print(
        f"    * GA Parameters: Population={config.population_size}, Max Generations={config.max_generations}, "
        f"Tournament k={config.tournament_k}, Mutation Rate={config.mutation_rate}, "
        f"Crossover Rate={config.crossover_rate}, Elitism={config.elitism_count}, Seed={config.seed}"
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
