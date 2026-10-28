"""
tuning_experiments.py — Reproducible Ablation Suite on TUNING seeds (1-10).

Re-evaluates the 5 parameter and operator variants explored during Phase 5 tuning:
1. Baseline: pop=50, gen=200, mut=0.08, k=3
2. Variant A: pop=50, gen=250, mut=0.08, k=4
3. Variant B: pop=70, gen=250, mut=0.09, k=3
4. Variant C: pop=60, gen=250, mut=0.09, k=3 (conflict_prob=0.85)
5. Variant D: Variant C + Random Immigrants (injecting 2 immigrants on 25-gen stagnation)

Outputs summary results to `results/tuning_results.csv`.
"""

import copy
import os
import random
import time
from typing import Any, Dict, List, Tuple
import pandas as pd

from evotime.config import GAConfig, NUM_DAYS, PERIODS_PER_DAY
from evotime.fitness import evaluate
from evotime.ga import (
    _get_clashing_gene_indices,
    crossover,
    init_population,
    tournament_selection,
)
from evotime.models import Gene, Timetable
from evotime.repair import repair_timetable
from evotime.sample_data import get_feasible_dataset
from evotime.validator import validate


def mutate_variant_c(
    timetable: Timetable,
    dataset: Any,
    mutation_rate: float,
    rng: random.Random,
    conflict_prob: float = 0.85,
    max_conflict_picks: int = 2,
) -> None:
    """Variant C mutation: up to 2 clashing genes picked with 85% probability."""
    lecture_map = {l.lecture_id: l for l in dataset.lectures}
    classrooms = [r.id for r in dataset.rooms.values() if r.kind == "classroom"]
    labs = [r.id for r in dataset.rooms.values() if r.kind == "lab"]

    clashing_indices = _get_clashing_gene_indices(timetable, dataset)
    if clashing_indices and rng.random() < conflict_prob:
        k = min(len(clashing_indices), max_conflict_picks)
        for target_idx in rng.sample(list(clashing_indices), k):
            gene = timetable.genes[target_idx]
            lec = lecture_map.get(gene.lecture_id)
            if lec:
                max_start = PERIODS_PER_DAY - lec.duration
                gene.day = rng.randint(0, NUM_DAYS - 1)
                gene.start_period = rng.randint(0, max_start)
                room_pool = labs if lec.needs_lab else classrooms
                if room_pool:
                    gene.room_id = rng.choice(room_pool)

    # Standard per-gene uniform mutation
    for idx, gene in enumerate(timetable.genes):
        if rng.random() < mutation_rate:
            lec = lecture_map.get(gene.lecture_id)
            if not lec:
                continue
            max_start = PERIODS_PER_DAY - lec.duration
            m_type = rng.choice(["time", "room", "swap"])
            if m_type == "time":
                gene.day = rng.randint(0, NUM_DAYS - 1)
                gene.start_period = rng.randint(0, max_start)
            elif m_type == "room":
                pool = labs if lec.needs_lab else classrooms
                if pool:
                    gene.room_id = rng.choice(pool)
            elif m_type == "swap":
                candidates = [
                    o_idx for o_idx, o_g in enumerate(timetable.genes)
                    if o_idx != idx and lecture_map[o_g.lecture_id].duration == lec.duration
                ]
                if candidates:
                    o_idx = rng.choice(candidates)
                    other_g = timetable.genes[o_idx]
                    gene.day, other_g.day = other_g.day, gene.day
                    gene.start_period, other_g.start_period = other_g.start_period, gene.start_period


def run_variant(
    dataset: Any,
    config: GAConfig,
    use_variant_c: bool = False,
    immigrants_on_stagnation: bool = False,
) -> Dict[str, Any]:
    rng = random.Random(config.seed)
    population = init_population(dataset, config.population_size, rng)

    breakdowns = [evaluate(ind, dataset, config) for ind in population]
    penalties = [b.penalty for b in breakdowns]

    best_idx = min(range(len(population)), key=lambda i: penalties[i])
    current_best_ind = copy.deepcopy(population[best_idx])
    current_best_b = copy.deepcopy(breakdowns[best_idx])

    stagnant_count = 0

    for gen in range(1, config.max_generations + 1):
        ranked_indices = sorted(range(len(population)), key=lambda i: penalties[i])
        new_population = [
            copy.deepcopy(population[idx]) for idx in ranked_indices[: config.elitism_count]
        ]

        if immigrants_on_stagnation and stagnant_count >= 25:
            new_population.extend(init_population(dataset, 2, rng))
            stagnant_count = 0

        while len(new_population) < config.population_size:
            p_a = tournament_selection(population, penalties, config.tournament_k, rng)
            p_b = tournament_selection(population, penalties, config.tournament_k, rng)
            child = crossover(p_a, p_b, config.crossover_rate, rng)

            if use_variant_c:
                mutate_variant_c(child, dataset, config.mutation_rate, rng)
            else:
                from evotime.ga import mutate
                mutate(child, dataset, config.mutation_rate, rng)

            repair_timetable(child, dataset, rng)
            new_population.append(child)

        population = new_population
        breakdowns = [evaluate(ind, dataset, config) for ind in population]
        penalties = [b.penalty for b in breakdowns]

        gen_best_idx = min(range(len(population)), key=lambda i: penalties[i])
        gen_best_b = breakdowns[gen_best_idx]

        if gen_best_b.penalty < current_best_b.penalty:
            current_best_ind = copy.deepcopy(population[gen_best_idx])
            current_best_b = copy.deepcopy(gen_best_b)
            stagnant_count = 0
        else:
            stagnant_count += 1

        if current_best_b.hard_total == 0 and current_best_b.penalty <= 300:
            break

    val = validate(current_best_ind, dataset)
    return {
        "seed": config.seed,
        "is_valid": val.is_valid,
        "hard": current_best_b.hard_total,
        "val_hard": val.hard_count,
        "soft": current_best_b.soft_weighted,
        "gens": gen,
    }


def evaluate_variant(name: str, params_str: str, config_factory, **kwargs) -> Dict[str, Any]:
    dataset = get_feasible_dataset()
    results = []
    t0 = time.perf_counter()
    for s in range(1, 11):
        cfg = config_factory(s)
        res = run_variant(dataset, cfg, **kwargs)
        results.append(res)
    elapsed = time.perf_counter() - t0

    succ = sum(1 for r in results if r["is_valid"])
    avg_hard = sum(r["val_hard"] for r in results) / len(results)
    avg_gens = (
        sum(r["gens"] for r in results if r["is_valid"]) / max(1, succ)
        if succ > 0 else float("nan")
    )
    print(
        f"  {name:<12} | {params_str:<32} | {succ}/10 ({succ*10.0:4.1f}%) | "
        f"Avg Hard: {avg_hard:4.2f} | Avg Gens: {avg_gens:5.1f} | {elapsed:5.1f}s"
    )
    return {
        "variant": name,
        "parameters": params_str,
        "success_count": f"{succ}/10",
        "success_rate_pct": succ * 10.0,
        "avg_final_hard_violations": round(avg_hard, 2),
        "avg_generations_for_success": round(avg_gens, 1) if succ > 0 else None,
        "runtime_sec": round(elapsed, 2),
    }


def main():
    print("\n" + "=" * 90)
    print("  EVOTIME TUNING ABLATION SUITE (SEEDS 1–10)")
    print("=" * 90)

    rows = []

    # 1. Baseline
    rows.append(
        evaluate_variant(
            "Baseline",
            "pop=50, gen=200, mut=0.08, k=3",
            lambda s: GAConfig(population_size=50, max_generations=200, mutation_rate=0.08, tournament_k=3, seed=s),
        )
    )

    # 2. Variant A
    rows.append(
        evaluate_variant(
            "Variant A",
            "pop=50, gen=250, mut=0.08, k=4",
            lambda s: GAConfig(population_size=50, max_generations=250, mutation_rate=0.08, tournament_k=4, seed=s),
        )
    )

    # 3. Variant B
    rows.append(
        evaluate_variant(
            "Variant B",
            "pop=70, gen=250, mut=0.09, k=3",
            lambda s: GAConfig(population_size=70, max_generations=250, mutation_rate=0.09, tournament_k=3, seed=s),
        )
    )

    # 4. Variant C
    rows.append(
        evaluate_variant(
            "Variant C",
            "pop=60, gen=250, mut=0.09, conflict_prob=0.85",
            lambda s: GAConfig(population_size=60, max_generations=250, mutation_rate=0.09, tournament_k=3, seed=s),
            use_variant_c=True,
        )
    )

    # 5. Variant D
    rows.append(
        evaluate_variant(
            "Variant D",
            "Variant C + Random Immigrants",
            lambda s: GAConfig(population_size=60, max_generations=250, mutation_rate=0.09, tournament_k=3, seed=s),
            use_variant_c=True,
            immigrants_on_stagnation=True,
        )
    )

    print("=" * 90)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/tuning_results.csv", index=False)
    print("[+] Saved tuning results to: results/tuning_results.csv\n")


if __name__ == "__main__":
    main()

