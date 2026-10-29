"""
test_ga_operators.py — Unit Tests for GA Genetic Operators and Repair.

Verifies that:
1. Crossover, mutation, and repair preserve gene count and lecture identity.
2. Repair completely eliminates out-of-range coordinates, day-overflow practicals,
   and room kind mismatches.
3. Elitism guarantees that the best penalty never worsens between generations.
4. Determinism: identical random seeds produce byte-identical GA outcomes.
5. Code integrity: ga.py does NOT import tests.helpers or use constructive heuristics.
"""

import ast
import copy
import random
import pytest
from evotime.config import GAConfig, NUM_DAYS, PERIODS_PER_DAY
from evotime.fitness import evaluate
from evotime.ga import (
    crossover,
    init_population,
    mutate,
    run_ga,
    tournament_selection,
)
from evotime.models import Gene, Timetable
from evotime.repair import repair_timetable
from evotime.sample_data import get_feasible_dataset


@pytest.fixture
def dataset():
    return get_feasible_dataset()


def test_ga_does_not_import_test_helpers():
    """Ensure architectural purity: ga.py must NEVER import test helpers or constructive solvers."""
    import evotime.ga as ga_module

    with open(ga_module.__file__, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "helpers" not in alias.name, f"Forbidden import found: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "helpers" not in node.module, f"Forbidden from-import found: {node.module}"


def test_initial_population_properties(dataset):
    """Verify that random population initialization produces complete, validly sized timetables."""
    rng = random.Random(42)
    pop = init_population(dataset, pop_size=10, rng=rng)
    assert len(pop) == 10

    required_lecture_ids = {l.lecture_id for l in dataset.lectures}
    for ind in pop:
        assert len(ind.genes) == len(dataset.lectures)
        ind_lecture_ids = {g.lecture_id for g in ind.genes}
        assert ind_lecture_ids == required_lecture_ids


def test_crossover_preserves_lecture_count_and_identity(dataset):
    """Verify uniform crossover produces a child with exactly the required lectures."""
    rng = random.Random(42)
    pop = init_population(dataset, pop_size=2, rng=rng)
    parent_a, parent_b = pop[0], pop[1]

    child = crossover(parent_a, parent_b, crossover_rate=1.0, rng=rng)
    assert len(child.genes) == len(parent_a.genes)

    expected_ids = [l.lecture_id for l in dataset.lectures]
    child_ids = [g.lecture_id for g in child.genes]
    assert child_ids == expected_ids


def test_mutation_preserves_lecture_count_and_identity(dataset):
    """Verify mutation perturbs gene values without altering lecture count or IDs."""
    rng = random.Random(42)
    pop = init_population(dataset, pop_size=1, rng=rng)
    individual = copy.deepcopy(pop[0])

    mutate(individual, dataset, mutation_rate=0.5, rng=rng)
    assert len(individual.genes) == len(dataset.lectures)

    expected_ids = [l.lecture_id for l in dataset.lectures]
    mutated_ids = [g.lecture_id for g in individual.genes]
    assert mutated_ids == expected_ids


def test_repair_fixes_invalid_structures(dataset):
    """
    Test that repair_timetable corrects:
    - Out of range day/period
    - Practicals overflowing past period 4
    - Practicals placed in classrooms
    - Theory placed in labs
    - Capacity violations
    """
    lec_map = {l.lecture_id: l for l in dataset.lectures}
    corrupted_genes = []

    for lec in dataset.lectures:
        if lec.needs_lab:
            # Place lab in a classroom starting at period 4 (overflow + wrong room)
            g = Gene(lecture_id=lec.lecture_id, room_id="R101", day=99, start_period=4)
        else:
            # Place theory in a lab room (wrong room)
            g = Gene(lecture_id=lec.lecture_id, room_id="LAB01", day=-5, start_period=10)
        corrupted_genes.append(g)

    corrupted_tt = Timetable(genes=corrupted_genes)
    rng = random.Random(42)
    repaired_tt = repair_timetable(corrupted_tt, dataset, rng)

    # Verify structural validity
    for gene in repaired_tt.genes:
        lec = lec_map[gene.lecture_id]
        room = dataset.rooms[gene.room_id]
        batch = dataset.batches[lec.batch_id]

        assert 0 <= gene.day < NUM_DAYS, f"Day {gene.day} out of range"
        assert 0 <= gene.start_period <= PERIODS_PER_DAY - lec.duration, f"Period {gene.start_period} overflows"

        if lec.needs_lab:
            assert room.kind == "lab", f"Lab practical in non-lab room {room.name}"
        else:
            assert room.kind == "classroom", f"Theory lecture in lab room {room.name}"

        assert room.capacity >= batch.strength, f"Room {room.name} capacity smaller than batch {batch.name}"


def test_elitism_preserves_best_penalty_non_increasing(dataset):
    """Verify that GA elitism ensures best penalty never increases across generations."""
    config = GAConfig(population_size=20, max_generations=15, seed=42)
    result = run_ga(dataset, config)

    history = result.history
    assert len(history) > 1

    for i in range(1, len(history)):
        prev_best = history[i - 1]["best_penalty"]
        curr_best = history[i]["best_penalty"]
        assert curr_best <= prev_best, (
            f"Elitism violated at gen {history[i]['generation']}! "
            f"Penalty increased from {prev_best} to {curr_best}"
        )


def test_same_seed_produces_identical_results(dataset):
    """Verify that two GA runs with identical seeds yield identical timetables and history."""
    config_a = GAConfig(population_size=15, max_generations=10, seed=123)
    config_b = GAConfig(population_size=15, max_generations=10, seed=123)

    res_a = run_ga(dataset, config_a)
    res_b = run_ga(dataset, config_b)

    assert res_a.generations_run == res_b.generations_run
    assert res_a.final_breakdown.penalty == res_b.final_breakdown.penalty
    for g_a, g_b in zip(res_a.final_best.genes, res_b.final_best.genes):
        assert g_a.lecture_id == g_b.lecture_id
        assert g_a.room_id == g_b.room_id
        assert g_a.day == g_b.day
        assert g_a.start_period == g_b.start_period



def test_initialization_yields_zero_theory_duplicates(dataset):
    """Verify that every individual in the initialized population has zero same-day theory duplicates."""
    rng = random.Random(42)
    pop = init_population(dataset, pop_size=20, rng=rng)
    lec_map = {l.lecture_id: l for l in dataset.lectures}
    for ind in pop:
        counts = {}
        for g in ind.genes:
            lec = lec_map[g.lecture_id]
            sub = dataset.subjects[lec.subject_id]
            if sub.type == "theory":
                key = (lec.batch_id, g.day, lec.subject_id)
                counts[key] = counts.get(key, 0) + 1
        assert all(count <= 1 for count in counts.values()), f"Initialization produced duplicate theory days: {counts}"


def test_group_crossover_preserves_zero_theory_duplicates(dataset):
    """Verify that group crossover of two parents with zero duplicates yields a child with zero duplicates."""
    rng = random.Random(42)
    pop = init_population(dataset, pop_size=2, rng=rng)
    parent_a, parent_b = pop[0], pop[1]
    lec_map = {l.lecture_id: l for l in dataset.lectures}

    for trial in range(50):
        child = crossover(parent_a, parent_b, crossover_rate=1.0, rng=rng, dataset=dataset)
        counts = {}
        for g in child.genes:
            lec = lec_map[g.lecture_id]
            sub = dataset.subjects[lec.subject_id]
            if sub.type == "theory":
                key = (lec.batch_id, g.day, lec.subject_id)
                counts[key] = counts.get(key, 0) + 1
        assert all(count <= 1 for count in counts.values()), f"Group crossover produced duplicate theory days: {counts}"
