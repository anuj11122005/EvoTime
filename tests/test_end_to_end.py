"""
test_end_to_end.py — End-to-End System Integration Tests for EvoTime.

Tests full end-to-end runs of the Genetic Algorithm pipeline:
1. Verified Feasible Seeds: Asserts that the GA produces zero-clash, valid timetables
   confirmed by the independent validator for verified seeds.
2. Infeasible Dataset Integrity: Asserts that the feasibility pre-check rejects
   impossible datasets, and even if forced, the GA never achieves "SUCCESS".
3. Validator Authority Invariant: Proves that result.status == "SUCCESS" strictly
   implies validator.validate(final_best).is_valid is True.
"""

import pytest
from evotime.config import GAConfig
from evotime.feasibility import check_feasibility
from evotime.ga import run_ga
from evotime.sample_data import get_feasible_dataset, get_infeasible_dataset
from evotime.validator import validate


@pytest.fixture
def feasible_dataset():
    return get_feasible_dataset()


@pytest.fixture
def infeasible_dataset():
    return get_infeasible_dataset()


@pytest.mark.parametrize("seed", [101, 103, 112])
def test_end_to_end_feasible_dataset_succeeds(feasible_dataset, seed):
    """
    End-to-end test on verified feasible seeds (101, 103, 112):
    Executes the tuned GA and asserts that the independent validator
    confirms zero hard violations (is_valid = True).
    """
    config = GAConfig(seed=seed)
    result = run_ga(feasible_dataset, config)

    # Assert algorithm claims success
    assert result.status == "SUCCESS", f"Seed {seed} failed to reach SUCCESS status"
    assert result.final_breakdown.hard_total == 0

    # Assert independent validator independently confirms valid timetable
    val_report = validate(result.final_best, feasible_dataset)
    assert val_report.is_valid is True
    assert val_report.hard_count == 0
    assert len(val_report.hard_violations) == 0

    # Recount same-day theory duplicates directly in the test itself
    lec_map = {l.lecture_id: l for l in feasible_dataset.lectures}
    theory_counts = {}
    for g in result.final_best.genes:
        lec = lec_map[g.lecture_id]
        sub = feasible_dataset.subjects[lec.subject_id]
        if sub.type == "theory":
            key = (lec.batch_id, g.day, lec.subject_id)
            theory_counts[key] = theory_counts.get(key, 0) + 1
    assert all(count <= 1 for count in theory_counts.values()), (
        f"Seed {seed} produced same-day theory duplicates: "
        f"{[k for k, v in theory_counts.items() if v > 1]}"
    )


def test_end_to_end_infeasible_dataset_rejected(infeasible_dataset):
    """Verify that the feasibility pre-check immediately flags the impossible dataset."""
    report = check_feasibility(infeasible_dataset)
    assert report.ok is False
    assert len(report.errors) >= 2


@pytest.mark.parametrize("seed", [42, 99, 101])
def test_end_to_end_infeasible_dataset_never_reaches_success(infeasible_dataset, seed):
    """
    Verify that even when forced to run on an impossible dataset,
    the GA never claims SUCCESS and the independent validator confirms violations.
    """
    config = GAConfig(population_size=30, max_generations=60, seed=seed)
    result = run_ga(infeasible_dataset, config)

    assert result.status == "FAILED"
    assert result.final_breakdown.hard_total > 0

    # Validator must report remaining violations
    val_report = validate(result.final_best, infeasible_dataset)
    assert val_report.is_valid is False
    assert val_report.hard_count > 0
    assert len(val_report.hard_violations) > 0


def test_status_success_strictly_implies_validator_valid(feasible_dataset):
    """
    Architectural invariant test:
    Proves that result.status == 'SUCCESS' strictly implies validate().is_valid is True.
    A timetable is never allowed to be marked 'SUCCESS' unless the independent auditor approves.
    """
    for seed in [101, 115]:  # 101 succeeds, 115 fails
        config = GAConfig(seed=seed)
        result = run_ga(feasible_dataset, config)
        independent_report = validate(result.final_best, feasible_dataset)

        if result.status == "SUCCESS":
            assert independent_report.is_valid is True
            assert independent_report.hard_count == 0
        else:
            assert result.status == "FAILED"
            assert independent_report.is_valid is False
            assert independent_report.hard_count > 0

