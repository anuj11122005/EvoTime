"""
test_sample_data.py — Unit Tests for EvoTime Sample Datasets.

Verifies that sample_data provides valid data structures for both feasible
and infeasible scenarios, ensures all lecture IDs are strictly unique, and verifies
the exact lecture and period counts match specifications.
"""

import pytest
from evotime.sample_data import (
    get_feasible_dataset,
    get_infeasible_dataset,
    TimetableDataset,
)


def test_feasible_dataset_structure():
    """Verify the feasible dataset contents, counts, and unpackability."""
    dataset = get_feasible_dataset()
    assert isinstance(dataset, TimetableDataset)
    assert len(dataset.batches) == 2
    assert len(dataset.teachers) == 5
    assert len(dataset.rooms) == 5
    assert len(dataset.subjects) == 8

    # Verify tuple unpacking works cleanly
    subjects, teachers, rooms, batches, lectures = dataset
    assert len(lectures) == 38  # 19 sessions per batch * 2 batches


def test_feasible_dataset_lecture_and_period_counts():
    """
    Verify that each batch requires exactly 19 lecture sessions
    and 21 total periods, resulting in 42 periods total out of 50 available.
    """
    dataset = get_feasible_dataset()

    for batch_id in ["B_CSE_A", "B_CSE_B"]:
        batch_lectures = [l for l in dataset.lectures if l.batch_id == batch_id]
        assert len(batch_lectures) == 19

        total_periods = sum(l.duration for l in batch_lectures)
        assert total_periods == 21

        # Check theory vs practical breakdown
        theory_lectures = [l for l in batch_lectures if not l.needs_lab]
        lab_lectures = [l for l in batch_lectures if l.needs_lab]

        assert len(theory_lectures) == 17  # 3+3+3+3+3+2 sessions
        assert sum(l.duration for l in theory_lectures) == 17

        assert len(lab_lectures) == 2      # 2 lab sessions
        assert sum(l.duration for l in lab_lectures) == 4

    total_all_periods = sum(l.duration for l in dataset.lectures)
    assert total_all_periods == 42


def test_lecture_ids_are_strictly_unique():
    """Ensure that every generated lecture requirement has a globally unique ID."""
    dataset = get_feasible_dataset()
    lecture_ids = [l.lecture_id for l in dataset.lectures]
    assert len(lecture_ids) == len(set(lecture_ids))


def test_infeasible_dataset_structure():
    """Verify that the infeasible dataset is created with the expected overload properties."""
    dataset = get_infeasible_dataset()
    # Dr. Mehta has 3 available slots but teaches 6 periods of Soft Computing
    mehta = dataset.teachers["T_MEHTA"]
    assert len(mehta.available_slots) == 3

    # Batch CSE-A has extra subject CS399 making its required periods > 25
    csea_lectures = [l for l in dataset.lectures if l.batch_id == "B_CSE_A"]
    csea_periods = sum(l.duration for l in csea_lectures)
    assert csea_periods == 27  # 21 standard + 6 robotics = 27 (> 25 max slots)

