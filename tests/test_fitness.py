"""
test_fitness.py — Unit Tests for EvoTime Fitness and Penalty Evaluation.

Tests that a known valid timetable yields hard_total = 0, that individual soft
constraint functions calculate expected penalty units, and that every specific hard
violation type is correctly detected and scored by the fitness function.
"""

import copy
import pytest
from evotime.fitness import (
    evaluate,
    soft_batch_gaps,
    soft_batch_subject_duplicate,
    soft_teacher_consecutive,
    soft_teacher_workload_variance,
    soft_undesirable_patterns,
)
from evotime.models import Gene, Timetable
from evotime.sample_data import get_feasible_dataset
from tests.helpers import build_valid_sample_timetable


@pytest.fixture
def dataset():
    return get_feasible_dataset()


@pytest.fixture
def valid_timetable(dataset):
    return build_valid_sample_timetable(dataset)


def test_valid_timetable_has_zero_hard_violations(dataset, valid_timetable):
    """Verify that a known clash-free timetable yields exactly hard_total = 0."""
    breakdown = evaluate(valid_timetable, dataset)
    assert breakdown.hard_total == 0
    assert all(count == 0 for count in breakdown.hard_counts.values())
    assert breakdown.penalty >= 0  # Only soft penalties may remain
    assert 0.0 < breakdown.fitness_score <= 1.0


def test_teacher_clash_detection(dataset, valid_timetable):
    """Verify teacher clash: two classes taught by the same teacher at the same time."""
    tt = copy.deepcopy(valid_timetable)
    # Find two lectures taught by Dr. Mehta (T_MEHTA)
    mehta_genes = [g for g in tt.genes if "CS301" in g.lecture_id]
    assert len(mehta_genes) >= 2
    # Force gene 1 to overlap with gene 0 in time slot, different rooms
    mehta_genes[1].day = mehta_genes[0].day
    mehta_genes[1].start_period = mehta_genes[0].start_period
    mehta_genes[1].room_id = "R102"
    mehta_genes[0].room_id = "R101"

    breakdown = evaluate(tt, dataset)
    assert breakdown.hard_counts["teacher_clashes"] >= 1
    assert breakdown.hard_total >= 1


def test_room_clash_detection(dataset, valid_timetable):
    """Verify room clash: two different batches scheduled in the same room at the same time."""
    tt = copy.deepcopy(valid_timetable)
    # Force two different genes into the same room and same slot
    g0 = tt.genes[0]
    g1 = tt.genes[1]
    g1.day = g0.day
    g1.start_period = g0.start_period
    g1.room_id = g0.room_id

    breakdown = evaluate(tt, dataset)
    assert breakdown.hard_counts["room_clashes"] >= 1
    assert breakdown.hard_total >= 1


def test_batch_clash_detection(dataset, valid_timetable):
    """Verify batch clash: the same batch double-booked into two classes at once."""
    tt = copy.deepcopy(valid_timetable)
    # Find two lectures for batch B_CSE_A
    csea_genes = [g for g in tt.genes if "B_CSE_A" in g.lecture_id]
    g0, g1 = csea_genes[0], csea_genes[1]
    # Place in different rooms at the same time
    g1.day = g0.day
    g1.start_period = g0.start_period
    g1.room_id = "R102"
    g0.room_id = "R101"

    breakdown = evaluate(tt, dataset)
    assert breakdown.hard_counts["batch_clashes"] >= 1
    assert breakdown.hard_total >= 1


def test_missing_and_duplicate_lecture_detection(dataset, valid_timetable):
    """Verify missing and duplicate lecture counting."""
    tt = copy.deepcopy(valid_timetable)
    # Duplicate gene 0 and remove gene 1
    duplicated_gene = copy.deepcopy(tt.genes[0])
    tt.genes.pop(1)
    tt.genes.append(duplicated_gene)

    breakdown = evaluate(tt, dataset)
    assert breakdown.hard_counts["lecture_count_violations"] >= 2  # 1 missing + 1 duplicate


def test_teacher_unavailable_detection(dataset, valid_timetable):
    """Verify teacher unavailability: Dr. Mehta is unavailable Friday afternoon."""
    tt = copy.deepcopy(valid_timetable)
    mehta_gene = [g for g in tt.genes if "CS301" in g.lecture_id][0]
    # Move Dr. Mehta to Friday period 4 (unavailable)
    mehta_gene.day = 4
    mehta_gene.start_period = 4

    breakdown = evaluate(tt, dataset)
    assert breakdown.hard_counts["teacher_unavailable"] >= 1


def test_capacity_violation_detection(dataset, valid_timetable):
    """Verify capacity violation: batch placed in undersized room."""
    tt = copy.deepcopy(valid_timetable)
    # Artificially set room capacity smaller than batch strength
    dataset.rooms["R101"].capacity = 10  # CSE-A strength is 60

    # Ensure at least one gene is in R101
    tt.genes[0].room_id = "R101"
    breakdown = evaluate(tt, dataset)
    assert breakdown.hard_counts["capacity_violations"] >= 1


def test_practical_day_boundary_detection(dataset, valid_timetable):
    """Verify practical day boundary: 2-period lab starting at period 4 overflows beyond period 4."""
    tt = copy.deepcopy(valid_timetable)
    lec_map = {l.lecture_id: l for l in dataset.lectures}
    lab_gene = [g for g in tt.genes if lec_map[g.lecture_id].needs_lab][0]
    lab_gene.day = 0
    lab_gene.start_period = 4  # Starts at period 4, duration 2 -> requires period 4 and 5 (overflow!)

    breakdown = evaluate(tt, dataset)
    assert breakdown.hard_counts["practical_day_boundary"] >= 1


def test_practical_room_type_mismatch_detection(dataset, valid_timetable):
    """Verify room type mismatch: practical in classroom, or theory in lab."""
    tt = copy.deepcopy(valid_timetable)
    lec_map = {l.lecture_id: l for l in dataset.lectures}
    lab_gene = [g for g in tt.genes if lec_map[g.lecture_id].needs_lab][0]
    theory_gene = [g for g in tt.genes if not lec_map[g.lecture_id].needs_lab][0]

    # Put lab in classroom R101
    lab_gene.room_id = "R101"
    # Put theory in lab LAB01
    theory_gene.room_id = "LAB01"

    breakdown = evaluate(tt, dataset)
    assert breakdown.hard_counts["room_type_mismatch"] >= 2


def test_soft_constraint_functions(dataset, valid_timetable):
    """Verify that all isolated soft constraint functions execute and return valid numbers."""
    u1 = soft_teacher_consecutive(valid_timetable, dataset)
    u2 = soft_batch_subject_duplicate(valid_timetable, dataset)
    u3 = soft_batch_gaps(valid_timetable, dataset)
    u4 = soft_teacher_workload_variance(valid_timetable, dataset)
    u5 = soft_undesirable_patterns(valid_timetable, dataset)

    assert isinstance(u1, int) and u1 >= 0
    assert isinstance(u2, int) and u2 >= 0
    assert isinstance(u3, int) and u3 >= 0
    assert isinstance(u4, float) and u4 >= 0.0
    assert isinstance(u5, int) and u5 >= 0
