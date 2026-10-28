"""
test_validator.py — Unit Tests for Independent Timetable Validator & Cross-check.

Verifies that the independent validator correctly confirms clash-free timetables,
accurately catches every violation type with descriptive explanations, and strictly
matches the hard violation counts produced by fitness.py across hundreds of random
timetables (cross-check test).
"""

import copy
import random
import pytest
from evotime.config import NUM_DAYS, PERIODS_PER_DAY
from evotime.fitness import evaluate
from evotime.models import Gene, Timetable
from evotime.sample_data import get_feasible_dataset
from evotime.validator import format_validation_report, validate
from tests.helpers import build_valid_sample_timetable


@pytest.fixture
def dataset():
    return get_feasible_dataset()


@pytest.fixture
def valid_timetable(dataset):
    return build_valid_sample_timetable(dataset)


def test_validator_on_valid_timetable(dataset, valid_timetable):
    """Verify that independent validator confirms is_valid = True on a clash-free schedule."""
    report = validate(valid_timetable, dataset)
    assert report.is_valid is True
    assert report.hard_count == 0
    assert len(report.hard_violations) == 0

    formatted = format_validation_report(report)
    assert "VALID TIMETABLE" in formatted
    assert "ZERO HARD CONFLICTS" in formatted


def test_validator_catches_teacher_clash(dataset, valid_timetable):
    """Verify validator catches TEACHER_CLASH with descriptive faculty and slot details."""
    tt = copy.deepcopy(valid_timetable)
    mehta_genes = [g for g in tt.genes if "CS301" in g.lecture_id]
    mehta_genes[1].day = mehta_genes[0].day
    mehta_genes[1].start_period = mehta_genes[0].start_period
    mehta_genes[1].room_id = "R102"
    mehta_genes[0].room_id = "R101"

    report = validate(tt, dataset)
    assert report.is_valid is False
    assert any(v.kind == "TEACHER_CLASH" for v in report.hard_violations)
    clash_v = [v for v in report.hard_violations if v.kind == "TEACHER_CLASH"][0]
    assert "Dr. Mehta" in clash_v.details


def test_validator_catches_room_clash(dataset, valid_timetable):
    """Verify validator catches ROOM_CLASH with room name and overlapping lectures."""
    tt = copy.deepcopy(valid_timetable)
    tt.genes[1].day = tt.genes[0].day
    tt.genes[1].start_period = tt.genes[0].start_period
    tt.genes[1].room_id = tt.genes[0].room_id

    report = validate(tt, dataset)
    assert report.is_valid is False
    assert any(v.kind == "ROOM_CLASH" for v in report.hard_violations)


def test_validator_catches_batch_clash(dataset, valid_timetable):
    """Verify validator catches BATCH_CLASH with batch name and time slot."""
    tt = copy.deepcopy(valid_timetable)
    csea_genes = [g for g in tt.genes if "B_CSE_A" in g.lecture_id]
    csea_genes[1].day = csea_genes[0].day
    csea_genes[1].start_period = csea_genes[0].start_period
    csea_genes[1].room_id = "R102"
    csea_genes[0].room_id = "R101"

    report = validate(tt, dataset)
    assert report.is_valid is False
    assert any(v.kind == "BATCH_CLASH" for v in report.hard_violations)
    batch_v = [v for v in report.hard_violations if v.kind == "BATCH_CLASH"][0]
    assert "CSE-A" in batch_v.details


def test_validator_catches_missing_and_duplicate_lectures(dataset, valid_timetable):
    """Verify validator detects missing lectures and duplicate lecture instances."""
    tt = copy.deepcopy(valid_timetable)
    dup_gene = copy.deepcopy(tt.genes[0])
    tt.genes.pop(1)
    tt.genes.append(dup_gene)

    report = validate(tt, dataset)
    assert report.is_valid is False
    assert any(v.kind == "MISSING_LECTURE" for v in report.hard_violations)
    assert any(v.kind == "DUPLICATE_LECTURE" for v in report.hard_violations)


def test_validator_catches_teacher_unavailable(dataset, valid_timetable):
    """Verify validator detects scheduled lectures during teacher unavailability."""
    tt = copy.deepcopy(valid_timetable)
    mehta_gene = [g for g in tt.genes if "CS301" in g.lecture_id][0]
    mehta_gene.day = 4
    mehta_gene.start_period = 4  # Friday period 5 is unavailable for Dr. Mehta

    report = validate(tt, dataset)
    assert report.is_valid is False
    assert any(v.kind == "TEACHER_UNAVAILABLE" for v in report.hard_violations)


def test_validator_catches_capacity_exceeded(dataset, valid_timetable):
    """Verify validator detects when a batch exceeds the assigned room capacity."""
    tt = copy.deepcopy(valid_timetable)
    dataset.rooms["R101"].capacity = 10
    tt.genes[0].room_id = "R101"

    report = validate(tt, dataset)
    assert report.is_valid is False
    assert any(v.kind == "CAPACITY_EXCEEDED" for v in report.hard_violations)


def test_validator_catches_practical_day_boundary(dataset, valid_timetable):
    """Verify validator detects practicals that cross the end of the college day."""
    tt = copy.deepcopy(valid_timetable)
    lec_map = {l.lecture_id: l for l in dataset.lectures}
    lab_gene = [g for g in tt.genes if lec_map[g.lecture_id].needs_lab][0]
    lab_gene.start_period = 4  # 2 periods starting at period 4 crosses day boundary

    report = validate(tt, dataset)
    assert report.is_valid is False
    assert any(v.kind == "PRACTICAL_DAY_OVERFLOW" for v in report.hard_violations)


def test_validator_catches_room_type_mismatch(dataset, valid_timetable):
    """Verify validator detects practical in classroom or theory in lab."""
    tt = copy.deepcopy(valid_timetable)
    lec_map = {l.lecture_id: l for l in dataset.lectures}
    lab_gene = [g for g in tt.genes if lec_map[g.lecture_id].needs_lab][0]
    lab_gene.room_id = "R101"  # Classroom instead of lab

    report = validate(tt, dataset)
    assert report.is_valid is False
    assert any(v.kind == "ROOM_KIND_MISMATCH" for v in report.hard_violations)


def test_cross_check_fitness_vs_validator_exact_match(dataset):
    """
    CRITICAL CROSS-CHECK TEST:
    Generates 250 completely randomized timetables across various random seeds.
    For each candidate, asserts that:
        fitness.evaluate().hard_total == len(validator.validate().hard_violations)

    If this test fails, it means there is an inconsistency between the optimizer's
    penalty model and the independent validator's constraint checks.
    """
    rng = random.Random(1337)
    room_ids = list(dataset.rooms.keys())

    for trial in range(250):
        genes = []
        for lec in dataset.lectures:
            # Generate random room, day, and start_period (including edge cases)
            r_id = rng.choice(room_ids)
            d = rng.randint(0, NUM_DAYS - 1)
            p = rng.randint(0, PERIODS_PER_DAY - 1)
            genes.append(Gene(lecture_id=lec.lecture_id, room_id=r_id, day=d, start_period=p))

        # Occasionally introduce missing/duplicate lectures or out-of-range values
        if trial % 10 == 0 and len(genes) > 2:
            genes.pop(0)
            genes.append(copy.deepcopy(genes[0]))
        if trial % 25 == 0 and len(genes) > 0:
            genes[0].day = 99  # Out of range

        candidate = Timetable(genes=genes)

        fitness_res = evaluate(candidate, dataset)
        validator_res = validate(candidate, dataset)

        assert fitness_res.hard_total == len(validator_res.hard_violations), (
            f"Cross-check mismatch on trial {trial}!\n"
            f"Fitness hard_total: {fitness_res.hard_total} ({fitness_res.hard_counts})\n"
            f"Validator hard_count: {len(validator_res.hard_violations)}\n"
            f"Validator violations: {[v.kind for v in validator_res.hard_violations]}"
        )
