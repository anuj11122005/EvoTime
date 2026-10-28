"""
test_models.py — Unit Tests for EvoTime Data Models.

Tests that all dataclasses (Subject, Teacher, Room, Batch, Lecture, Gene, Timetable)
initialize correctly with expected fields and types, and that the slot occupancy
helpers accurately calculate single-period and consecutive multi-period practical slots.
"""

import pytest
from evotime.models import (
    Subject,
    Teacher,
    Room,
    Batch,
    Lecture,
    Gene,
    Timetable,
    get_gene_slots,
)


def test_subject_creation():
    """Verify Subject creation for both theory and practical subjects."""
    theory = Subject(
        id="CS301",
        name="Soft Computing",
        type="theory",
        lectures_per_week=3,
        duration=1,
        teacher_id="T01",
    )
    assert theory.id == "CS301"
    assert theory.duration == 1
    assert theory.type == "theory"

    lab = Subject(
        id="CS302L",
        name="AI Lab",
        type="practical",
        lectures_per_week=1,
        duration=2,
        teacher_id="T02",
    )
    assert lab.duration == 2
    assert lab.type == "practical"


def test_teacher_creation():
    """Verify Teacher model with specific available time slots."""
    available = {(0, 0), (0, 1), (1, 3), (2, 4)}
    teacher = Teacher(id="T01", name="Dr. Sharma", available_slots=available)
    assert teacher.id == "T01"
    assert (0, 0) in teacher.available_slots
    assert (3, 0) not in teacher.available_slots


def test_room_creation():
    """Verify Room model for classroom and computer lab."""
    classroom = Room(id="R101", name="Room 101", capacity=60, kind="classroom")
    lab = Room(id="LAB01", name="Computing Lab 1", capacity=35, kind="lab")
    assert classroom.kind == "classroom"
    assert lab.kind == "lab"
    assert classroom.capacity == 60


def test_batch_creation():
    """Verify Batch model with strength and subject list."""
    batch = Batch(
        id="B_CSE_A",
        name="CSE Section A",
        strength=50,
        subject_ids=["CS301", "CS302L"],
    )
    assert batch.strength == 50
    assert len(batch.subject_ids) == 2


def test_lecture_creation():
    """Verify Lecture requirement creation."""
    lecture = Lecture(
        lecture_id="L_CS301_A_1",
        subject_id="CS301",
        teacher_id="T01",
        batch_id="B_CSE_A",
        duration=1,
        needs_lab=False,
    )
    assert lecture.lecture_id == "L_CS301_A_1"
    assert not lecture.needs_lab
    assert lecture.duration == 1


def test_gene_occupied_slots_single_period():
    """Verify that a 1-period theory gene occupies exactly one (day, period) coordinate."""
    # Gene on Tuesday (day 1) at Period 3 (start_period 2)
    gene = Gene(lecture_id="L_CS301_A_1", room_id="R101", day=1, start_period=2)
    slots = gene.get_occupied_slots(duration=1)
    assert slots == [(1, 2)]

    # Test top-level helper function
    assert get_gene_slots(gene, duration=1) == [(1, 2)]


def test_gene_occupied_slots_practical_two_periods():
    """
    Verify that a 2-period practical gene occupies two consecutive periods
    on the same day.
    """
    # Practical on Monday (day 0) starting at Period 2 (start_period 1)
    # Should occupy period 1 and period 2
    lab_gene = Gene(lecture_id="L_CS302L_A_1", room_id="LAB01", day=0, start_period=1)
    slots = lab_gene.get_occupied_slots(duration=2)
    assert slots == [(0, 1), (0, 2)]
    assert get_gene_slots(lab_gene, duration=2) == [(0, 1), (0, 2)]


def test_timetable_chromosome_creation():
    """Verify Timetable container holds list of genes."""
    g1 = Gene(lecture_id="L1", room_id="R101", day=0, start_period=0)
    g2 = Gene(lecture_id="L2", room_id="R102", day=1, start_period=3)
    timetable = Timetable(genes=[g1, g2])
    assert len(timetable.genes) == 2
    assert timetable.genes[0].lecture_id == "L1"
    assert timetable.genes[1].lecture_id == "L2"

