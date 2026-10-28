"""
models.py — Data Models and Representations for EvoTime.

This module defines all core entities for college timetable scheduling using
Python standard dataclasses. It defines the structure for subjects, teachers,
rooms, student batches, individual lecture requirements, and the fundamental
building blocks of the Genetic Algorithm: Gene and Timetable (Chromosome).

Why dataclasses?
Dataclasses provide automatic __init__, __repr__, and equality checking without
boilerplate code. They keep models lightweight, typed, and clean for viva explanation.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple


@dataclass
class Subject:
    """
    Represents an academic subject taught in a semester.

    Example:
        Subject(
            id="CS301",
            name="Soft Computing",
            type="theory",          # "theory" or "practical"
            lectures_per_week=3,    # 3 sessions needed per week
            duration=1,             # 1 period per theory session
            teacher_id="T01"        # Dr. Sharma teaches this
        )
    """
    id: str
    name: str
    type: str               # "theory" or "practical"
    lectures_per_week: int
    duration: int           # 1 period for theory, 2 periods for lab practical
    teacher_id: str


@dataclass
class Teacher:
    """
    Represents a faculty member and their availability.

    Faculty members might not be on campus every period. Their availability
    is modeled as a set of (day, period) coordinate tuples where:
        day in 0..4 (Mon..Fri)
        period in 0..4 (Period 1..5)

    Example:
        Teacher(
            id="T01",
            name="Dr. Sharma",
            available_slots={(0, 0), (0, 1), (1, 2), ...}
        )
    """
    id: str
    name: str
    available_slots: Set[Tuple[int, int]] = field(default_factory=set)


@dataclass
class Room:
    """
    Represents a college room where classes take place.

    Rooms have a seating capacity and a specific kind ("classroom" for theory,
    "lab" for computer/hardware labs with equipment).

    Example:
        Room(
            id="R101",
            name="Lecture Hall 101",
            capacity=60,
            kind="classroom"        # "classroom" or "lab"
        )
    """
    id: str
    name: str
    capacity: int
    kind: str               # "classroom" or "lab"


@dataclass
class Batch:
    """
    Represents a class or section of students taking a fixed set of subjects.

    Example:
        Batch(
            id="B_CSE_A",
            name="CSE Section A (3rd Year)",
            strength=55,            # 55 students
            subject_ids=["CS301", "CS302", "CS303L"]
        )
    """
    id: str
    name: str
    strength: int
    subject_ids: List[str] = field(default_factory=list)


@dataclass
class Lecture:
    """
    Represents a single required lecture session that MUST be scheduled.

    If Soft Computing has 3 lectures per week for Batch A, we generate 3 distinct
    Lecture requirement objects: L_CS301_A_1, L_CS301_A_2, L_CS301_A_3.
    These requirements are immutable; the Genetic Algorithm never alters them.

    Example:
        Lecture(
            lecture_id="L_CS301_A_1",
            subject_id="CS301",
            teacher_id="T01",
            batch_id="B_CSE_A",
            duration=1,
            needs_lab=False
        )
    """
    lecture_id: str
    subject_id: str
    teacher_id: str
    batch_id: str
    duration: int
    needs_lab: bool


@dataclass
class Gene:
    """
    Represents the variable assignment (the GENE in our Genetic Algorithm)
    for a specific lecture requirement.

    What is fixed:
        lecture_id (which lecture requirement this gene fulfills).
    What is evolved:
        room_id, day (0..4), start_period (0..4).

    Example:
        Gene(
            lecture_id="L_CS301_A_1",
            room_id="R101",
            day=0,              # Monday
            start_period=1      # Period 2 (10:00 - 11:00)
        )
    """
    lecture_id: str
    room_id: str
    day: int
    start_period: int

    def get_occupied_slots(self, duration: int = 1) -> List[Tuple[int, int]]:
        """
        Returns all (day, period) time slot coordinates occupied by this gene.

        For a 1-period theory lecture starting at period 1:
            Returns [(day, 1)]
        For a 2-period practical starting at period 2:
            Returns [(day, 2), (day, 3)]

        Args:
            duration: Number of consecutive periods (default 1).

        Returns:
            List of (day, period) coordinate tuples.
        """
        return [(self.day, self.start_period + p) for p in range(duration)]


def get_gene_slots(gene: Gene, duration: int) -> List[Tuple[int, int]]:
    """
    Helper function to get all (day, period) slots occupied by a gene given its duration.

    Args:
        gene: The Gene instance.
        duration: Duration in periods (e.g. 1 for theory, 2 for practical).

    Returns:
        List of (day, period) tuples.
    """
    return gene.get_occupied_slots(duration=duration)


@dataclass
class Timetable:
    """
    Represents one complete candidate weekly timetable (the CHROMOSOME in the GA).

    A timetable is a collection of Genes, with exactly one Gene per required Lecture.
    The order of genes is index-aligned with the master list of lectures so that
    crossover and mutation can cleanly operate gene-by-gene.

    Example:
        Timetable(genes=[gene1, gene2, ..., geneN])
    """
    genes: List[Gene] = field(default_factory=list)

