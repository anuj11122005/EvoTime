"""
sample_data.py — Benchmark College Datasets for EvoTime.

This module provides two pre-configured college datasets:
1. get_feasible_dataset(): A mathematically feasible, realistic college schedule
   representing 2 batches (CSE-A and CSE-B), 5 teachers, 5 rooms, and 38 lecture
   sessions (42 periods out of 50 total batch slots = 84% utilization).
2. get_infeasible_dataset(): A deliberately impossible dataset used to verify that
   our feasibility checker and Genetic Algorithm properly reject impossible schedules.

Why are datasets separated from algorithm code?
Separating the data ensures our GA remains a general-purpose solver. We can swap
in new departments, teachers, or constraints without altering the core optimizer.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

from evotime.config import NUM_DAYS, PERIODS_PER_DAY
from evotime.models import Batch, Lecture, Room, Subject, Teacher


@dataclass
class TimetableDataset:
    """
    Container bundle holding all raw data required for timetable generation.
    Supports attribute access as well as 5-tuple unpacking:
        subjects, teachers, rooms, batches, lectures = dataset
    """
    name: str
    description: str
    subjects: Dict[str, Subject]
    teachers: Dict[str, Teacher]
    rooms: Dict[str, Room]
    batches: Dict[str, Batch]
    lectures: List[Lecture] = field(default_factory=list)

    def __iter__(self) -> Iterator:
        """Allow tuple unpacking: (subjects, teachers, rooms, batches, lectures)."""
        return iter((self.subjects, self.teachers, self.rooms, self.batches, self.lectures))


def _all_week_slots() -> set[Tuple[int, int]]:
    """Helper to generate all 25 (day, period) coordinate slots in a 5x5 college week."""
    return {(d, p) for d in range(NUM_DAYS) for p in range(PERIODS_PER_DAY)}


def _generate_lectures_for_batches(
    batches: Dict[str, Batch],
    subjects: Dict[str, Subject],
) -> List[Lecture]:
    """
    Generates the master list of individual Lecture requirement objects.

    If Soft Computing has 3 lectures/week for CSE-A, 3 distinct Lecture objects
    are created: L_B_CSE_A_CS301_1, L_B_CSE_A_CS301_2, L_B_CSE_A_CS301_3.
    """
    lectures: List[Lecture] = []

    for batch_id, batch in batches.items():
        for sub_id in batch.subject_ids:
            subject = subjects[sub_id]
            for session_idx in range(1, subject.lectures_per_week + 1):
                lecture_id = f"L_{batch_id}_{sub_id}_{session_idx}"
                lecture = Lecture(
                    lecture_id=lecture_id,
                    subject_id=subject.id,
                    teacher_id=subject.teacher_id,
                    batch_id=batch.id,
                    duration=subject.duration,
                    needs_lab=(subject.type == "practical"),
                )
                lectures.append(lecture)

    return lectures


def get_feasible_dataset() -> TimetableDataset:
    """
    Creates and returns the FEASIBLE college dataset.

    Specifications:
    - 2 Batches:
        * CSE-A (strength 60)
        * CSE-B (strength 55)
    - 6 Theory Subjects per batch (17 periods):
        * Soft Computing (3), Machine Learning (3), DBMS (3),
          Operating Systems (3), Computer Networks (3), Software Engineering (2)
    - 2 Practical Subjects per batch (4 periods):
        * DBMS Lab (1 session of 2 periods)
        * ML Lab (1 session of 2 periods)
    - Total periods per batch: 17 + 4 = 21 periods (out of 25 available slots, 84% utilization)
    - Total lecture requirements: 19 sessions per batch * 2 = 38 lecture objects (42 periods total)
    - 5 Teachers:
        * Dr. Mehta (SC: 6 periods load)
        * Prof. Shah (ML + ML Lab: 10 periods load)
        * Dr. Iyer (DBMS + DBMS Lab: 10 periods load)
        * Prof. Khan (OS: 6 periods load)
        * Dr. Rao (CN + SE: 10 periods load)
        Each has >= 21 available slots (comfortably above workload).
    - 5 Rooms:
        * 3 Classrooms: R101 (cap 70), R102 (cap 70), R103 (cap 60)
        * 2 Labs: LAB01 (cap 60), LAB02 (cap 60)
    """
    all_slots = _all_week_slots()

    # 5 Teachers with realistic Indian names and specific non-overlapping unavailability
    teachers: Dict[str, Teacher] = {
        "T_MEHTA": Teacher(
            id="T_MEHTA",
            name="Dr. Mehta",
            # Unavailable Friday afternoon (research / meeting)
            available_slots=all_slots - {(4, 3), (4, 4)},
        ),
        "T_SHAH": Teacher(
            id="T_SHAH",
            name="Prof. Shah",
            # Unavailable Monday morning and Wednesday late afternoon
            available_slots=all_slots - {(0, 0), (0, 1), (2, 4)},
        ),
        "T_IYER": Teacher(
            id="T_IYER",
            name="Dr. Iyer",
            # Unavailable Tuesday morning and Thursday afternoon
            available_slots=all_slots - {(1, 0), (1, 1), (3, 3), (3, 4)},
        ),
        "T_KHAN": Teacher(
            id="T_KHAN",
            name="Prof. Khan",
            # Unavailable Wednesday morning
            available_slots=all_slots - {(2, 0), (2, 1)},
        ),
        "T_RAO": Teacher(
            id="T_RAO",
            name="Dr. Rao",
            # Unavailable Friday morning and Monday late afternoon
            available_slots=all_slots - {(4, 0), (4, 1), (0, 4)},
        ),
    }

    # 8 Subjects (6 Theory, 2 Practicals)
    subjects: Dict[str, Subject] = {
        "CS301": Subject(
            id="CS301",
            name="Soft Computing",
            type="theory",
            lectures_per_week=3,
            duration=1,
            teacher_id="T_MEHTA",
        ),
        "CS302": Subject(
            id="CS302",
            name="Machine Learning",
            type="theory",
            lectures_per_week=3,
            duration=1,
            teacher_id="T_SHAH",
        ),
        "CS303": Subject(
            id="CS303",
            name="DBMS",
            type="theory",
            lectures_per_week=3,
            duration=1,
            teacher_id="T_IYER",
        ),
        "CS304": Subject(
            id="CS304",
            name="Operating Systems",
            type="theory",
            lectures_per_week=3,
            duration=1,
            teacher_id="T_KHAN",
        ),
        "CS305": Subject(
            id="CS305",
            name="Computer Networks",
            type="theory",
            lectures_per_week=3,
            duration=1,
            teacher_id="T_RAO",
        ),
        "CS306": Subject(
            id="CS306",
            name="Software Engineering",
            type="theory",
            lectures_per_week=2,
            duration=1,
            teacher_id="T_RAO",
        ),
        "CS307L": Subject(
            id="CS307L",
            name="DBMS Lab",
            type="practical",
            lectures_per_week=1,
            duration=2,  # 2 consecutive periods
            teacher_id="T_IYER",
        ),
        "CS308L": Subject(
            id="CS308L",
            name="ML Lab",
            type="practical",
            lectures_per_week=1,
            duration=2,  # 2 consecutive periods
            teacher_id="T_SHAH",
        ),
    }

    # 5 Rooms: 3 Classrooms, 2 Labs
    rooms: Dict[str, Room] = {
        "R101": Room(id="R101", name="Classroom 101", capacity=70, kind="classroom"),
        "R102": Room(id="R102", name="Classroom 102", capacity=70, kind="classroom"),
        "R103": Room(id="R103", name="Classroom 103", capacity=60, kind="classroom"),
        "LAB01": Room(id="LAB01", name="Computing Lab 1", capacity=60, kind="lab"),
        "LAB02": Room(id="LAB02", name="Computing Lab 2", capacity=60, kind="lab"),
    }

    # 2 Batches (use independent list copies for subject_ids)
    batches: Dict[str, Batch] = {
        "B_CSE_A": Batch(
            id="B_CSE_A",
            name="CSE-A (3rd Year)",
            strength=60,
            subject_ids=list(subjects.keys()),
        ),
        "B_CSE_B": Batch(
            id="B_CSE_B",
            name="CSE-B (3rd Year)",
            strength=55,
            subject_ids=list(subjects.keys()),
        ),
    }

    lectures = _generate_lectures_for_batches(batches, subjects)

    return TimetableDataset(
        name="Feasible College Dataset",
        description="Standard 2-batch engineering department scenario with 84% slot utilization.",
        subjects=subjects,
        teachers=teachers,
        rooms=rooms,
        batches=batches,
        lectures=lectures,
    )


def get_infeasible_dataset() -> TimetableDataset:
    """
    Creates and returns an INFEASIBLE college dataset.

    Why this dataset is mathematically impossible to schedule:
    1. Teacher Overload: Dr. Mehta is assigned 6 periods of Soft Computing
       (3 for CSE-A + 3 for CSE-B), but is made available for only 3 slots
       {(0, 0), (0, 1), (0, 2)}. By the Pigeonhole Principle, at least 3 periods
       cannot be scheduled without a teacher clash or an availability violation.
    2. Batch Overload: CSE-A is assigned an extra 6-period course, bringing its
       required periods to 27 periods. Since a 5-day x 5-period week has only
       25 slots, scheduling 27 periods in 25 slots is mathematically impossible.
    """
    dataset = get_feasible_dataset()

    # Modify teacher availability to cause an explicit teacher load impossibility
    overloaded_mehta = Teacher(
        id="T_MEHTA",
        name="Dr. Mehta",
        # Available for only 3 periods in the entire week! (Needs 6 periods)
        available_slots={(0, 0), (0, 1), (0, 2)},
    )
    dataset.teachers["T_MEHTA"] = overloaded_mehta

    # Add an impossible subject load to Batch A (exceeding 25 slots per week)
    extra_subject = Subject(
        id="CS399",
        name="Advanced Robotics",
        type="theory",
        lectures_per_week=6,  # 6 extra periods
        duration=1,
        teacher_id="T_KHAN",
    )
    dataset.subjects["CS399"] = extra_subject
    dataset.batches["B_CSE_A"].subject_ids.append("CS399")

    # Regenerate lecture requirements
    dataset.lectures = _generate_lectures_for_batches(dataset.batches, dataset.subjects)
    dataset.name = "Infeasible College Dataset"
    dataset.description = (
        "Deliberately impossible dataset: Dr. Mehta has 6 periods load but only 3 available slots; "
        "Batch CSE-A requires 27 periods in a 25-period week."
    )

    return dataset
