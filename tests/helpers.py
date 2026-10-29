"""
helpers.py — Test Utilities and Deterministic Solvers for Testing.

Provides helper routines to construct a known, mathematically valid timetable
for the standard feasible college dataset, so unit tests can evaluate zero-violation
baselines and create targeted single-constraint violations.
"""

from typing import Optional
from evotime.config import NUM_DAYS, PERIODS_PER_DAY
from evotime.models import Gene, Timetable
from evotime.sample_data import TimetableDataset


def build_valid_sample_timetable(dataset: TimetableDataset) -> Timetable:
    """
    Deterministically builds a 100% clash-free, valid Timetable for the feasible dataset.
    Uses backtracking constraint-consistent placement to guarantee 0 hard violations,
    including the theory same-day duplicate hard constraint.
    """
    # Sort practicals first as they have duration 2 and are more constrained
    sorted_lectures = sorted(dataset.lectures, key=lambda l: (not l.needs_lab, l.lecture_id))

    def solve(idx, assigned):
        if idx == len(sorted_lectures):
            return assigned
        lec = sorted_lectures[idx]
        batch = dataset.batches[lec.batch_id]
        teacher = dataset.teachers[lec.teacher_id]
        needed_kind = "lab" if lec.needs_lab else "classroom"
        valid_rooms = [
            r for r in dataset.rooms.values()
            if r.kind == needed_kind and r.capacity >= batch.strength
        ]

        for d in range(NUM_DAYS):
            # Theory subjects at most once per day per batch
            if not lec.needs_lab:
                if any(
                    prev_lec.batch_id == lec.batch_id
                    and prev_lec.subject_id == lec.subject_id
                    and prev_gene.day == d
                    for prev_gene, prev_lec in assigned
                ):
                    continue

            for p in range(PERIODS_PER_DAY - lec.duration + 1):
                slots = [(d, p + i) for i in range(lec.duration)]
                if not all(s in teacher.available_slots for s in slots):
                    continue

                for room in valid_rooms:
                    clash = False
                    for prev_gene, prev_lec in assigned:
                        prev_slots = [
                            (prev_gene.day, prev_gene.start_period + i)
                            for i in range(prev_lec.duration)
                        ]
                        if set(slots).intersection(prev_slots):
                            if (
                                prev_gene.room_id == room.id
                                or prev_lec.teacher_id == lec.teacher_id
                                or prev_lec.batch_id == lec.batch_id
                            ):
                                clash = True
                                break
                    if not clash:
                        gene = Gene(
                            lecture_id=lec.lecture_id,
                            room_id=room.id,
                            day=d,
                            start_period=p,
                        )
                        res = solve(idx + 1, assigned + [(gene, lec)])
                        if res is not None:
                            return res
        return None

    assigned = solve(0, [])
    if not assigned:
        raise RuntimeError("Could not deterministically build a valid timetable for the dataset.")

    # Return genes ordered by original lecture order
    lec_order = {lec.lecture_id: idx for idx, lec in enumerate(dataset.lectures)}
    assigned.sort(key=lambda item: lec_order.get(item[0].lecture_id, 0))
    return Timetable(genes=[g for g, _ in assigned])
