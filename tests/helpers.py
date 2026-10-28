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
    Uses greedy constraint-consistent placement to guarantee 0 hard violations.
    """
    assigned = []
    for lec in dataset.lectures:
        batch = dataset.batches[lec.batch_id]
        teacher = dataset.teachers[lec.teacher_id]
        needed_kind = "lab" if lec.needs_lab else "classroom"
        valid_rooms = [
            r for r in dataset.rooms.values()
            if r.kind == needed_kind and r.capacity >= batch.strength
        ]

        placed = False
        for d in range(NUM_DAYS):
            if placed:
                break
            for p in range(PERIODS_PER_DAY - lec.duration + 1):
                if placed:
                    break
                slots = [(d, p + i) for i in range(lec.duration)]
                # Check teacher availability
                if not all(s in teacher.available_slots for s in slots):
                    continue

                for room in valid_rooms:
                    clash = False
                    for prev_gene, prev_lec in assigned:
                        prev_slots = [
                            (prev_gene.day, prev_gene.start_period + i)
                            for i in range(prev_lec.duration)
                        ]
                        overlap = set(slots).intersection(prev_slots)
                        if overlap:
                            if prev_gene.room_id == room.id:
                                clash = True
                                break
                            if prev_lec.teacher_id == lec.teacher_id:
                                clash = True
                                break
                            if prev_lec.batch_id == lec.batch_id:
                                clash = True
                                break
                    if not clash:
                        gene = Gene(
                            lecture_id=lec.lecture_id,
                            room_id=room.id,
                            day=d,
                            start_period=p,
                        )
                        assigned.append((gene, lec))
                        placed = True
                        break

        if not placed:
            raise RuntimeError(f"Could not deterministically place lecture {lec.lecture_id}")

    return Timetable(genes=[g for g, _ in assigned])

