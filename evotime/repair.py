"""
repair.py — Structural Repair Mechanism for EvoTime Candidate Timetables.

During Genetic Algorithm evolution, crossover and mutation can produce invalid
structures:
- A practical lecture (2 consecutive periods) might be placed starting at Period 5,
  spilling over past the end of the college day.
- A day or period index might be nudged out of valid bounds.
- A lab practical might be assigned to a standard lecture hall, or a theory class
  might be placed in a specialized computer lab.
- A batch might be placed in a room whose capacity is strictly smaller than the batch size.

What this module does:
`repair_timetable()` restores structural validity to candidate chromosomes:
1. Clamps `day` within [0, NUM_DAYS - 1].
2. Clamps `start_period` within [0, PERIODS_PER_DAY - lecture.duration] so practicals
   never cross the day boundary.
3. Enforces room kind (theory in classroom, practical in lab).
4. Enforces room capacity (switches to a room with capacity >= batch strength).

What this module DOES NOT do:
Repair does NOT solve teacher clashes, room clashes, batch double-booking, or teacher
unavailability. The Genetic Algorithm itself is responsible for evolving clash-free
schedules; repair merely guarantees that every gene is structurally and dimensionally sound.
Repair NEVER alters `lecture_id` and NEVER adds or deletes genes.
"""

import random
from typing import Any, Dict, List, Optional

from evotime.config import NUM_DAYS, PERIODS_PER_DAY
from evotime.models import Batch, Gene, Lecture, Room, Timetable


def repair_timetable(
    timetable: Timetable,
    dataset: Any,
    rng: Optional[random.Random] = None,
) -> Timetable:
    """
    Repairs structural invalidities in-place for every gene in the timetable.

    Args:
        timetable: The Timetable chromosome to inspect and repair.
        dataset: The college dataset (containing subjects, rooms, batches, lectures).
        rng: Optional seeded random.Random instance for deterministic choices.

    Returns:
        The repaired Timetable (same object modified in-place for efficiency).
    """
    if rng is None:
        rng = random.Random(42)

    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in dataset.lectures}
    rooms_by_kind: Dict[str, List[Room]] = {
        "classroom": [r for r in dataset.rooms.values() if r.kind == "classroom"],
        "lab": [r for r in dataset.rooms.values() if r.kind == "lab"],
    }

    # Track which rooms are occupied at (day, period) to prefer free rooms on reassignment
    occupied_room_slots = set()
    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if lec:
            for p in range(lec.duration):
                occupied_room_slots.add((gene.room_id, gene.day, gene.start_period + p))

    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if not lec:
            continue

        # --------------------------------------------------------------------
        # 1. Clamp Day and Start Period (Prevents Day-Boundary Spills)
        # --------------------------------------------------------------------
        # Ensure day is in [0, 4]
        gene.day = max(0, min(NUM_DAYS - 1, gene.day))

        # Ensure start_period allows full duration without spilling past period 4
        max_start = PERIODS_PER_DAY - lec.duration
        gene.start_period = max(0, min(max_start, gene.start_period))

        # --------------------------------------------------------------------
        # 2. Enforce Room Kind and Batch Capacity
        # --------------------------------------------------------------------
        needed_kind = "lab" if lec.needs_lab else "classroom"
        batch = dataset.batches.get(lec.batch_id)
        batch_strength = batch.strength if batch else 0

        current_room = dataset.rooms.get(gene.room_id)
        room_is_valid = (
            current_room is not None
            and current_room.kind == needed_kind
            and current_room.capacity >= batch_strength
        )

        if not room_is_valid:
            # Find all compatible rooms of the required kind that can accommodate the batch
            candidate_rooms = [
                r for r in rooms_by_kind.get(needed_kind, [])
                if r.capacity >= batch_strength
            ]

            if candidate_rooms:
                # Prefer a room that is not currently double-booked in this time slot
                free_candidates = []
                for cr in candidate_rooms:
                    is_free = all(
                        (cr.id, gene.day, gene.start_period + p) not in occupied_room_slots
                        for p in range(lec.duration)
                    )
                    if is_free:
                        free_candidates.append(cr)

                chosen_room = rng.choice(free_candidates) if free_candidates else rng.choice(candidate_rooms)
                gene.room_id = chosen_room.id

    return timetable

