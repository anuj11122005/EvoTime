"""
fitness.py — Fitness Function and Penalty Evaluation for EvoTime.

In this Genetic Algorithm project, the objective is to MINIMIZE penalties:
    penalty = (HARD_WEIGHT * hard_violations) + (SOFT_WEIGHT * weighted_soft_units)

Hard constraints are non-negotiable physical laws of the college schedule (no two
professors in the same room, no professor teaching two classes simultaneously,
no student batch double-booked, proper room capacities and lab assignments).
Because hard violations represent broken schedules, they carry a heavy penalty
(1000 points per violation) compared to soft preferences (10 points).

Penalty vs. Fitness:
- Penalty (minimization): Lower is better. A conflict-free, ideal schedule has penalty = 0.
- Fitness score (maximization): GAs traditionally maximize fitness in the range (0, 1].
  We derive fitness using:
      fitness = 1.0 / (1.0 + penalty)
  When penalty = 0, fitness is 1.0 (perfect). As penalties increase, fitness approaches 0.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from evotime.config import (
    GAConfig,
    HARD_PENALTY_WEIGHT,
    NUM_DAYS,
    PERIODS_PER_DAY,
    SOFT_PENALTY_WEIGHT,
    WEIGHT_BATCH_GAPS,
    WEIGHT_BATCH_SUBJECT_DUPLICATE,
    WEIGHT_TEACHER_CONSECUTIVE,
    WEIGHT_TEACHER_WORKLOAD_VARIANCE,
    WEIGHT_UNDESIRABLE_PATTERNS,
)
from evotime.models import Batch, Gene, Lecture, Room, Subject, Teacher, Timetable


@dataclass
class FitnessBreakdown:
    """
    Detailed breakdown of all evaluated hard and soft constraint violations.

    Attributes:
        hard_counts: Dictionary mapping each hard constraint kind to its violation count.
        hard_total: Total number of hard violations across all types.
        soft_units: Dictionary mapping each soft constraint kind to its violation units.
        soft_weighted: Weighted sum of soft violation units based on config weights.
        penalty: Total scalar penalty = (1000 * hard_total) + (10 * soft_weighted).
        fitness_score: Normalized maximization score = 1.0 / (1.0 + penalty).
    """
    hard_counts: Dict[str, int]
    hard_total: int
    soft_units: Dict[str, Union[int, float]]
    soft_weighted: float
    penalty: float
    fitness_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert breakdown to a standard serializable dictionary."""
        return asdict(self)


# ============================================================================
# Soft Constraint Evaluation Functions (Isolated Modules)
# ============================================================================

def soft_teacher_consecutive(timetable: Timetable, dataset: Any) -> int:
    """
    Counts instances where a teacher teaches more than 3 consecutive periods in a single day.

    Why this matters:
    Teaching 4 or 5 consecutive hours without a break causes severe fatigue.
    If a teacher is scheduled for periods 0, 1, 2, 3 on Monday (4 consecutive periods),
    this function adds 1 violation unit (4 - 3 = 1).
    """
    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in dataset.lectures}
    # Map teacher -> day -> set of occupied period indices
    schedule: Dict[str, Dict[int, Set[int]]] = {
        t_id: {d: set() for d in range(NUM_DAYS)} for t_id in dataset.teachers
    }

    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if not lec:
            continue
        if 0 <= gene.day < NUM_DAYS:
            for p in range(lec.duration):
                period = gene.start_period + p
                if 0 <= period < PERIODS_PER_DAY:
                    schedule[lec.teacher_id][gene.day].add(period)

    violations = 0
    for t_id, days in schedule.items():
        for d, periods in days.items():
            if len(periods) <= 3:
                continue
            sorted_periods = sorted(periods)
            consecutive_run = 1
            for i in range(1, len(sorted_periods)):
                if sorted_periods[i] == sorted_periods[i - 1] + 1:
                    consecutive_run += 1
                    if consecutive_run > 3:
                        violations += 1
                else:
                    consecutive_run = 1

    return violations


def soft_batch_subject_duplicate(timetable: Timetable, dataset: Any) -> int:
    """
    Counts instances where the same subject is scheduled more than once on the same day for a batch.

    Why this matters:
    Students learn better when subjects are evenly distributed throughout the week
    rather than crammed into a single day (e.g. 2 separate Soft Computing lectures on Monday).
    """
    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in dataset.lectures}
    # Map batch -> day -> subject -> count
    counts: Dict[str, Dict[int, Dict[str, int]]] = {
        b_id: {d: {} for d in range(NUM_DAYS)} for b_id in dataset.batches
    }

    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if not lec:
            continue
        if 0 <= gene.day < NUM_DAYS:
            day_map = counts[lec.batch_id][gene.day]
            day_map[lec.subject_id] = day_map.get(lec.subject_id, 0) + 1

    violations = 0
    for b_id, days in counts.items():
        for d, subject_counts in days.items():
            for sub_id, count in subject_counts.items():
                if count > 1:
                    violations += (count - 1)

    return violations


def soft_batch_gaps(timetable: Timetable, dataset: Any) -> int:
    """
    Counts idle gap periods between classes for each student batch within each day.

    Why this matters:
    If CSE-A has classes at Period 1 (09:00-10:00) and Period 3 (11:15-12:15) with Period 2
    free, students must wait idly on campus. We penalize internal gaps (free periods before
    the first lecture or after the last lecture are not counted as gaps).
    """
    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in dataset.lectures}
    # Map batch -> day -> set of occupied period indices
    occupied: Dict[str, Dict[int, Set[int]]] = {
        b_id: {d: set() for d in range(NUM_DAYS)} for b_id in dataset.batches
    }

    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if not lec:
            continue
        if 0 <= gene.day < NUM_DAYS:
            for p in range(lec.duration):
                period = gene.start_period + p
                if 0 <= period < PERIODS_PER_DAY:
                    occupied[lec.batch_id][gene.day].add(period)

    total_gaps = 0
    for b_id, days in occupied.items():
        for d, periods in days.items():
            if len(periods) >= 2:
                min_p = min(periods)
                max_p = max(periods)
                span = max_p - min_p + 1
                gaps = span - len(periods)
                total_gaps += gaps

    return total_gaps


def soft_teacher_workload_variance(timetable: Timetable, dataset: Any) -> float:
    """
    Measures the unevenness of each teacher's daily workload across the 5 working days.

    Why this matters:
    A teacher who teaches 5 periods on Monday and 0 periods on Tuesday, Wednesday,
    Thursday, and Friday has poor work-life balance. We compute the sum of absolute
    deviations from each teacher's mean daily teaching load.
    """
    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in dataset.lectures}
    # Map teacher -> list of period counts for [Mon, Tue, Wed, Thu, Fri]
    loads: Dict[str, List[int]] = {t_id: [0] * NUM_DAYS for t_id in dataset.teachers}

    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if not lec:
            continue
        if 0 <= gene.day < NUM_DAYS:
            loads[lec.teacher_id][gene.day] += lec.duration

    total_deviation = 0.0
    for t_id, day_counts in loads.items():
        total_periods = sum(day_counts)
        if total_periods > 0:
            mean_load = total_periods / float(NUM_DAYS)
            deviation = sum(abs(cnt - mean_load) for cnt in day_counts)
            total_deviation += deviation

    return round(total_deviation, 2)


def soft_undesirable_patterns(timetable: Timetable, dataset: Any) -> int:
    """
    Penalizes undesirable schedule patterns:
    1. 2-period practicals scheduled at the last period of the day (Period 4 & 5).
    2. A batch having only 1 isolated lecture on an entire day (waste of commuting).
    """
    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in dataset.lectures}
    violations = 0

    # 1. Check last-period practicals
    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if lec and lec.needs_lab:
            # If a practical starts at period 3 (occupying 3 and 4 in 0-indexed 5-period day)
            if gene.start_period == PERIODS_PER_DAY - lec.duration:
                violations += 1

    # 2. Check 1-lecture days for batches
    daily_batch_lectures: Dict[str, Dict[int, int]] = {
        b_id: {d: 0 for d in range(NUM_DAYS)} for b_id in dataset.batches
    }
    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if lec and 0 <= gene.day < NUM_DAYS:
            daily_batch_lectures[lec.batch_id][gene.day] += 1

    for b_id, days in daily_batch_lectures.items():
        for d, count in days.items():
            if count == 1:
                violations += 1

    return violations


# ============================================================================
# Core Fitness Evaluation Function
# ============================================================================

def evaluate(
    timetable: Timetable,
    dataset: Any,
    config: Optional[GAConfig] = None,
) -> FitnessBreakdown:
    """
    Evaluates a candidate Timetable chromosome against the dataset requirements.

    Counting Rules (Strictly Aligned with Validator):
    - Teacher/Room/Batch Clash: For every (resource, day, period) slot occupied by k > 1 genes,
      count k - 1 violations.
    - Teacher Unavailable: 1 violation per gene whose occupied slots fall outside teacher's available slots.
    - Room Capacity: 1 violation per gene where room capacity < batch strength.
    - Practical Day Boundary: 1 violation per practical gene that crosses day boundary.
    - Room Type Mismatch: 1 violation per practical in non-lab room, and 1 per theory in lab room.
    - Lecture Counts: 1 per missing lecture, 1 per duplicate gene (k - 1 for k > 1), 1 per unknown lecture.
    - Out of Range: 1 per gene with invalid day or start_period coordinate.

    Args:
        timetable: The Timetable candidate individual to evaluate.
        dataset: The college dataset (TimetableDataset or object with subjects, teachers, rooms, batches, lectures).
        config: Optional GAConfig instance (falls back to defaults if None).

    Returns:
        FitnessBreakdown containing hard counts, soft units, penalty, and fitness score.
    """
    if config is None:
        config = GAConfig()

    subjects = dataset.subjects
    teachers = dataset.teachers
    rooms = dataset.rooms
    batches = dataset.batches
    lectures = dataset.lectures
    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in lectures}

    # Tracking slot occupancy for clash detection: (resource_id, day, period) -> count
    teacher_occupancy: Dict[Tuple[str, int, int], int] = {}
    room_occupancy: Dict[Tuple[str, int, int], int] = {}
    batch_occupancy: Dict[Tuple[str, int, int], int] = {}

    teacher_unavailable = 0
    capacity_violations = 0
    practical_day_boundary = 0
    room_type_mismatch = 0
    out_of_range = 0
    unknown_lectures = 0

    gene_counts_per_lecture: Dict[str, int] = {l.lecture_id: 0 for l in lectures}

    for gene in timetable.genes:
        # Check unknown lecture
        if gene.lecture_id not in lecture_map:
            unknown_lectures += 1
            continue

        gene_counts_per_lecture[gene.lecture_id] += 1
        lec = lecture_map[gene.lecture_id]
        room = rooms.get(gene.room_id)
        teacher = teachers.get(lec.teacher_id)
        batch = batches.get(lec.batch_id)

        # 1. Out of range check
        if gene.day < 0 or gene.day >= NUM_DAYS or gene.start_period < 0 or gene.start_period >= PERIODS_PER_DAY:
            out_of_range += 1
            continue

        # 2. Practical day boundary check
        if lec.duration > 1 and (gene.start_period + lec.duration > PERIODS_PER_DAY):
            practical_day_boundary += 1

        # 3. Room type mismatch check
        if room:
            if lec.needs_lab and room.kind != "lab":
                room_type_mismatch += 1
            elif not lec.needs_lab and room.kind == "lab":
                room_type_mismatch += 1

            # 4. Room capacity check
            if batch and room.capacity < batch.strength:
                capacity_violations += 1
        else:
            room_type_mismatch += 1

        # Calculate valid in-day slots occupied by this gene
        occupied_slots: List[Tuple[int, int]] = []
        for p in range(lec.duration):
            period = gene.start_period + p
            if period < PERIODS_PER_DAY:
                occupied_slots.append((gene.day, period))

        # 5. Teacher availability check
        if teacher:
            # All occupied slots must fall within teacher's available slots
            if not all(slot in teacher.available_slots for slot in occupied_slots):
                teacher_unavailable += 1

        # Populate occupancy maps for valid slots
        for day, period in occupied_slots:
            t_key = (lec.teacher_id, day, period)
            teacher_occupancy[t_key] = teacher_occupancy.get(t_key, 0) + 1

            if room:
                r_key = (room.id, day, period)
                room_occupancy[r_key] = room_occupancy.get(r_key, 0) + 1

            if batch:
                b_key = (batch.id, day, period)
                batch_occupancy[b_key] = batch_occupancy.get(b_key, 0) + 1

    # 6. Clash violations (for k > 1 genes in a slot, count k - 1 violations)
    teacher_clashes = sum(count - 1 for count in teacher_occupancy.values() if count > 1)
    room_clashes = sum(count - 1 for count in room_occupancy.values() if count > 1)
    batch_clashes = sum(count - 1 for count in batch_occupancy.values() if count > 1)

    # 7. Lecture requirement counts
    missing_lectures = sum(1 for count in gene_counts_per_lecture.values() if count == 0)
    duplicate_lectures = sum(count - 1 for count in gene_counts_per_lecture.values() if count > 1)
    lecture_count_violations = missing_lectures + duplicate_lectures + unknown_lectures

    hard_counts: Dict[str, int] = {
        "teacher_clashes": teacher_clashes,
        "room_clashes": room_clashes,
        "batch_clashes": batch_clashes,
        "teacher_unavailable": teacher_unavailable,
        "capacity_violations": capacity_violations,
        "practical_day_boundary": practical_day_boundary,
        "room_type_mismatch": room_type_mismatch,
        "lecture_count_violations": lecture_count_violations,
        "out_of_range": out_of_range,
    }
    hard_total = sum(hard_counts.values())

    # Evaluate soft constraints
    u_teacher_consec = soft_teacher_consecutive(timetable, dataset)
    u_batch_dup = soft_batch_subject_duplicate(timetable, dataset)
    u_batch_gaps = soft_batch_gaps(timetable, dataset)
    u_teacher_var = soft_teacher_workload_variance(timetable, dataset)
    u_undesirable = soft_undesirable_patterns(timetable, dataset)

    soft_units: Dict[str, Union[int, float]] = {
        "teacher_consecutive": u_teacher_consec,
        "batch_subject_duplicate": u_batch_dup,
        "batch_gaps": u_batch_gaps,
        "teacher_workload_variance": u_teacher_var,
        "undesirable_patterns": u_undesirable,
    }

    soft_weighted = (
        (config.weight_teacher_consecutive * u_teacher_consec)
        + (config.weight_batch_subject_duplicate * u_batch_dup)
        + (config.weight_batch_gaps * u_batch_gaps)
        + (config.weight_teacher_workload_variance * u_teacher_var)
        + (config.weight_undesirable_patterns * u_undesirable)
    )

    penalty = (config.hard_weight * hard_total) + (config.soft_weight * soft_weighted)
    fitness_score = 1.0 / (1.0 + penalty)

    return FitnessBreakdown(
        hard_counts=hard_counts,
        hard_total=hard_total,
        soft_units=soft_units,
        soft_weighted=round(soft_weighted, 2),
        penalty=round(penalty, 2),
        fitness_score=fitness_score,
    )

