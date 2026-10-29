"""
validator.py — Independent Timetable Verification Module for EvoTime.

CRITICAL ARCHITECTURAL PRINCIPLE:
This module is strictly INDEPENDENT of fitness.py. It does NOT import fitness.py.
It re-evaluates candidate timetables directly from first principles using its own
independent slot-occupancy tracking.

Why is an independent validator required?
In AI and heuristic optimization, a bug in the fitness function could falsely lead the
algorithm to believe it found a valid schedule when it actually violated constraints.
The validator acts as an impartial auditor: neither the UI nor the CLI is permitted to
declare a timetable "VALID" or "conflict-free" unless this independent validator confirms
ZERO hard violations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from evotime.config import NUM_DAYS, PERIODS_PER_DAY
from evotime.models import Batch, Gene, Lecture, Room, Subject, Teacher, Timetable


@dataclass
class Violation:
    """
    Represents an individual constraint violation with descriptive human-readable details.

    Attributes:
        kind: Category identifier (e.g., "TEACHER_CLASH", "ROOM_CLASH", "TEACHER_UNAVAILABLE").
        details: Specific explanation naming the faculty, room, batch, time slot, and lectures involved.
    """
    kind: str
    details: str


@dataclass
class ValidationReport:
    """
    Structured outcome of the independent timetable verification.

    Attributes:
        is_valid: True ONLY if zero hard constraint violations exist.
        hard_violations: List of specific Violation objects detected.
        hard_count: Total count of hard violations.
        soft_summary: Informational summary of soft preference metrics.
    """
    is_valid: bool
    hard_violations: List[Violation] = field(default_factory=list)
    hard_count: int = 0
    soft_summary: Dict[str, Any] = field(default_factory=dict)


def validate(
    timetable: Timetable,
    dataset: Any,
) -> ValidationReport:
    """
    Independently inspects every gene in the timetable against the dataset specifications.

    Counting Rules (Strictly Aligned with Fitness Evaluation):
    - Teacher/Room/Batch Clash: For every (resource, day, period) slot occupied by k > 1 genes,
      exactly k - 1 violations are logged.
    - Teacher Unavailable: 1 violation per gene whose occupied slots fall outside teacher's available slots.
    - Room Capacity: 1 violation per gene where room capacity < batch strength.
    - Practical Day Boundary: 1 violation per practical gene that crosses the day boundary.
    - Room Type Mismatch: 1 violation per practical in a non-lab room, and 1 per theory in a lab room.
    - Lecture Counts: 1 per missing required lecture, 1 per duplicate gene (k - 1 for k > 1),
      and 1 per unknown lecture ID.
    - Out of Range: 1 per gene with invalid day or period index.

    Args:
        timetable: The Timetable candidate chromosome to verify.
        dataset: The college dataset (TimetableDataset or object with subjects, teachers, rooms, batches, lectures).

    Returns:
        ValidationReport containing is_valid, list of Violation objects, and soft metric summary.
    """
    subjects: Dict[str, Subject] = dataset.subjects
    teachers: Dict[str, Teacher] = dataset.teachers
    rooms: Dict[str, Room] = dataset.rooms
    batches: Dict[str, Batch] = dataset.batches
    lectures: List[Lecture] = dataset.lectures
    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in lectures}

    violations: List[Violation] = []

    # Independent slot tracking maps: (resource_id, day, period) -> list of Gene objects
    teacher_slots: Dict[Tuple[str, int, int], List[Gene]] = {}
    room_slots: Dict[Tuple[str, int, int], List[Gene]] = {}
    batch_slots: Dict[Tuple[str, int, int], List[Gene]] = {}
    batch_theory_slots: Dict[Tuple[str, int, str], List[Gene]] = {}

    # Lecture occurrence counts
    lecture_genes: Dict[str, List[Gene]] = {l.lecture_id: [] for l in lectures}

    # ------------------------------------------------------------------------
    # 1. Per-Gene Constraint Verification
    # ------------------------------------------------------------------------
    for gene in timetable.genes:
        # Check unknown lecture ID
        if gene.lecture_id not in lecture_map:
            violations.append(
                Violation(
                    kind="UNKNOWN_LECTURE",
                    details=f"Gene references unknown lecture_id '{gene.lecture_id}' not found in curriculum requirements.",
                )
            )
            continue

        lecture_genes[gene.lecture_id].append(gene)
        lec = lecture_map[gene.lecture_id]
        room = rooms.get(gene.room_id)
        teacher = teachers.get(lec.teacher_id)
        batch = batches.get(lec.batch_id)

        # A. Check Out of Range Coordinates
        if gene.day < 0 or gene.day >= NUM_DAYS or gene.start_period < 0 or gene.start_period >= PERIODS_PER_DAY:
            violations.append(
                Violation(
                    kind="OUT_OF_RANGE",
                    details=f"Lecture '{lec.lecture_id}' assigned invalid coordinates (day={gene.day}, start_period={gene.start_period}).",
                )
            )
            continue

        # B. Check Practical Day Overflow (consecutive periods crossing day end)
        if lec.duration > 1 and (gene.start_period + lec.duration > PERIODS_PER_DAY):
            violations.append(
                Violation(
                    kind="PRACTICAL_DAY_OVERFLOW",
                    details=(
                        f"Practical lecture '{lec.lecture_id}' (duration {lec.duration}) starts at period {gene.start_period} "
                        f"on day {gene.day}, overflowing beyond the maximum daily period ({PERIODS_PER_DAY - 1})."
                    ),
                )
            )

        # C. Check Room Kind Mismatch
        if room:
            if lec.needs_lab and room.kind != "lab":
                violations.append(
                    Violation(
                        kind="ROOM_KIND_MISMATCH",
                        details=f"Practical lab lecture '{lec.lecture_id}' placed in non-lab room '{room.name}' (kind='{room.kind}').",
                    )
                )
            elif not lec.needs_lab and room.kind == "lab":
                violations.append(
                    Violation(
                        kind="ROOM_KIND_MISMATCH",
                        details=f"Theory lecture '{lec.lecture_id}' placed in lab room '{room.name}' instead of a classroom.",
                    )
                )

            # D. Check Room Capacity
            if batch and room.capacity < batch.strength:
                violations.append(
                    Violation(
                        kind="CAPACITY_EXCEEDED",
                        details=(
                            f"Room '{room.name}' capacity ({room.capacity}) is smaller than batch '{batch.name}' "
                            f"strength ({batch.strength}) for lecture '{lec.lecture_id}'."
                        ),
                    )
                )
        else:
            violations.append(
                Violation(
                    kind="ROOM_NOT_FOUND",
                    details=f"Lecture '{lec.lecture_id}' assigned non-existent room_id '{gene.room_id}'.",
                )
            )

        # Determine valid in-day periods occupied
        valid_slots: List[Tuple[int, int]] = []
        for p in range(lec.duration):
            period = gene.start_period + p
            if period < PERIODS_PER_DAY:
                valid_slots.append((gene.day, period))

        # E. Check Teacher Availability
        if teacher:
            if not all(slot in teacher.available_slots for slot in valid_slots):
                unavailable_slots = [s for s in valid_slots if s not in teacher.available_slots]
                violations.append(
                    Violation(
                        kind="TEACHER_UNAVAILABLE",
                        details=(
                            f"Teacher '{teacher.name}' is scheduled during unavailable slot(s) {unavailable_slots} "
                            f"for lecture '{lec.lecture_id}'."
                        ),
                    )
                )

        # Record same-day theory subject slots (practicals exempt)
        sub = subjects.get(lec.subject_id)
        if sub and sub.type == "theory" and 0 <= gene.day < NUM_DAYS:
            batch_theory_slots.setdefault((lec.batch_id, gene.day, lec.subject_id), []).append(gene)

        # Record occupancy for clash detection
        for d, p in valid_slots:
            t_key = (lec.teacher_id, d, p)
            teacher_slots.setdefault(t_key, []).append(gene)

            if room:
                r_key = (room.id, d, p)
                room_slots.setdefault(r_key, []).append(gene)

            if batch:
                b_key = (batch.id, d, p)
                batch_slots.setdefault(b_key, []).append(gene)

    # ------------------------------------------------------------------------
    # 2. Resource Clash Verification (Teacher, Room, Batch Overlaps)
    # ------------------------------------------------------------------------
    # Teacher Clashes
    for (t_id, d, p), genes_list in teacher_slots.items():
        if len(genes_list) > 1:
            t_name = teachers[t_id].name if t_id in teachers else t_id
            clashing_ids = [g.lecture_id for g in genes_list]
            for _ in range(len(genes_list) - 1):
                violations.append(
                    Violation(
                        kind="TEACHER_CLASH",
                        details=f"Teacher '{t_name}' double-booked on day {d}, period {p} across lectures: {clashing_ids}.",
                    )
                )

    # Room Clashes
    for (r_id, d, p), genes_list in room_slots.items():
        if len(genes_list) > 1:
            r_name = rooms[r_id].name if r_id in rooms else r_id
            clashing_ids = [g.lecture_id for g in genes_list]
            for _ in range(len(genes_list) - 1):
                violations.append(
                    Violation(
                        kind="ROOM_CLASH",
                        details=f"Room '{r_name}' double-booked on day {d}, period {p} across lectures: {clashing_ids}.",
                    )
                )

    # Batch Clashes
    for (b_id, d, p), genes_list in batch_slots.items():
        if len(genes_list) > 1:
            b_name = batches[b_id].name if b_id in batches else b_id
            clashing_ids = [g.lecture_id for g in genes_list]
            for _ in range(len(genes_list) - 1):
                violations.append(
                    Violation(
                        kind="BATCH_CLASH",
                        details=f"Batch '{b_name}' double-booked on day {d}, period {p} across lectures: {clashing_ids}.",
                    )
                )

    # ------------------------------------------------------------------------
    # 3. Lecture Requirement Completeness (Missing & Duplicate)
    # ------------------------------------------------------------------------
    for lec_id, assigned_genes in lecture_genes.items():
        if len(assigned_genes) == 0:
            violations.append(
                Violation(
                    kind="MISSING_LECTURE",
                    details=f"Required lecture '{lec_id}' is completely missing from the timetable.",
                )
            )
        elif len(assigned_genes) > 1:
            for _ in range(len(assigned_genes) - 1):
                violations.append(
                    Violation(
                        kind="DUPLICATE_LECTURE",
                        details=f"Lecture '{lec_id}' appears {len(assigned_genes)} times in the timetable (duplicates detected).",
                    )
                )

    # ------------------------------------------------------------------------
    # 4. Same-Day Theory Duplicates (Batch Hard Constraint)
    # ------------------------------------------------------------------------
    for (b_id, d, sub_id), genes_list in batch_theory_slots.items():
        if len(genes_list) > 1:
            b_name = batches[b_id].name if b_id in batches else b_id
            sub = subjects.get(sub_id)
            sub_name = sub.name if sub else sub_id
            clashing_ids = [g.lecture_id for g in genes_list]
            for _ in range(len(genes_list) - 1):
                violations.append(
                    Violation(
                        kind="THEORY_SAME_DAY_DUPLICATE",
                        details=(
                            f"Theory subject '{sub_name}' ({sub_id}) scheduled {len(genes_list)} times on day {d} "
                            f"for batch '{b_name}' across lectures: {clashing_ids}."
                        ),
                    )
                )

    # ------------------------------------------------------------------------
    # 5. Informational Soft Summary (Auditing Quality)
    # ------------------------------------------------------------------------
    # Calculate informative metrics without importing fitness module
    soft_summary: Dict[str, Any] = {
        "scheduled_genes": len(timetable.genes),
        "required_lectures": len(lectures),
        "hard_violations_count": len(violations),
    }

    is_valid = (len(violations) == 0)

    return ValidationReport(
        is_valid=is_valid,
        hard_violations=violations,
        hard_count=len(violations),
        soft_summary=soft_summary,
    )


def format_validation_report(report: ValidationReport) -> str:
    """Formats the ValidationReport into a clean text block."""
    lines: List[str] = []
    lines.append("=" * 65)
    status_str = "VALID TIMETABLE (ZERO HARD CONFLICTS)" if report.is_valid else "INVALID TIMETABLE (VIOLATIONS DETECTED)"
    lines.append(f"  STATUS: {status_str}")
    lines.append(f"  Total Hard Violations: {report.hard_count}")
    lines.append("=" * 65)

    if report.hard_violations:
        lines.append("\n[Violations Detected by Independent Validator]")
        for idx, v in enumerate(report.hard_violations, 1):
            lines.append(f"  {idx}. [{v.kind}] {v.details}")
    else:
        lines.append("\n  All constraints satisfied: no teacher, room, or batch clashes;")
        lines.append("  all faculty availability respected; all room capacities valid;")
        lines.append("  all practicals placed in labs; all lectures present exactly once.")

    lines.append("=" * 65)
    return "\n".join(lines)

