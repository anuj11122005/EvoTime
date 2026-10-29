"""
feasibility.py — Pre-execution Feasibility Checker and Capacity Analysis.

Before launching an expensive Genetic Algorithm run (which could run for hundreds
of generations), this module verifies necessary mathematical conditions for a valid
timetable:
1. Batch Slot Limit: Required periods per batch <= total weekly periods (25 slots).
2. Teacher Availability Limit: Total periods assigned to a teacher <= their available slots.
3. Lab Room Existence: Lab practicals have dedicated lab rooms with sufficient slots.
4. Room Capacity Compatibility: Every batch can physically fit in at least one classroom
   and at least one lab.
5. Global Room Slot Capacity: Total lecture periods <= total room-slots.

Why pre-check feasibility?
A Genetic Algorithm is a heuristic search. If a configuration is mathematically
impossible (e.g., trying to fit 27 hours into a 25-hour week), the GA will run to its
maximum generations, fail, and waste compute. Pre-checking catches fundamental
impossibilities instantly and provides clear, human-readable explanations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from evotime.config import NUM_DAYS, PERIODS_PER_DAY, TOTAL_SLOTS_PER_WEEK
from evotime.models import Batch, Lecture, Room, Subject, Teacher


@dataclass
class FeasibilityReport:
    """
    Detailed report produced by the feasibility pre-checker.

    Attributes:
        ok: True if all necessary feasibility conditions pass, False if impossible.
        errors: List of fatal impossibility error descriptions.
        warnings: List of cautionary notes (e.g. tight utilization).
        summary: Dictionary containing exact numerical arithmetic for batches, teachers, and rooms.
    """
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def check_feasibility(
    subjects: Union[Dict[str, Subject], Any],
    teachers: Union[Dict[str, Teacher], Any] = None,
    rooms: Union[Dict[str, Room], Any] = None,
    batches: Union[Dict[str, Batch], Any] = None,
    lectures: Union[List[Lecture], Any] = None,
) -> FeasibilityReport:
    """
    Performs necessary mathematical pre-checks on the college dataset.
    Can be called with individual entity dicts/lists or unpacked from a TimetableDataset.

    Returns:
        FeasibilityReport indicating whether the problem is mathematically solvable.
    """
    # Support passing a TimetableDataset instance as the first argument
    if hasattr(subjects, "subjects") and hasattr(subjects, "teachers"):
        dataset = subjects
        subjects = dataset.subjects
        teachers = dataset.teachers
        rooms = dataset.rooms
        batches = dataset.batches
        lectures = dataset.lectures

    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}

    max_batch_slots = TOTAL_SLOTS_PER_WEEK  # 25 slots

    # ------------------------------------------------------------------------
    # 1. Batch Slot Limit Check
    # ------------------------------------------------------------------------
    batch_periods: Dict[str, int] = {b_id: 0 for b_id in batches}
    batch_lectures_count: Dict[str, int] = {b_id: 0 for b_id in batches}

    for lec in lectures:
        batch_periods[lec.batch_id] = batch_periods.get(lec.batch_id, 0) + lec.duration
        batch_lectures_count[lec.batch_id] = batch_lectures_count.get(lec.batch_id, 0) + 1

    batch_summary = {}
    for b_id, b in batches.items():
        needed = batch_periods.get(b_id, 0)
        utilization = (needed / max_batch_slots) * 100.0 if max_batch_slots > 0 else 0.0
        batch_summary[b.name] = {
            "batch_id": b_id,
            "required_periods": needed,
            "available_slots": max_batch_slots,
            "utilization_pct": round(utilization, 1),
            "lecture_sessions": batch_lectures_count.get(b_id, 0),
        }
        if needed > max_batch_slots:
            errors.append(
                f"Batch '{b.name}' requires {needed} periods per week, which exceeds the "
                f"maximum weekly slots ({max_batch_slots})."
            )
        elif needed > 0.90 * max_batch_slots:
            warnings.append(
                f"Batch '{b.name}' has high slot utilization ({utilization:.1f}%). "
                f"Timetable search space may be constrained."
            )

    summary["batches"] = batch_summary

    # ------------------------------------------------------------------------
    # 2. Teacher Availability Limit Check
    # ------------------------------------------------------------------------
    teacher_loads: Dict[str, int] = {t_id: 0 for t_id in teachers}
    for lec in lectures:
        teacher_loads[lec.teacher_id] = teacher_loads.get(lec.teacher_id, 0) + lec.duration

    teacher_summary = {}
    for t_id, t in teachers.items():
        assigned = teacher_loads.get(t_id, 0)
        available = len(t.available_slots)
        teacher_summary[t.name] = {
            "teacher_id": t_id,
            "assigned_periods": assigned,
            "available_slots": available,
        }
        if assigned > available:
            errors.append(
                f"Teacher '{t.name}' is assigned {assigned} periods of teaching, but is only "
                f"available for {available} slots in the week."
            )

    summary["teachers"] = teacher_summary

    # ------------------------------------------------------------------------
    # 3. Lab Room Existence and Capacity Check
    # ------------------------------------------------------------------------
    lab_rooms = [r for r in rooms.values() if r.kind == "lab"]
    classroom_rooms = [r for r in rooms.values() if r.kind == "classroom"]

    practical_lectures = [l for l in lectures if l.needs_lab]
    practical_periods = sum(l.duration for l in practical_lectures)
    lab_capacity_slots = len(lab_rooms) * max_batch_slots

    summary["practicals"] = {
        "required_periods": practical_periods,
        "available_lab_slots": lab_capacity_slots,
        "lab_rooms_count": len(lab_rooms),
    }

    if practical_lectures and not lab_rooms:
        errors.append(
            f"Dataset requires {len(practical_lectures)} practical sessions, but 0 lab rooms exist."
        )
    elif practical_periods > lab_capacity_slots:
        errors.append(
            f"Total practical load ({practical_periods} periods) exceeds total available "
            f"lab room capacity ({lab_capacity_slots} slots)."
        )

    # ------------------------------------------------------------------------
    # 4. Room Capacity Compatibility for Each Batch
    # ------------------------------------------------------------------------
    for b_id, b in batches.items():
        can_fit_classroom = any(r.capacity >= b.strength for r in classroom_rooms)
        can_fit_lab = any(r.capacity >= b.strength for r in lab_rooms)

        # Check if batch takes theory
        takes_theory = any(not l.needs_lab and l.batch_id == b_id for l in lectures)
        takes_lab = any(l.needs_lab and l.batch_id == b_id for l in lectures)

        if takes_theory and not can_fit_classroom:
            errors.append(
                f"Batch '{b.name}' (strength {b.strength}) cannot fit into any classroom. "
                f"Largest classroom capacity is {max((r.capacity for r in classroom_rooms), default=0)}."
            )
        if takes_lab and not can_fit_lab:
            errors.append(
                f"Batch '{b.name}' (strength {b.strength}) cannot fit into any lab room. "
                f"Largest lab capacity is {max((r.capacity for r in lab_rooms), default=0)}."
            )

    # ------------------------------------------------------------------------
    # 5. Global Room Slot Capacity Check
    # ------------------------------------------------------------------------
    total_required_periods = sum(l.duration for l in lectures)
    total_room_slots = len(rooms) * max_batch_slots
    summary["global"] = {
        "total_required_periods": total_required_periods,
        "total_room_slots": total_room_slots,
        "room_count": len(rooms),
        "total_lectures": len(lectures),
    }

    if total_required_periods > total_room_slots:
        errors.append(
            f"Total lecture periods ({total_required_periods}) exceed total institution "
            f"room-slots ({total_room_slots})."
        )

    # ------------------------------------------------------------------------
    # 6. Theory Subject Daily Distribution Feasibility (Pigeonhole Check)
    # ------------------------------------------------------------------------
    for b_id, b in batches.items():
        for sub_id in b.subject_ids:
            sub = subjects.get(sub_id)
            if sub and sub.type == "theory" and sub.lectures_per_week > NUM_DAYS:
                errors.append(
                    f"Batch '{b.name}' requires {sub.lectures_per_week} lectures/week for theory subject "
                    f"'{sub.name}', which exceeds the number of working days ({NUM_DAYS}). "
                    f"By the pigeonhole principle, this violates the hard constraint of at most one theory session per day."
                )

    is_feasible = (len(errors) == 0)
    return FeasibilityReport(
        ok=is_feasible,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def format_feasibility_summary(report: FeasibilityReport) -> str:
    """
    Formats the feasibility report arithmetic into a clean, human-readable text summary.
    Suitable for display in CLI output, logs, or Streamlit status panels.
    """
    lines: List[str] = []
    lines.append("=" * 65)
    status_str = "FEASIBLE PRE-CHECK PASSED" if report.ok else "FEASIBILITY PRE-CHECK FAILED"
    lines.append(f"  {status_str}")
    lines.append("=" * 65)

    if "batches" in report.summary:
        lines.append("\n[Batch Slot Utilization (Arithmetic)]")
        for name, data in report.summary["batches"].items():
            used = data["required_periods"]
            total = data["available_slots"]
            pct = data["utilization_pct"]
            sessions = data["lecture_sessions"]
            lines.append(f"  * {name}: {used} of {total} slots used ({pct}%) | {sessions} sessions")

    if "teachers" in report.summary:
        lines.append("\n[Faculty Workload vs. Availability]")
        for name, data in report.summary["teachers"].items():
            assigned = data["assigned_periods"]
            available = data["available_slots"]
            margin = available - assigned
            lines.append(f"  * {name}: {assigned} periods load (available for {available} slots, buffer: +{margin})")

    if "practicals" in report.summary:
        p_info = report.summary["practicals"]
        lines.append(
            f"\n[Labs] Required: {p_info['required_periods']} periods across "
            f"{p_info['lab_rooms_count']} labs (Max capacity: {p_info['available_lab_slots']} slots)"
        )

    if report.warnings:
        lines.append("\n[Warnings]")
        for w in report.warnings:
            lines.append(f"  ! {w}")

    if not report.ok:
        lines.append("\n[Fatal Impossibilities Detected]")
        for err in report.errors:
            lines.append(f"  X {err}")

    lines.append("=" * 65)
    return "\n".join(lines)


def print_feasibility_summary(report: FeasibilityReport) -> None:
    """Directly prints the formatted feasibility summary to stdout."""
    print(format_feasibility_summary(report))

