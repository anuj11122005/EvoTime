"""
results.py — Timetable Visualizations, Grids, Reports, and Data Export for EvoTime.

This module provides pure Python functions for transforming timetable chromosomes
and validation telemetry into structured grid representations, styled HTML tables,
comparison DataFrames, and exportable CSV files.

Architectural Rule:
This module contains NO Streamlit calls (`st.*`). Keeping all data transformations
in pure functions allows them to be independently unit tested with pytest.
"""

from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from evotime.config import DAYS, NUM_DAYS, PERIOD_LABELS, PERIODS_PER_DAY
from evotime.fitness import FitnessBreakdown
from evotime.models import Batch, Gene, Lecture, Room, Subject, Teacher, Timetable
from evotime.validator import ValidationReport


# ============================================================================
# Color Palette for Subject Representation
# ============================================================================

SUBJECT_COLORS: Dict[str, str] = {
    "CS301": "#Dbeafe",   # Soft blue (Soft Computing)
    "CS302": "#Dcfce7",   # Soft green (Machine Learning)
    "CS303": "#Fef3c7",   # Soft amber/orange (DBMS)
    "CS304": "#F3e8ff",   # Soft purple (Operating Systems)
    "CS305": "#Ccfae5",   # Soft teal (Computer Networks)
    "CS306": "#Fce7f3",   # Soft pink (Software Engineering)
    "CS307L": "#Fef9c3",  # Soft yellow (DBMS Lab)
    "CS308L": "#E0e7ff",  # Soft indigo (ML Lab)
    "CS399": "#Fee2e2",   # Soft red (Robotics)
}
DEFAULT_SUBJECT_COLOR = "#F3f4f6"
CLASH_COLOR = "#Fee2e2"
EMPTY_COLOR = "#F9fafb"


# ============================================================================
# Timetable Grid Structure
# ============================================================================

@dataclass
class GridCellEntry:
    """Represents a single lecture entry scheduled inside a time slot cell."""
    lecture_id: str
    subject_id: str
    subject_name: str
    teacher_name: str
    room_name: str
    batch_name: str
    duration: int
    needs_lab: bool


@dataclass
class GridCell:
    """Represents a period x day slot coordinate in the timetable grid."""
    day: int
    period: int
    entries: List[GridCellEntry] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.entries) == 0

    @property
    def is_clash(self) -> bool:
        return len(self.entries) > 1

    @property
    def display_text(self) -> str:
        if self.is_empty:
            return "—"
        formatted_items = []
        for e in self.entries:
            lab_badge = " [LAB]" if e.needs_lab else ""
            formatted_items.append(
                f"{e.subject_name}{lab_badge}\n({e.teacher_name} | {e.room_name} | {e.batch_name})"
            )
        if self.is_clash:
            return "⚠️ CLASH ⚠️\n" + "\n---\n".join(formatted_items)
        return formatted_items[0]


@dataclass
class TimetableGrid:
    """
    Two-dimensional (Periods x Days) representation of a weekly schedule.
    Rows = periods (0..4), Columns = days (0..4).
    """
    view: str
    entity_id: str
    entity_name: str
    cells: List[List[GridCell]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the grid cells into a clean pandas DataFrame with period/day labels."""
        data = {day_name: [] for day_name in DAYS}
        for p in range(PERIODS_PER_DAY):
            for d, day_name in enumerate(DAYS):
                data[day_name].append(self.cells[p][d].display_text)
        df = pd.DataFrame(data, index=PERIOD_LABELS)
        df.index.name = "Period"
        return df


def build_grid(
    timetable: Timetable,
    dataset: Any,
    view: str = "batch",
    entity_id: Optional[str] = None,
) -> TimetableGrid:
    """
    Constructs a 5x5 (periods x days) timetable grid for a specific entity.

    Args:
        timetable: The Timetable candidate chromosome.
        dataset: The college dataset (containing subjects, teachers, rooms, batches, lectures).
        view: One of 'batch', 'teacher', or 'room'.
        entity_id: The specific ID of the batch, teacher, or room to filter by.
                   (If None, defaults to the first available entity of that kind).

    Returns:
        TimetableGrid containing GridCell objects for each (period, day) slot.
    """
    if view == "batch":
        if not entity_id or entity_id not in dataset.batches:
            entity_id = list(dataset.batches.keys())[0]
        entity_name = dataset.batches[entity_id].name
    elif view == "teacher":
        if not entity_id or entity_id not in dataset.teachers:
            entity_id = list(dataset.teachers.keys())[0]
        entity_name = dataset.teachers[entity_id].name
    elif view == "room":
        if not entity_id or entity_id not in dataset.rooms:
            entity_id = list(dataset.rooms.keys())[0]
        entity_name = dataset.rooms[entity_id].name
    else:
        raise ValueError(f"Invalid view '{view}'. Must be 'batch', 'teacher', or 'room'.")

    # Initialize empty 5x5 grid of cells
    cells = [
        [GridCell(day=d, period=p, entries=[]) for d in range(NUM_DAYS)]
        for p in range(PERIODS_PER_DAY)
    ]

    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in dataset.lectures}

    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if not lec:
            continue

        # Check whether this gene belongs to the requested entity view
        matches = False
        if view == "batch" and lec.batch_id == entity_id:
            matches = True
        elif view == "teacher" and lec.teacher_id == entity_id:
            matches = True
        elif view == "room" and gene.room_id == entity_id:
            matches = True

        if not matches:
            continue

        sub = dataset.subjects.get(lec.subject_id)
        sub_name = sub.name if sub else lec.subject_id
        t = dataset.teachers.get(lec.teacher_id)
        t_name = t.name if t else lec.teacher_id
        r = dataset.rooms.get(gene.room_id)
        r_name = r.name if r else gene.room_id
        b = dataset.batches.get(lec.batch_id)
        b_name = b.name if b else lec.batch_id

        entry = GridCellEntry(
            lecture_id=lec.lecture_id,
            subject_id=lec.subject_id,
            subject_name=sub_name,
            teacher_name=t_name,
            room_name=r_name,
            batch_name=b_name,
            duration=lec.duration,
            needs_lab=lec.needs_lab,
        )

        # Place the lecture in all occupied periods (2-period practicals occupy period & period + 1)
        for p_offset in range(lec.duration):
            p = gene.start_period + p_offset
            d = gene.day
            if 0 <= d < NUM_DAYS and 0 <= p < PERIODS_PER_DAY:
                cells[p][d].entries.append(entry)

    return TimetableGrid(
        view=view,
        entity_id=entity_id,
        entity_name=entity_name,
        cells=cells,
    )


def grid_to_html(grid: TimetableGrid) -> str:
    """
    Renders the TimetableGrid as a beautifully styled HTML table for Streamlit.
    Color codes subjects, highlights clashes in bold red, and colors empty slots in light grey.
    """
    html = ['<div style="overflow-x:auto;">']
    html.append('<table style="width:100%; border-collapse:collapse; font-family:sans-serif; text-align:center;">')

    # Header Row
    html.append('<thead><tr style="background:#1e293b; color:white;">')
    html.append('<th style="padding:10px; border:1px solid #cbd5e1; width:160px;">Time / Day</th>')
    for day in DAYS:
        html.append(f'<th style="padding:10px; border:1px solid #cbd5e1;">{day}</th>')
    html.append('</tr></thead><tbody>')

    for p in range(PERIODS_PER_DAY):
        html.append('<tr>')
        html.append(
            f'<td style="padding:8px; font-weight:bold; background:#f1f5f9; border:1px solid #cbd5e1; font-size:12px;">'
            f'{PERIOD_LABELS[p]}</td>'
        )

        for d in range(NUM_DAYS):
            cell = grid.cells[p][d]
            if cell.is_empty:
                bg = EMPTY_COLOR
                content = '<span style="color:#94a3b8; font-size:13px;">— Free —</span>'
                border = "1px solid #e2e8f0"
            elif cell.is_clash:
                bg = CLASH_COLOR
                border = "2px solid #ef4444"
                clash_items = []
                for e in cell.entries:
                    clash_items.append(
                        f'<div style="font-size:11px; margin-bottom:4px;">'
                        f'<b>{e.subject_name}</b><br>'
                        f'<span style="color:#64748b;">{e.teacher_name} | {e.room_name} | {e.batch_name}</span>'
                        f'</div>'
                    )
                content = (
                    f'<div style="color:#b91c1c; font-weight:bold; font-size:11px; margin-bottom:4px;">'
                    f'⚠️ CLASH ({len(cell.entries)} classes)</div>'
                    + '<hr style="border:0.5px solid #f87171; margin:4px 0;">'.join(clash_items)
                )
            else:
                entry = cell.entries[0]
                bg = SUBJECT_COLORS.get(entry.subject_id, DEFAULT_SUBJECT_COLOR)
                border = "1px solid #cbd5e1"
                lab_tag = (
                    ' <span style="background:#3b82f6; color:white; font-size:9px; padding:2px 4px; border-radius:3px;">LAB</span>'
                    if entry.needs_lab else ""
                )
                content = (
                    f'<div style="font-size:13px; font-weight:bold; color:#0f172a; margin-bottom:2px;">'
                    f'{entry.subject_name}{lab_tag}</div>'
                    f'<div style="font-size:11px; color:#334155;">{entry.teacher_name}</div>'
                    f'<div style="font-size:11px; color:#475569;"><b>{entry.room_name}</b> ({entry.batch_name})</div>'
                )

            html.append(
                f'<td style="background:{bg}; border:{border}; padding:8px; vertical-align:top; border-radius:4px;">'
                f'{content}</td>'
            )
        html.append('</tr>')

    html.append('</tbody></table></div>')
    return "".join(html)


# ============================================================================
# Violations & Audit Tables
# ============================================================================

def violations_table(report: ValidationReport) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs detailed violation DataFrames from an independent ValidationReport.

    Returns:
        (detailed_df, summary_df):
        - detailed_df: Columns ["Violation Type", "Details"]
        - summary_df: Columns ["Violation Kind", "Count"]
    """
    if not report.hard_violations:
        detailed_df = pd.DataFrame([{"Violation Type": "None", "Details": "All constraints satisfied (0 hard violations)"}])
        summary_df = pd.DataFrame([{"Violation Kind": "Zero Violations", "Count": 0}])
        return detailed_df, summary_df

    records = [{"Violation Type": v.kind, "Details": v.details} for v in report.hard_violations]
    detailed_df = pd.DataFrame(records)

    kind_counts = detailed_df["Violation Type"].value_counts().reset_index()
    kind_counts.columns = ["Violation Kind", "Count"]

    return detailed_df, kind_counts


def history_dataframe(history: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts the generation history list into a typed pandas DataFrame."""
    if not history:
        return pd.DataFrame(columns=["Generation", "Best Penalty", "Average Penalty", "Hard Violations", "Soft Units"])
    df = pd.DataFrame(history)
    df.rename(
        columns={
            "generation": "Generation",
            "best_penalty": "Best Penalty",
            "avg_penalty": "Average Penalty",
            "best_hard": "Hard Violations",
            "best_soft": "Soft Units",
        },
        inplace=True,
    )
    return df


def comparison_table(
    initial_eval: FitnessBreakdown,
    final_eval: FitnessBreakdown,
    initial_report: ValidationReport,
    final_report: ValidationReport,
) -> pd.DataFrame:
    """
    Constructs a side-by-side comparison table between the initial random timetable (Gen 0)
    and the final evolved timetable.
    """
    metrics = [
        ("Independent Validator Verdict", "VALID" if initial_report.is_valid else "FAILED", "VALID" if final_report.is_valid else "FAILED"),
        ("Hard Violations (Auditor Count)", initial_report.hard_count, final_report.hard_count),
        ("Total Penalty Score", f"{initial_eval.penalty:.1f}", f"{final_eval.penalty:.1f}"),
        ("Normalized Fitness (1/(1+P))", f"{initial_eval.fitness_score:.6f}", f"{final_eval.fitness_score:.6f}"),
        ("Teacher Clashes", initial_eval.hard_counts.get("teacher_clashes", 0), final_eval.hard_counts.get("teacher_clashes", 0)),
        ("Room Clashes", initial_eval.hard_counts.get("room_clashes", 0), final_eval.hard_counts.get("room_clashes", 0)),
        ("Batch Double-Bookings", initial_eval.hard_counts.get("batch_clashes", 0), final_eval.hard_counts.get("batch_clashes", 0)),
        ("Teacher Unavailable Slots", initial_eval.hard_counts.get("teacher_unavailable", 0), final_eval.hard_counts.get("teacher_unavailable", 0)),
        ("Room Capacity Exceeded", initial_eval.hard_counts.get("capacity_violations", 0), final_eval.hard_counts.get("capacity_violations", 0)),
        ("Practical Day Overflow", initial_eval.hard_counts.get("practical_day_boundary", 0), final_eval.hard_counts.get("practical_day_boundary", 0)),
        ("Room Kind Mismatch", initial_eval.hard_counts.get("room_type_mismatch", 0), final_eval.hard_counts.get("room_type_mismatch", 0)),
        ("Missing / Duplicate Lectures", initial_eval.hard_counts.get("lecture_count_violations", 0), final_eval.hard_counts.get("lecture_count_violations", 0)),
        ("Soft Penalty Units (Ergonomics)", f"{initial_eval.soft_weighted:.1f}", f"{final_eval.soft_weighted:.1f}"),
    ]

    rows = []
    for name, init_val, fin_val in metrics:
        if isinstance(init_val, (int, float)) and isinstance(fin_val, (int, float)):
            diff = fin_val - init_val
            change_str = f"{diff:+}" if diff != 0 else "0"
        else:
            change_str = "Resolved" if init_val != fin_val else "—"
        rows.append({
            "Metric": name,
            "Initial Timetable (Gen 0)": str(init_val),
            "Final Timetable": str(fin_val),
            "Net Change": change_str,
        })

    return pd.DataFrame(rows)


# ============================================================================
# CSV Data Export
# ============================================================================

def export_csv(
    timetable: Timetable,
    dataset: Any,
    status: str,
    hard_violations_count: int = 0,
) -> Tuple[str, str]:
    """
    Exports the timetable into a structured long-format CSV string and returns a descriptive filename.

    Args:
        timetable: The Timetable instance.
        dataset: The college dataset.
        status: "SUCCESS" or "FAILED".
        hard_violations_count: Number of remaining hard violations.

    Returns:
        (csv_string, filename)
    """
    lecture_map: Dict[str, Lecture] = {l.lecture_id: l for l in dataset.lectures}
    rows = []

    status_tag = (
        "SUCCESS - Valid conflict-free"
        if status == "SUCCESS"
        else f"FAILED - {hard_violations_count} hard violations remaining"
    )

    for gene in timetable.genes:
        lec = lecture_map.get(gene.lecture_id)
        if not lec:
            continue
        sub = dataset.subjects.get(lec.subject_id)
        teacher = dataset.teachers.get(lec.teacher_id)
        room = dataset.rooms.get(gene.room_id)
        batch = dataset.batches.get(lec.batch_id)

        day_name = DAYS[gene.day] if 0 <= gene.day < NUM_DAYS else f"Day {gene.day}"
        start_label = PERIOD_LABELS[gene.start_period] if 0 <= gene.start_period < PERIODS_PER_DAY else f"Period {gene.start_period + 1}"

        rows.append({
            "Lecture ID": lec.lecture_id,
            "Batch": batch.name if batch else lec.batch_id,
            "Subject Code": lec.subject_id,
            "Subject Name": sub.name if sub else lec.subject_id,
            "Subject Type": sub.type if sub else ("practical" if lec.needs_lab else "theory"),
            "Teacher": teacher.name if teacher else lec.teacher_id,
            "Room": room.name if room else gene.room_id,
            "Room Kind": room.kind if room else "unknown",
            "Day": day_name,
            "Start Period": start_label,
            "Duration (Periods)": lec.duration,
            "Timetable Status": status_tag,
        })

    df = pd.DataFrame(rows)
    # Sort logically by Batch, Day, Start Period
    df.sort_values(by=["Batch", "Day", "Start Period"], inplace=True)

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()

    filename = (
        "VALID_timetable_schedule.csv"
        if status == "SUCCESS"
        else f"INVALID_timetable_{hard_violations_count}_violations.csv"
    )

    return csv_str, filename

