"""
test_feasibility.py — Unit Tests for EvoTime Feasibility Pre-checker.

Verifies that the feasibility module accurately validates the feasible college dataset,
correctly rejects impossible configurations, and generates proper arithmetic summaries.
"""

import pytest
from evotime.feasibility import (
    check_feasibility,
    format_feasibility_summary,
    FeasibilityReport,
)
from evotime.models import Batch, Lecture, Room, Subject, Teacher
from evotime.sample_data import get_feasible_dataset, get_infeasible_dataset


def test_feasible_dataset_passes_feasibility():
    """Verify that the standard feasible dataset passes all checks with no errors."""
    dataset = get_feasible_dataset()
    report = check_feasibility(dataset)

    assert report.ok is True
    assert len(report.errors) == 0

    # Check that arithmetic summary is populated
    assert "batches" in report.summary
    assert "CSE-A (3rd Year)" in report.summary["batches"]
    assert report.summary["batches"]["CSE-A (3rd Year)"]["required_periods"] == 21
    assert report.summary["batches"]["CSE-A (3rd Year)"]["available_slots"] == 25
    assert report.summary["batches"]["CSE-A (3rd Year)"]["utilization_pct"] == 84.0

    assert "teachers" in report.summary
    assert "practicals" in report.summary
    assert "global" in report.summary
    assert report.summary["global"]["total_required_periods"] == 42


def test_infeasible_dataset_fails_feasibility():
    """Verify that the infeasible dataset fails with explicit error descriptions."""
    dataset = get_infeasible_dataset()
    report = check_feasibility(dataset)

    assert report.ok is False
    assert len(report.errors) >= 2

    # Check that specific errors for teacher and batch are caught
    errors_text = " ".join(report.errors)
    assert "Dr. Mehta" in errors_text
    assert "assigned 6 periods" in errors_text
    assert "available for 3 slots" in errors_text
    assert "CSE-A" in errors_text
    assert "exceeds the maximum weekly slots" in errors_text


def test_capacity_mismatch_detected():
    """Verify that a batch with strength exceeding all room capacities is flagged."""
    dataset = get_feasible_dataset()
    # Artificially swell batch size beyond room limits
    dataset.batches["B_CSE_A"].strength = 999

    report = check_feasibility(dataset)
    assert report.ok is False
    assert any("cannot fit into any classroom" in err for err in report.errors)


def test_missing_labs_for_practicals_detected():
    """Verify that requiring practicals without any lab room fails."""
    dataset = get_feasible_dataset()
    # Convert all rooms to classrooms
    for room in dataset.rooms.values():
        room.kind = "classroom"

    report = check_feasibility(dataset)
    assert report.ok is False
    assert any("0 lab rooms exist" in err for err in report.errors)


def test_format_feasibility_summary_output():
    """Verify that format_feasibility_summary returns a formatted text string."""
    dataset = get_feasible_dataset()
    report = check_feasibility(dataset)
    summary_text = format_feasibility_summary(report)

    assert "FEASIBLE PRE-CHECK PASSED" in summary_text
    assert "Batch Slot Utilization" in summary_text
    assert "CSE-A (3rd Year): 21 of 25 slots used (84.0%)" in summary_text
    assert "Faculty Workload vs. Availability" in summary_text

