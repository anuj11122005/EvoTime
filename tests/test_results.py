"""
test_results.py — Unit Tests for Visualizations, Grids, Reports, and Data Export.

Verifies that:
1. build_grid properly indexes all lectures into (period, day) cells.
2. Multi-period practicals occupy multiple consecutive period rows.
3. Clashes preserve all overlapping lectures and are flagged as clashes.
4. comparison_table accurately maps metrics from evaluate() and validate().
5. export_csv generates valid CSV with correct gene counts and proper VALID/INVALID prefixes.
"""

import copy
import pandas as pd
import pytest
from evotime.fitness import evaluate
from evotime.models import Gene, Timetable
from evotime.results import (
    build_grid,
    comparison_table,
    export_csv,
    history_dataframe,
    violations_table,
)
from evotime.sample_data import get_feasible_dataset
from evotime.validator import validate
from tests.helpers import build_valid_sample_timetable


@pytest.fixture
def dataset():
    return get_feasible_dataset()


@pytest.fixture
def valid_timetable(dataset):
    return build_valid_sample_timetable(dataset)


def test_build_grid_places_lectures_accurately(dataset, valid_timetable):
    """Verify that build_grid maps every lecture for a batch into the correct cell."""
    grid = build_grid(valid_timetable, dataset, view="batch", entity_id="B_CSE_A")
    assert grid.entity_id == "B_CSE_A"
    assert len(grid.cells) == 5  # 5 periods
    assert len(grid.cells[0]) == 5  # 5 days

    # Count total entries across all 25 cells for batch A
    total_placed_periods = sum(len(grid.cells[p][d].entries) for p in range(5) for d in range(5))
    # Batch A has 17 theory (1 period each) + 2 labs (2 periods each) = 21 periods
    assert total_placed_periods == 21


def test_practical_fills_two_consecutive_periods(dataset, valid_timetable):
    """Verify that a 2-period practical appears in both start_period and start_period + 1."""
    lec_map = {l.lecture_id: l for l in dataset.lectures}
    lab_gene = [g for g in valid_timetable.genes if lec_map[g.lecture_id].needs_lab and "B_CSE_A" in g.lecture_id][0]

    grid = build_grid(valid_timetable, dataset, view="batch", entity_id="B_CSE_A")
    d = lab_gene.day
    p0 = lab_gene.start_period
    p1 = p0 + 1

    cell_0 = grid.cells[p0][d]
    cell_1 = grid.cells[p1][d]

    assert any(e.lecture_id == lab_gene.lecture_id for e in cell_0.entries)
    assert any(e.lecture_id == lab_gene.lecture_id for e in cell_1.entries)


def test_clash_cell_flagged_and_retains_all_lectures(dataset, valid_timetable):
    """Verify that when multiple lectures overlap in the same cell, the clash is flagged."""
    tt = copy.deepcopy(valid_timetable)
    csea_genes = [g for g in tt.genes if "B_CSE_A" in g.lecture_id]
    g0, g1 = csea_genes[0], csea_genes[1]
    g1.day = g0.day
    g1.start_period = g0.start_period

    grid = build_grid(tt, dataset, view="batch", entity_id="B_CSE_A")
    clash_cell = grid.cells[g0.start_period][g0.day]

    assert clash_cell.is_clash is True
    assert len(clash_cell.entries) >= 2
    assert "CLASH" in clash_cell.display_text


def test_comparison_table_matches_evaluation_outputs(dataset, valid_timetable):
    """Verify that comparison_table values strictly match evaluate() and validate() outputs."""
    init_eval = evaluate(valid_timetable, dataset)
    init_rep = validate(valid_timetable, dataset)

    comp_df = comparison_table(init_eval, init_eval, init_rep, init_rep)
    assert isinstance(comp_df, pd.DataFrame)
    assert "Metric" in comp_df.columns
    assert "Initial Timetable (Gen 0)" in comp_df.columns

    verdict_row = comp_df[comp_df["Metric"] == "Independent Validator Verdict"].iloc[0]
    assert verdict_row["Initial Timetable (Gen 0)"] == "VALID"


def test_export_csv_row_count_and_naming(dataset, valid_timetable):
    """Verify that export_csv produces a row per lecture and sets correct filename prefixes."""
    # Test valid export
    csv_str, filename = export_csv(valid_timetable, dataset, status="SUCCESS", hard_violations_count=0)
    assert filename.startswith("VALID_")
    df = pd.read_csv(pd.io.common.StringIO(csv_str))
    assert len(df) == len(dataset.lectures)
    assert "SUCCESS - Valid conflict-free" in df["Timetable Status"].values[0]

    # Test failed export
    csv_str_fail, filename_fail = export_csv(valid_timetable, dataset, status="FAILED", hard_violations_count=3)
    assert filename_fail.startswith("INVALID_")
    assert "3_violations" in filename_fail
    df_fail = pd.read_csv(pd.io.common.StringIO(csv_str_fail))
    assert "FAILED - 3 hard violations" in df_fail["Timetable Status"].values[0]

