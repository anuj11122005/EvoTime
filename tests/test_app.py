"""
test_app.py — Integration Tests for Streamlit Dashboard using AppTest.

Verifies that:
1. The Streamlit app starts and renders without exceptions.
2. Generating a timetable updates session_state and displays appropriate validation banners.
3. Infeasible dataset disables the Generate button unless 'force' is checked.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_renders_initially_without_exceptions():
    """Verify that app.py loads cleanly, renders tabs, and throws no exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception
    assert len(at.tabs) == 7
    # Verify title is displayed
    assert any("EvoTime" in str(title.value) for title in at.title)


def test_app_infeasible_disables_generate_button():
    """Verify that selecting the infeasible dataset disables the Generate button until forced and displays errors."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    # Switch to Infeasible Dataset
    at.selectbox[0].select("Infeasible College Dataset (Impossible)").run()
    assert not at.exception

    # Button should be disabled
    generate_btn = at.button[0]
    assert generate_btn.disabled is True

    # Check that pre-check problems are visibly displayed in markdown / error widgets
    error_texts = [str(e.value) for e in at.error]
    markdown_texts = [str(m.value) for m in at.markdown]
    combined_texts = " ".join(error_texts + markdown_texts)
    assert "27 periods" in combined_texts or "exceeds" in combined_texts
    assert "Dr. Mehta" in combined_texts or "available for 3 slots" in combined_texts

    # Check the "Run anyway (force)" checkbox
    at.checkbox[0].check().run()
    assert not at.exception

    # Button should now be enabled
    assert at.button[0].disabled is False


def test_app_generate_timetable_execution():
    """
    Verify that clicking 'Generate Timetable' runs the GA,
    stores results in session_state, and renders without exceptions.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()

    # Set very small pop and generations for quick test execution
    at.slider[0].set_value(10).run()  # pop size
    at.slider[1].set_value(10).run()  # max generations

    # Click Generate Timetable
    at.button[0].click().run()
    assert not at.exception

    # Verify session_state contains result
    assert "result" in at.session_state
    res = at.session_state["result"]
    assert res is not None

    # Verify that the banner matches the validator verdict
    if res.validation_report.is_valid:
        assert any("VALID TIMETABLE" in str(s.value) for s in at.success)
    else:
        assert any("FAILED" in str(e.value) for e in at.error)

