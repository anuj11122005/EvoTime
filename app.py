"""
app.py — Interactive Streamlit Dashboard for EvoTime.

Features:
- Sidebar controls for dataset selection, GA hyperparameters, and random seed.
- Feasibility pre-check guard: disables execution on impossible datasets unless forced.
- 7 Dedicated Tabs:
  1. Input Data: Curricula, faculty availability, room capacity, and slot arithmetic.
  2. Initial Timetable: Visual inspection of raw Generation 0 random schedule and clashes.
  3. Final Timetable: Color-coded weekly timetable grid (filterable by Batch, Teacher, Room).
  4. Validation Report: Independent validator audit report and soft preference breakdown.
  5. Optimization Graph: Matplotlib convergence curves for penalty and hard clashes.
  6. Comparison: Side-by-side Generation 0 vs Final metrics.
  7. Export: CSV download with explicit validity markers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Any

import streamlit as st
import pandas as pd

from evotime.config import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_ELITISM_COUNT,
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_MUTATION_RATE,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TOURNAMENT_K,
    GAConfig,
)
from evotime.feasibility import check_feasibility, format_feasibility_summary
from evotime.ga import run_ga
from evotime.results import (
    build_grid,
    comparison_table,
    export_csv,
    grid_to_html,
    history_dataframe,
    violations_table,
)
from evotime.sample_data import get_feasible_dataset, get_infeasible_dataset
from evotime.validator import validate


# Configure Streamlit page layout
st.set_page_config(
    page_title="EvoTime — Timetable Generator",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Sidebar: Dataset Selection & Hyperparameters
# ============================================================================

st.sidebar.title("📅 EvoTime Controls")
st.sidebar.caption("Intelligent Timetable Generator using Genetic Algorithm")

dataset_choice = st.sidebar.selectbox(
    "Select College Dataset",
    ["Feasible College Dataset", "Infeasible College Dataset (Impossible)"],
    index=0,
    help="Choose between a mathematically solvable 2-batch scenario or a deliberately overloaded impossible scenario.",
)

is_feasible_choice = (dataset_choice == "Feasible College Dataset")
current_dataset = get_feasible_dataset() if is_feasible_choice else get_infeasible_dataset()

# Run feasibility pre-check immediately
pre_check_report = check_feasibility(current_dataset)

force_execution = False
if not pre_check_report.ok:
    st.sidebar.error("⚠️ Feasibility Pre-Check Failed!")
    with st.sidebar.expander("View Detected Impossibilities", expanded=True):
        for err in pre_check_report.errors:
            st.write(f"❌ {err}")
    force_execution = st.sidebar.checkbox(
        "Run anyway (force execution)",
        value=False,
        help="Check this box to demonstrate that the GA fails and the validator reports violations on impossible data.",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🧬 GA Hyperparameters")

pop_size = st.sidebar.slider("Population Size", min_value=10, max_value=150, value=DEFAULT_POPULATION_SIZE, step=5)
max_gens = st.sidebar.slider("Max Generations", min_value=20, max_value=500, value=DEFAULT_MAX_GENERATIONS, step=10)
crossover_rate = st.sidebar.slider("Crossover Rate", min_value=0.1, max_value=1.0, value=DEFAULT_CROSSOVER_RATE, step=0.05)
mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.30, value=DEFAULT_MUTATION_RATE, step=0.01)
elitism_count = st.sidebar.number_input("Elitism Count", min_value=1, max_value=10, value=DEFAULT_ELITISM_COUNT, step=1)
tournament_k = st.sidebar.number_input("Tournament Size (k)", min_value=2, max_value=8, value=DEFAULT_TOURNAMENT_K, step=1)
seed = st.sidebar.number_input("Random Seed", value=DEFAULT_RANDOM_SEED, step=1)

# Disable generate button if pre-check fails and force is not checked
button_disabled = (not pre_check_report.ok and not force_execution)

generate_clicked = st.sidebar.button(
    "🚀 Generate Timetable",
    disabled=button_disabled,
    help="Launch the Genetic Algorithm optimization pipeline.",
    use_container_width=True,
)

if generate_clicked:
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    config = GAConfig(
        population_size=pop_size,
        max_generations=max_gens,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_count=elitism_count,
        tournament_k=tournament_k,
        seed=seed,
    )

    def ui_callback(gen: int, best_breakdown: Any, avg_penalty: float) -> None:
        pct = min(1.0, gen / float(max_gens))
        progress_bar.progress(pct)
        status_text.text(
            f"Gen {gen}/{max_gens} | Best Hard: {best_breakdown.hard_total} | Penalty: {best_breakdown.penalty:.1f}"
        )

    result = run_ga(current_dataset, config, callback=ui_callback)

    progress_bar.progress(1.0)
    status_text.text(f"Done! {result.stop_reason}")

    # Store state in session_state
    st.session_state["result"] = result
    st.session_state["dataset"] = current_dataset
    st.session_state["config"] = config
    st.session_state["dataset_choice"] = dataset_choice


# ============================================================================
# Main Page: Tabbed Interface
# ============================================================================

st.title("EvoTime — Intelligent College Timetable Generator")
st.caption("A Hand-Crafted Genetic Algorithm with Independent Constraint Verification")

(
    tab_input,
    tab_initial,
    tab_final,
    tab_validator,
    tab_graph,
    tab_comparison,
    tab_export,
) = st.tabs([
    "📋 Input Data",
    "🎲 Initial Timetable",
    "📅 Final Timetable",
    "🛡️ Validation Report",
    "📈 Optimization Graph",
    "⚖️ Comparison",
    "💾 Export",
])

has_run = ("result" in st.session_state and st.session_state.get("dataset_choice") == dataset_choice)


# ----------------------------------------------------------------------------
# Tab 1: Input Data
# ----------------------------------------------------------------------------
with tab_input:
    st.subheader(f"Dataset Specifications: {current_dataset.name}")
    st.write(current_dataset.description)

    # Feasibility Arithmetic Callout
    st.markdown("#### 🧮 Slot Capacity & Utilization Arithmetic")
    feasibility_text = format_feasibility_summary(pre_check_report)
    st.code(feasibility_text, language="text")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📚 Subjects & Courses")
        sub_rows = [
            {
                "ID": s.id,
                "Subject Name": s.name,
                "Type": s.type.upper(),
                "Sessions/Week": s.lectures_per_week,
                "Duration (Hrs)": s.duration,
                "Assigned Faculty": current_dataset.teachers[s.teacher_id].name if s.teacher_id in current_dataset.teachers else s.teacher_id,
            }
            for s in current_dataset.subjects.values()
        ]
        st.dataframe(pd.DataFrame(sub_rows), use_container_width=True, hide_index=True)

        st.markdown("##### 👥 Student Batches")
        batch_rows = [
            {
                "Batch ID": b.id,
                "Batch Name": b.name,
                "Student Strength": b.strength,
                "Enrolled Subjects": len(b.subject_ids),
            }
            for b in current_dataset.batches.values()
        ]
        st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("##### 👨‍🏫 Faculty Availability")
        teach_rows = [
            {
                "ID": t.id,
                "Teacher Name": t.name,
                "Available Slots (of 25)": len(t.available_slots),
                "Unavailable Slots": 25 - len(t.available_slots),
            }
            for t in current_dataset.teachers.values()
        ]
        st.dataframe(pd.DataFrame(teach_rows), use_container_width=True, hide_index=True)

        st.markdown("##### 🏢 Rooms & Laboratories")
        room_rows = [
            {
                "Room ID": r.id,
                "Room Name": r.name,
                "Seating Capacity": r.capacity,
                "Type": r.kind.upper(),
            }
            for r in current_dataset.rooms.values()
        ]
        st.dataframe(pd.DataFrame(room_rows), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Tab 2: Initial Timetable (Generation 0)
# ----------------------------------------------------------------------------
with tab_initial:
    if not has_run:
        st.info("👋 Please click **'🚀 Generate Timetable'** in the sidebar to run the Genetic Algorithm and view the initial candidate schedule.")
    else:
        res = st.session_state["result"]
        ds = st.session_state["dataset"]

        st.subheader("🎲 Initial Candidate Timetable (Generation 0)")
        st.markdown("This timetable was generated randomly prior to evolutionary optimization to establish our baseline.")

        # Status Banner
        init_hard = res.initial_breakdown.hard_total
        if init_hard == 0:
            st.success("✅ VALID TIMETABLE — Zero hard conflicts in Generation 0 (rare random occurrence)!")
        else:
            st.error(
                f"❌ UNOPTIMIZED SCHEDULE — Contains **{init_hard} Hard Clashes** "
                f"| Total Penalty Score: **{res.initial_breakdown.penalty:.1f}**"
            )

        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            init_view = st.radio("View Schedule By:", ["Batch", "Teacher", "Room"], key="init_view_radio", horizontal=True)
        with col_v2:
            if init_view == "Batch":
                init_entity = st.selectbox("Select Batch:", list(ds.batches.keys()), format_func=lambda x: ds.batches[x].name, key="init_batch_sel")
            elif init_view == "Teacher":
                init_entity = st.selectbox("Select Faculty:", list(ds.teachers.keys()), format_func=lambda x: ds.teachers[x].name, key="init_teach_sel")
            else:
                init_entity = st.selectbox("Select Room:", list(ds.rooms.keys()), format_func=lambda x: ds.rooms[x].name, key="init_room_sel")

        init_grid = build_grid(res.initial_best, ds, view=init_view.lower(), entity_id=init_entity)
        st.markdown(grid_to_html(init_grid), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Tab 3: Final Timetable (Evolved)
# ----------------------------------------------------------------------------
with tab_final:
    if not has_run:
        st.info("👋 Please click **'🚀 Generate Timetable'** in the sidebar to run the Genetic Algorithm and inspect the final schedule.")
    else:
        res = st.session_state["result"]
        ds = st.session_state["dataset"]

        st.subheader(f"📅 Final Evolved Timetable ({res.dataset_name})")

        # Independent Validator Status Banner
        if res.validation_report.is_valid:
            st.success(
                f"✅ **VALID TIMETABLE — VERIFIED CONFLICT-FREE BY INDEPENDENT VALIDATOR!**\n\n"
                f"Satisfies all hard constraints with 0 clashes. Total Penalty: **{res.final_breakdown.penalty:.1f}** "
                f"| Generations Executed: **{res.generations_run}** | Status: **{res.status}**"
            )
        else:
            st.error(
                f"❌ **FAILED — VIOLATIONS REMAINING (Not Conflict-Free)**\n\n"
                f"Independent validator detected **{res.validation_report.hard_count} hard violations** "
                f"after {res.generations_run} generations. Total Penalty: **{res.final_breakdown.penalty:.1f}**"
            )

        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            final_view = st.radio("View Schedule By:", ["Batch", "Teacher", "Room"], key="final_view_radio", horizontal=True)
        with col_f2:
            if final_view == "Batch":
                final_entity = st.selectbox("Select Batch:", list(ds.batches.keys()), format_func=lambda x: ds.batches[x].name, key="final_batch_sel")
            elif final_view == "Teacher":
                final_entity = st.selectbox("Select Faculty:", list(ds.teachers.keys()), format_func=lambda x: ds.teachers[x].name, key="final_teach_sel")
            else:
                final_entity = st.selectbox("Select Room:", list(ds.rooms.keys()), format_func=lambda x: ds.rooms[x].name, key="final_room_sel")

        final_grid = build_grid(res.final_best, ds, view=final_view.lower(), entity_id=final_entity)
        st.markdown(grid_to_html(final_grid), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Tab 4: Validation Report
# ----------------------------------------------------------------------------
with tab_validator:
    if not has_run:
        st.info("👋 Please click **'🚀 Generate Timetable'** in the sidebar to run the algorithm and view the validation report.")
    else:
        res = st.session_state["result"]
        st.subheader("🛡️ Independent Constraint Validation Report")
        st.caption("Audited from first principles by validator.py (sharing no code with the fitness engine)")

        val_rep = res.validation_report
        detailed_df, summary_df = violations_table(val_rep)

        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            st.metric("Independent Auditor Verdict", "VALID ✅" if val_rep.is_valid else "FAILED ❌")
            st.metric("Total Hard Violations", val_rep.hard_count)
            st.markdown("##### Violations by Category")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with col_v2:
            st.markdown("##### Detailed Violation Audit Log")
            st.dataframe(detailed_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🧘 Soft Pedagogical Preferences (Quality Breakdown)")
        st.caption("Note: Soft constraints reflect schedule comfort and ergonomics; they do not invalidate a schedule.")

        soft_units = res.final_breakdown.soft_units
        soft_rows = [
            {"Soft Constraint Preference": "Teacher Fatigue (>3 Consecutive Periods)", "Violation Units": soft_units.get("teacher_consecutive", 0)},
            {"Soft Constraint Preference": "Subject Clustering (>1 Same Subject / Day)", "Violation Units": soft_units.get("batch_subject_duplicate", 0)},
            {"Soft Constraint Preference": "Student Idle Gap Periods", "Violation Units": soft_units.get("batch_gaps", 0)},
            {"Soft Constraint Preference": "Faculty Daily Workload Variance", "Violation Units": soft_units.get("teacher_workload_variance", 0)},
            {"Soft Constraint Preference": "Undesirable Patterns (Last-period Lab, 1-Lecture Day)", "Violation Units": soft_units.get("undesirable_patterns", 0)},
        ]
        st.dataframe(pd.DataFrame(soft_rows), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Tab 5: Optimization Graph
# ----------------------------------------------------------------------------
with tab_graph:
    if not has_run:
        st.info("👋 Please click **'🚀 Generate Timetable'** in the sidebar to run the GA and view convergence charts.")
    else:
        res = st.session_state["result"]
        st.subheader("📈 Evolutionary Convergence Curves")

        hist_df = history_dataframe(res.history)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Initial Best Penalty", f"{res.initial_breakdown.penalty:.1f}")
        col_m2.metric("Final Best Penalty", f"{res.final_breakdown.penalty:.1f}", delta=f"{res.final_breakdown.penalty - res.initial_breakdown.penalty:.1f}")
        col_m3.metric("Generations Run", res.generations_run)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Subplot 1: Penalty Curves
        ax1.plot(hist_df["Generation"], hist_df["Best Penalty"], label="Best Penalty (Elitist)", color="#16a34a", linewidth=2.5)
        ax1.plot(hist_df["Generation"], hist_df["Average Penalty"], label="Population Average Penalty", color="#3b82f6", linestyle="--", alpha=0.7)
        ax1.set_ylabel("Penalty Score (Lower is Better)")
        ax1.set_title("Fitness Penalty Minimization Across Generations")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")

        # Subplot 2: Hard Violations
        ax2.plot(hist_df["Generation"], hist_df["Hard Violations"], label="Hard Violations Count", color="#dc2626", linewidth=2.0)
        ax2.set_xlabel("Generation Number")
        ax2.set_ylabel("Hard Clashes")
        ax2.grid(True, linestyle=":", alpha=0.6)

        # Highlight generation when hard violations first hit 0
        zero_hard_df = hist_df[hist_df["Hard Violations"] == 0]
        if not zero_hard_df.empty:
            first_zero_gen = int(zero_hard_df.iloc[0]["Generation"])
            ax2.axvline(x=first_zero_gen, color="#16a34a", linestyle="--", linewidth=1.5)
            ax2.annotate(
                f"Zero Clashes reached at Gen {first_zero_gen}",
                xy=(first_zero_gen, 0),
                xytext=(first_zero_gen + 10, max(hist_df["Hard Violations"]) * 0.4),
                arrowprops=dict(facecolor="#16a34a", shrink=0.08, width=1, headwidth=6),
                fontweight="bold",
                color="#16a34a",
            )

        ax2.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Tab 6: Comparison (Before & After)
# ----------------------------------------------------------------------------
with tab_comparison:
    if not has_run:
        st.info("👋 Please click **'🚀 Generate Timetable'** in the sidebar to generate schedules for comparison.")
    else:
        res = st.session_state["result"]
        ds = st.session_state["dataset"]

        st.subheader("⚖️ Initial (Gen 0) vs Final Evolved Schedule Comparison")

        init_report = validate(res.initial_best, ds)
        comp_df = comparison_table(res.initial_breakdown, res.final_breakdown, init_report, res.validation_report)

        init_hard = init_report.hard_count
        final_hard = res.validation_report.hard_count
        diff_hard = init_hard - final_hard

        st.markdown(
            f"**Real Performance Summary:**\n"
            f"* Hard constraint violations reduced from **{init_hard}** in Generation 0 to **{final_hard}** "
            f"in Generation {res.generations_run} (a reduction of **{diff_hard} clashes**).\n"
            f"* Overall penalty score improved from **{res.initial_breakdown.penalty:.1f}** down to "
            f"**{res.final_breakdown.penalty:.1f}** (Normalized fitness improved from "
            f"`{res.initial_breakdown.fitness_score:.6f}` to `{res.final_breakdown.fitness_score:.6f}`).\n"
            f"* Independent validator outcome changed from **{'VALID' if init_report.is_valid else 'FAILED'}** to **{'VALID' if res.validation_report.is_valid else 'FAILED'}**."
        )

        st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Tab 7: Export
# ----------------------------------------------------------------------------
with tab_export:
    if not has_run:
        st.info("👋 Please click **'🚀 Generate Timetable'** in the sidebar to synthesize a schedule for export.")
    else:
        res = st.session_state["result"]
        ds = st.session_state["dataset"]

        st.subheader("💾 Export Weekly Timetable Schedule")
        st.markdown("Download the full timetable schedule in long-format CSV suitable for importing into Excel, Google Sheets, or ERP systems.")

        csv_str, filename = export_csv(
            res.final_best,
            ds,
            status=res.status,
            hard_violations_count=res.validation_report.hard_count,
        )

        if res.status != "SUCCESS":
            st.warning(
                f"⚠️ **Caution:** The timetable contains **{res.validation_report.hard_count} hard violations** "
                f"and is marked as **FAILED**. The downloaded file will carry the prefix `{filename}` to prevent accidental deployment."
            )
        else:
            st.success("✅ This timetable is verified conflict-free and approved for academic scheduling.")

        st.download_button(
            label="📥 Download Schedule (CSV)",
            data=csv_str,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("##### Preview of Export Table (First 10 Rows)")
        preview_df = pd.read_csv(pd.io.common.StringIO(csv_str))
        st.dataframe(preview_df.head(10), use_container_width=True, hide_index=True)

