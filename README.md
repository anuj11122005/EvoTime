# EvoTime — Intelligent College Timetable Generator Using a Genetic Algorithm

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-60%20passed-success.svg)]()
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**EvoTime** is an automated college timetable generator developed from first principles in pure Python for Soft Computing. It formulates weekly timetable generation as a constrained combinatorial optimization problem and solves it using a **hand-written Genetic Algorithm (GA)** without external optimization libraries.

A core architectural principle of EvoTime is **academic integrity through independent verification**: candidate schedules are audited by a decoupled validator that re-evaluates all hard constraints from scratch before declaring a timetable valid.

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository, create a virtual environment (verified on **Python 3.12.3**; compatible with Python 3.10+), and install dependencies:

```bash
# Clone the repository
git clone https://github.com/anuj/EvoTime.git
cd EvoTime

# Create and activate a virtual environment
# On Windows:
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Test Suite (60/60 Tests)
```bash
python -m pytest tests/ -v
```

### 3. Run the Command-Line Interface (CLI)
```bash
# Run on the standard feasible college dataset (Seed 42)
python run_cli.py

# Run on the deliberately impossible infeasible dataset with --force
python run_cli.py --dataset infeasible --force
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📂 Project Architecture & Folder Structure

```
EvoTime/
├── app.py                      # Interactive Streamlit Web Dashboard (7 Tabs)
├── run_cli.py                  # Terminal-based GA execution & audit runner
├── experiments.py              # Multi-seed benchmarking & evaluation suite
├── tuning_experiments.py       # Reproducible parameter ablation test harness
├── requirements.txt            # Streamlit, Pandas, NumPy, Matplotlib, Pytest
├── README.md                   # Project documentation, guides & viva notes
├── architecture.md             # High-level system design & specifications
├── rule.md                     # Binding coding & academic honesty standards
├── document.md                 # Complete viva defense prep & report outline
│
├── evotime/                    # Core Python Package
│   ├── __init__.py             # Package declaration (v1.0.0)
│   ├── config.py               # Hyperparameters, penalty weights & grid bounds
│   ├── models.py               # Dataclasses: Subject, Teacher, Room, Batch, Gene, Timetable
│   ├── sample_data.py          # Benchmark feasible & infeasible college datasets
│   ├── feasibility.py          # Deterministic pre-run capacity & slot arithmetic analyzer
│   ├── fitness.py              # Penalty scoring engine (hard clashes + soft preferences)
│   ├── repair.py               # Structural repair (clamps bounds, enforce lab kinds)
│   ├── ga.py                   # Pure hand-written Genetic Algorithm search engine
│   ├── validator.py            # INDEPENDENT auditor (shares no code with fitness.py)
│   └── results.py              # Pure-function grid generators, HTML styling & CSV export
│
├── tests/                      # Automated Regression & Unit Test Suite
│   ├── helpers.py              # Deterministic test fixture generator
│   ├── test_models.py          # Dataclass initialization & slot expansion tests
│   ├── test_sample_data.py     # Dataset counts, uniqueness & structure tests
│   ├── test_feasibility.py     # Feasibility pre-check validation tests
│   ├── test_fitness.py         # Hard and soft penalty calculation tests
│   ├── test_validator.py       # Independent validator & 250-trial cross-check tests
│   ├── test_ga_operators.py    # Crossover, mutation, repair & elitism tests
│   ├── test_end_to_end.py      # End-to-end multi-seed integration tests
│   ├── test_results.py         # Grid builder, HTML table & CSV export tests
│   └── test_app.py             # Streamlit AppTest integration tests
│
└── results/                    # Empirical Benchmark Datasets
    ├── baseline_results.csv    # 20 hold-out test seeds on baseline config (85.0%)
    ├── tuning_results.csv      # 10 tuning seeds across 5 parameter variants
    └── tuned_results.csv       # 20 hold-out test seeds on tuned config (95.0%)
```

---

## 🧩 Module Overview

| Module | Purpose | Key Functions / Responsibilities |
|---|---|---|
| `config.py` | Configuration | Defines schedule bounds (5 days × 5 periods = 25 slots), GA parameters (`pop=50, gen=250, mut=0.08, k=4`), and constraint penalty weights ($1000 \times \text{hard} + 10 \times \text{soft}$). |
| `models.py` | Data Representations | Lightweight `dataclasses` for `Subject`, `Teacher`, `Room`, `Batch`, `Lecture` (immutable requirements), `Gene` (variable assignments: room, day, period), and `Timetable` (chromosome). |
| `sample_data.py` | Benchmarks | Realistic 2-batch engineering scenario (`CSE-A` strength 60, `CSE-B` strength 55) requiring 42 of 50 available batch-slots (**84% utilization**), and an impossible overloaded dataset. |
| `feasibility.py` | Pre-Checker | Evaluates necessary mathematical conditions (batch slot limit $\le 25$, faculty availability $\ge$ load, lab counts, room capacities) before launching the GA. |
| `fitness.py` | Objective Function | Computes scalar penalty score. Hard violations carry 1,000 points; soft preferences carry 10 points. Derives normalized fitness score: $\text{Fitness} = 1 / (1 + \text{Penalty})$. |
| `repair.py` | Structural Repair | Guarantees dimensional validity: clamps out-of-range days/periods, prevents 2-period practicals from overflowing past period 4, and forces practicals into labs. Never alters lecture counts. |
| `ga.py` | Genetic Algorithm | Hand-written GA: population init (random + availability bias), tournament selection ($k=4$), uniform crossover, conflict-directed mutation, and elitism ($E=2$). |
| `validator.py` | Independent Auditor | **Strictly decoupled from `fitness.py`**. Re-evaluates candidate timetables from raw first principles using independent slot-occupancy dictionaries. Only authority on validity. |
| `results.py` | Data Formatting | Transforms chromosomes into 2D schedule grids (by Batch, Faculty, or Room), generates color-coded HTML tables, audit tables, and long-format CSV exports. |
| `app.py` | Streamlit Dashboard | User-friendly dashboard featuring 7 dedicated tabs, live progress bar, Matplotlib convergence plots, and guarded CSV export. |
| `run_cli.py` | Command-Line Runner | Terminal-based workflow runner with pre-check validation, progress streaming, and independent validation audit. |
| `experiments.py`| Benchmark Harness | Automated multi-seed evaluation tool that records success rates, average generations, and runtimes across independent seeds. |

---

## 🧬 Genetic Algorithm Design

```
   ┌────────────────────────────────────────────────────────┐
   │  1. Population Initialization                          │
   │  50 random candidate timetables (repaired)             │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │  2. Evaluation & Elitism                               │
   │  Evaluate penalty: 1000 * hard + 10 * soft             │
   │  Preserve Top 2 individuals unchanged (Elitism)        │
   └──────────────────────────┬─────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
   ┌───────────────────────┐     ┌────────────────────────┐
   │  3. Selection         │     │  4. Uniform Crossover  │
   │  Tournament (k = 4)   │────▶│  50% Gene inheritance  │
   └───────────────────────┘     └───────────┬────────────┘
                                             │
                                             ▼
   ┌───────────────────────┐     ┌────────────────────────┐
   │  6. Repair Mechanism  │     │  5. Mutation           │
   │  Clamp bounds & labs  │◀────│  Room / Time / Swap +  │
   └───────────┬───────────┘     │  Conflict-directed     │
               │                 └────────────────────────┘
               ▼
   ┌────────────────────────────────────────────────────────┐
   │  7. Termination Condition                              │
   │  Zero hard clashes & penalty <= 300, OR 250 gens       │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │  8. Independent Auditor (validator.py)                 │
   │  Zero hard violations? ──▶ SUCCESS (Valid) / FAILED    │
   └────────────────────────────────────────────────────────┘
```

* **Chromosome:** A list of 38 `Gene` assignments, index-aligned 1-to-1 with curriculum lecture requirements.
* **Gene:** `(lecture_id, room_id, day, start_period)`. Because lecture requirements are fixed, **genetic operators can never drop, duplicate, or omit a lecture**.
* **Tournament Selection ($k=4$):** Randomly samples 4 candidate timetables and picks the one with the lowest penalty.
* **Uniform Crossover:** For each lecture, the child inherits the room and time slot from Parent A or Parent B with 50% probability.
* **Conflict-Directed Mutation:** Identifies double-booked faculty, rooms, or batches and prioritizes nudging clashing genes into open or teacher-available slots.
* **Elitism ($E=2$):** Preserves the two best timetables unchanged into the next generation so best fitness never degrades.

---

## 📊 Empirical Benchmarks & Tuning Results

EvoTime was systematically tuned on 10 tuning seeds (1–10) and independently evaluated on 20 hold-out test seeds (101–120):

| Configuration | Test Seeds | Success Rate | Avg Generations | Avg Soft Penalty | Real Performance Note |
|---|---|---|---|---|---|
| **Baseline** (`k=3, gen=200`) | 101–120 | 17 / 20 (85.0%) | 134.5 gens | 23.9 units | 3 seeds failed at gen 200 with 1–2 clashes. |
| **Tuned** (`k=4, gen=250`) | 101–120 | **19 / 20 (95.0%)** | **80.4 gens** | **23.5 units** | **40% faster convergence**, 1 failure (Seed 115). |

### Transparent Analysis of the Single Failed Test Seed (Seed 115)
In Seed 115, the GA resolved all teacher, batch, and room overlaps, but could not swap `Prof. Shah`'s lab out of Monday morning before the 250-generation limit expired:
```
[TEACHER_UNAVAILABLE]: Teacher 'Prof. Shah' is scheduled during unavailable slot [(0, 1)] for lecture 'L_B_CSE_B_CS308L_1'
```
Because 1 hard violation remained, EvoTime honestly declared status `FAILED` and refused to generate a valid certificate.

---

## 🎯 10-Minute Viva Demonstration Script

1. **Start the Dashboard:** Run `streamlit run app.py` and open `http://localhost:8501`.
2. **Tab 1 (Input Data):** Show the curriculum, faculty constraints, and the pre-run arithmetic showing **84.0% slot utilization** (21 periods needed of 25 available).
3. **Generate Timetable:** Click **"🚀 Generate Timetable"** in the sidebar. Point out the live progress bar advancing generation-by-generation.
4. **Tab 2 (Initial Timetable):** Show the red unoptimized banner from Generation 0 (~15 hard clashes, penalty ~15,400). Toggle the radio button between Batch, Teacher, and Room.
5. **Tab 3 (Final Timetable):** Show the green **VALID TIMETABLE** banner verified by the independent validator. Highlight how 2-period practicals neatly occupy 2 consecutive periods in lab rooms.
6. **Tab 4 (Validation Report):** Point out the hard violations table showing **0 violations**, and explain that the validator shares no code with the fitness function.
7. **Tab 5 (Optimization Graph):** Explain the downward penalty curve, and show the green vertical dashed line marking the exact generation where clashes reached zero.
8. **Tab 6 (Comparison):** Point out the side-by-side Generation 0 vs Final table showing hard clashes dropping from 15 to 0.
9. **Tab 7 (Export):** Download the `VALID_timetable_schedule.csv` file.
10. **Infeasible Demonstration:** Select *"Infeasible College Dataset"* in the sidebar. Note that the Generate button is disabled until checking *"Run anyway (force)"*. Run it to show the GA terminating honestly with status **FAILED** and 9 violations.

---

## 🛠️ Troubleshooting Guide

### 1. Virtual Environment Activation
* **Windows (PowerShell):** If you receive `execution of scripts is disabled on this system`, run PowerShell as Administrator and execute:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
* **macOS / Linux:** Ensure you source the activation script: `source .venv/bin/activate`.

### 2. Streamlit Port Already in Use (Port 8501)
If port 8501 is occupied by another process:
```bash
streamlit run app.py --server.port 8502
```

### 3. Module Import Errors (`ModuleNotFoundError: No module named 'evotime'`)
Ensure you run scripts from the project root directory. If running outside the root, set your `PYTHONPATH`:
```bash
# Windows (PowerShell):
$env:PYTHONPATH="."
python run_cli.py

# Linux/macOS:
PYTHONPATH=. python3 run_cli.py
```

### 4. Matplotlib GUI / Backend Errors in Headless Environments
If running on a headless server or container, Matplotlib may raise `UserWarning: Matplotlib is currently using agg, which is a non-GUI backend`. This is intentional: `app.py` explicitly sets `matplotlib.use("Agg")` to ensure thread-safe, non-blocking plot generation.

---

## ⚠️ Known Limitations & Academic Honesty

1. **Heuristic Nature (No Optimality Guarantee):** A Genetic Algorithm is a heuristic search. It provides no mathematical guarantee of finding a globally valid schedule on 100% of runs within finite generations (as proven by our 95% test seed success rate).
2. **Fixed Grid Dimensions:** The current implementation assumes a 5-day week (Monday to Friday) and 5 periods per day (25 slots). Extending to 6-day weeks or variable period lengths requires updating `config.py`.
3. **CSV Export Primary:** Per the specification, CSV export is the primary export format. Direct PDF generation via `reportlab` was skipped because `reportlab` is not bundled in standard project dependencies.
4. **Soft-Constraint Weight Tuning:** Soft constraint penalty weights (10 points each) are hand-tuned institutional preferences.

