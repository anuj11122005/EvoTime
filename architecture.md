# architecture.md — EvoTime System Architecture

## 1. Overview
EvoTime is a local Streamlit application driven by a hand-written Genetic Algorithm (GA). The GA searches for a weekly timetable that satisfies all hard constraints and minimizes soft-constraint penalties. A separate validator independently verifies the final result.

```
┌────────────────────┐
│  Streamlit UI      │  app.py
│  (controls, grid,  │
│   graphs, reports) │
└─────────┬──────────┘
          │ config + button click
          ▼
┌────────────────────┐      ┌──────────────────────┐
│  data module       │─────▶│  GA engine           │
│  sample_data.py    │      │  ga.py               │
│  models.py         │      │  init/select/cross/  │
└────────────────────┘      │  mutate/elitism      │
                            └───────┬──────────────┘
                                    │ uses
                       ┌────────────▼─────────────┐
                       │ fitness.py (penalty)     │
                       │ repair.py (fix genes)    │
                       └────────────┬─────────────┘
                                    │ best timetable + history
                                    ▼
                       ┌──────────────────────────┐
                       │ validator.py             │  independent, no shared
                       │ (hard-constraint checks) │  code with fitness.py
                       └────────────┬─────────────┘
                                    ▼
                       ┌──────────────────────────┐
                       │ results.py               │
                       │ DataFrames, grid, plots, │
                       │ CSV/PDF export           │
                       └──────────────────────────┘
```

## 2. Folder structure
```
evotime/
├── app.py                  # Streamlit dashboard (built LAST)
├── requirements.txt
├── README.md
├── rule.md
├── architecture.md
├── document.md
├── evotime/
│   ├── __init__.py
│   ├── config.py           # GA parameters, penalty weights, days/periods
│   ├── models.py           # dataclasses: Subject, Teacher, Room, Batch, Lecture, Gene, Timetable
│   ├── sample_data.py      # feasible + infeasible datasets
│   ├── feasibility.py      # pre-run slot/capacity sanity checks
│   ├── ga.py               # GA loop, init, selection, crossover, mutation, elitism
│   ├── fitness.py          # penalty calculation (hard + soft)
│   ├── repair.py           # repair mechanism
│   ├── validator.py        # independent hard-constraint validator
│   └── results.py          # tables, grid, plots, export
├── tests/
│   ├── test_models.py
│   ├── test_fitness.py
│   ├── test_validator.py
│   ├── test_ga_operators.py
│   └── test_end_to_end.py  # feasible + infeasible scenarios
└── run_cli.py              # command-line GA runner for Phases 1–4
```

## 3. Data model
| Entity | Fields |
|---|---|
| Subject | id, name, type (`theory`/`practical`), lectures_per_week, duration (1 or 2 periods), teacher_id |
| Teacher | id, name, available_slots (set of (day, period)) |
| Room | id, name, capacity, kind (`classroom`/`lab`) |
| Batch | id, name, strength, subject_ids |
| Lecture (requirement) | lecture_id, subject_id, teacher_id, batch_id, duration, needs_lab |
| Gene (assignment) | lecture_id + room_id, day, start_period |
| Chromosome / Timetable | list of Genes, exactly one per Lecture |

A time slot is `(day, period)` with day ∈ 0–4 (Mon–Fri) and period ∈ 0–4.

## 4. Chromosome and gene representation
- **Chromosome = one complete timetable** = list of N genes, where N = total required lecture sessions.
- **Gene = one lecture session.** Fixed (from requirements): lecture_id, subject, teacher, batch, duration. **Variable (evolved):** room, day, start_period.
- Because every gene is tied to a unique `lecture_id`, the chromosome always has exactly the required lectures: crossover/mutation cannot lose or duplicate one.
- A practical (duration 2) gene occupies `start_period` and `start_period + 1`, so `start_period ≤ periods_per_day − 2`.

## 5. Genetic Algorithm design
| Step | Design |
|---|---|
| Initialization | Random valid-range genes per lecture (room matches kind, practical start fits in day). Optionally bias to teacher-available slots. Population size 50 (configurable). |
| Penalty | `1000 × hard + 10 × soft` (weights in config). Lower is better. Fitness for display = `1/(1+penalty)`. |
| Selection | Tournament selection, k = 3. |
| Crossover | Uniform crossover **by lecture_id**: for each lecture, child takes (room, day, start) from parent A or B with probability 0.5. Alternative: single-point on the gene list (both are index-aligned). Rate ≈ 0.8. |
| Mutation | Per-gene with rate ≈ 0.05–0.1: change room, or move to another day/period, or swap slots of two genes. Conflict-directed mutation (prefer genes involved in clashes) is encouraged. |
| Repair | After operators: clamp ranges, fix practical blocks that overflow the day, force lab rooms for practicals, replace undersized rooms with a suitable free room. |
| Elitism | Top E (default 2) individuals copied unchanged to next generation. |
| Termination | Hard violations = 0 and penalty ≤ threshold, or max generations (default 100–300). Stagnation restart optional. |
| Reproducibility | Seeded `random.Random`. |

## 6. Constraints
### Hard (each violation counts 1 × 1000)
1. Teacher clash (same teacher, same slot).
2. Room clash.
3. Batch clash.
4. Lecture count: every required lecture exactly once (missing/duplicate).
5. Teacher unavailable in assigned slot.
6. Room capacity < batch strength.
7. Practical not in consecutive periods / not in a lab / crossing day boundary.

### Soft (each unit counts 1 × 10)
1. Teacher has more than 3 consecutive periods.
2. Same subject repeated more than once on a day for a batch (poor distribution).
3. Gaps (free periods) between a batch's classes in a day.
4. Teacher daily workload deviates from the mean.
5. Undesirable patterns (e.g. batch's day with only one lecture, last-period practicals).

## 7. Validator (independence requirement)
`validator.py` re-derives every check from the raw timetable using its own slot-occupancy maps. It imports models only, **never `fitness.py`**. Output:
```
ValidationReport:
  is_valid: bool                # True only if zero hard violations
  hard_violations: list[Violation(kind, details)]
  soft_summary: dict            # informational
```
The UI and CLI use only `is_valid` from this report to claim "conflict-free".

## 8. Feasibility pre-check (`feasibility.py`)
Before running the GA, verify necessary conditions:
- For each batch: total required periods ≤ working days × periods per day.
- For each teacher: assigned periods ≤ number of available slots.
- Enough lab rooms for practicals; each batch fits at least one room of required kind.
- Total room-slots ≥ total required periods.
A failed pre-check shows an explicit "infeasible configuration" message. (Passing the pre-check is necessary, not sufficient; the validator still decides.)

## 9. Sample dataset sizing (feasible)
- 5 days × 5 periods = 25 slots per batch; 2 batches → 50 batch-slots.
- Required per batch: 6 theory subjects (3+3+3+3+3+2 = 17 periods) + 2 labs (2 periods each = 4) → ~21 periods; both batches → ~42 of 50 slots (≈84% utilization: feasible but non-trivial).
- 5 teachers, each with ≈8–10 periods/week against ≥15 available slots.
- 5 rooms: 3 classrooms, 2 labs (capacities ≥ batch strengths).
- Infeasible dataset: e.g. add subjects until a batch needs >25 periods, or make one teacher available for fewer slots than assigned.

## 10. Data flow
1. UI loads `sample_data` → builds `Lecture` requirements.
2. `feasibility` checks the configuration.
3. `ga.init_population` creates random timetables; the **initial best** is recorded and validated (shown as "before").
4. Each generation: evaluate with `fitness`, select, crossover, mutate, `repair`, apply elitism, log best/avg penalty.
5. On termination, the best timetable goes to `validator`.
6. `results` builds the colored grid, violation tables, convergence plot, before/after comparison, CSV/PDF export.

## 11. UI layout (Streamlit)
- Sidebar: dataset choice (feasible / infeasible), population size, generations, crossover rate, mutation rate, elitism count, tournament size, seed, **Generate Timetable** button.
- Tabs: Input Data · Initial Timetable · Final Timetable · Validation Report · Optimization Graph · Comparison · Export.
- Grid: rows = periods, columns = days; one grid per batch; color per subject; each cell shows subject, teacher, room, batch.
- Status banner: ✅ VALID (validator confirmed) or ❌ FAILED (violations listed).

## 12. Complexity (for viva)
Per generation: O(P × N × c) for evaluation, where P = population, N = genes, c = cost per gene check (hash-map slot occupancy makes this near-linear). Total ≈ O(G × P × N).

## 13. Known limitations
- Heuristic search: no guarantee of global optimum or of finding a feasible solution within the generation limit.
- Single-institution small dataset; no editing of data beyond the sample files in v1.
- Soft-constraint weights are hand-tuned.
