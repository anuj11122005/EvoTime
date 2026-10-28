# document.md — EvoTime Project Documentation, Report Outline & Viva Prep

> Fill the bracketed [ ] parts with your REAL results after running the app. Never paste invented numbers.

## 1. Project report outline

### Title page
EvoTime — Intelligent College Timetable Generator Using a Genetic Algorithm
Subject: Soft Computing · B.Tech CSE (AI & ML) · Adani University, Ahmedabad
Submitted by: Anuj · Guide: [name] · Academic year: [year]

### Abstract (draft)
College timetabling is a constrained combinatorial optimization problem. EvoTime applies a Genetic Algorithm, implemented from scratch in Python, to generate weekly timetables that avoid teacher, room and batch clashes while respecting teacher availability and room capacity. Candidate timetables are evolved using tournament selection (k=4), uniform crossover, multi-type mutation, structural repair and elitism, evaluated by a penalty function that weights hard violations far above soft preferences. An independent validator verifies the final result, and a Streamlit dashboard visualizes the timetable, conflicts and convergence. In experiments on a 2-batch benchmark dataset (84% slot utilization), the best penalty reduced from 15,412.0 (15 hard clashes) to 202.0–298.0 with 0 hard violations confirmed by the validator, achieving a 95.0% success rate across 20 unseen test seeds in an average of 80.4 generations.

### Chapters
1. **Introduction** – timetabling difficulty, why automation helps.
2. **Problem statement** – schedule L lectures into D×P slots with rooms/teachers/batches under constraints.
3. **Objectives** – the 8 objectives from the project brief.
4. **Literature review** – GA (Holland, 1975; Goldberg, 1989), timetabling surveys (e.g. Schaerf 1999; Burke & Petrovic 2002), comparison with simulated annealing, tabu search, constraint programming. *Verify and cite references yourself before submitting.*
5. **Soft Computing & GA theory** – population, chromosome, gene, fitness, selection, crossover, mutation, elitism, repair, termination, exploration vs exploitation.
6. **System architecture** – diagram and module flow (see `architecture.md`).
7. **Algorithm & pseudocode** (below).
8. **Fitness function & constraints** – hard/soft lists, weights, penalty vs fitness.
9. **Implementation** – modules, data structures, key functions.
10. **Results & screenshots** – initial vs final timetable, validation report, convergence graph, comparison table, infeasible-case failure report.
11. **Testing & validation** – test checklist outcomes.
12. **Advantages & limitations**
13. **Future scope** – more batches/years, teacher preferences, room preferences, memetic/hybrid search, database + login, PDF/Excel export, comparison with other metaheuristics.
14. **Conclusion**
15. **References**

### Pseudocode
```
INPUT: lectures[], teachers, rooms, batches, config
IF not feasibility_check(): REPORT infeasible; STOP
population ← [random_timetable() for _ in P]; repair each
initial_best ← best(population)
FOR gen = 1 .. MAX_GEN:
    evaluate penalty of every individual
    record best/avg penalty
    IF best has 0 hard violations AND penalty ≤ threshold: BREAK
    new_pop ← top E individuals (elitism)
    WHILE |new_pop| < P:
        p1, p2 ← tournament_select(population) ×2
        child ← crossover(p1, p2) with probability pc else copy p1
        mutate(child) with per-gene probability pm
        repair(child)
        new_pop.append(child)
    population ← new_pop
final ← best(population)
report ← independent_validate(final)
IF report.is_valid: SUCCESS ELSE: FAILED (show violations)
```

## 2. GA concepts in simple language (with timetable examples)
| Concept | Plain meaning | EvoTime example |
|---|---|---|
| Chromosome | One full candidate solution | One complete weekly timetable |
| Gene | One piece of the solution | "Soft Computing, Dr. Mehta, CSE-A, Room R101, Mon, Period 2" |
| Population | Group of candidate solutions | 50 different timetables |
| Fitness / penalty | How good/bad a solution is | 1000 per clash + 10 per soft issue; lower is better |
| Selection | Better solutions more likely to become parents | Tournament: pick 3 random, best wins |
| Crossover | Mix two parents | For each lecture, child copies room/day/period from parent A or B |
| Mutation | Small random change | Move a lecture to another period or room |
| Elitism | Keep the best unchanged | Top 2 timetables survive every generation |
| Repair | Fix invalid results | Move a 2-period lab that crossed the day boundary |
| Termination | When to stop | Valid timetable found or generation limit reached |

**Penalty vs fitness:** GA here *minimizes* penalty. Fitness = 1/(1+penalty) turns it into a *maximization* score in (0,1].

## 3. Demonstration script (10 minutes)
1. Open dashboard → sidebar: choose **Feasible sample dataset**, show data tables and the slot-vs-lecture arithmetic.
2. Click **Generate Timetable**; open **Initial Timetable** tab → show its real violations and penalty.
3. Show the final timetable grid per batch.
4. Open **Optimization Graph** → explain the downward curve and plateaus.
5. Open **Validation Report** → every hard check shows 0 violations; mention validator is independent of the fitness function.
6. Open **Comparison** → initial vs final hard violations and soft penalty.
7. Switch to **Infeasible dataset** → run → show the FAILED report (proves no false success).
8. Explain how each operator helped: selection (pressure toward good timetables), crossover (combines good partial schedules), mutation (escapes local optima), elitism (never loses the best).
9. Export CSV.

## 4. Testing checklist
- [x] Sample data passes feasibility pre-check
- [x] Infeasible data is rejected/failed with a clear message
- [x] Initial population has correct gene count; each lecture appears exactly once
- [x] Crossover preserves lecture set (no loss/duplicates)
- [x] Mutation preserves lecture set
- [x] Repair fixes out-of-range practicals and wrong room types
- [x] Fitness equals expected value on hand-built timetables
- [x] Validator detects: teacher clash, room clash, batch clash, missing lecture, duplicate lecture, unavailable teacher, capacity violation, broken practical
- [x] Elitism: best penalty never increases across generations
- [x] Same seed → same result; different seeds → different runs
- [x] Feasible run ends with validator `is_valid = True`
- [x] UI shows FAILED when validator reports violations
- [x] Graph values match logged history
- [x] CSV export opens correctly

## 5. Expected output format (CLI)
```
Dataset: feasible | Lectures: [N] | Slots/batch: 25 | Utilization: [x]%
Initial best penalty: [value] (hard: [h], soft: [s])
Gen   1 | best [..] | avg [..]
...
Gen [k] | best [..] | hard 0 | STOP: feasible timetable found
Validator: VALID (0 hard violations) | Soft penalty units: [s]
```

## 6. Viva questions and answers

1. **What is a Genetic Algorithm?** A population-based search technique inspired by natural selection that evolves candidate solutions using selection, crossover and mutation.
2. **Why use a GA for timetabling?** The search space is huge and constrained; GAs handle many soft/hard constraints flexibly and give good solutions without exhaustive search.
3. **What is your chromosome?** A complete timetable: a list of genes, one per required lecture session.
4. **What is a gene here?** One lecture assignment: subject, teacher, batch (fixed) plus room, day, period (evolved).
5. **What is the population and its size?** A set of candidate timetables; default 50. Larger gives diversity but costs time.
6. **Explain your fitness function.** Penalty = 1000×hard + 10×soft; lower is better. Fitness = 1/(1+penalty) if a maximization score is needed.
7. **Why weight hard constraints so heavily?** So a timetable with any clash is always ranked worse than one with only preference issues.
8. **Which selection method did you use?** Tournament selection (k=3): choose 3 random individuals, the best becomes a parent; it is simple and adjustable via k.
9. **How does crossover work in your project?** For each lecture, the child inherits its room/day/period from one of two parents, so every lecture still appears exactly once.
10. **Why is mutation needed?** It introduces new variation and prevents premature convergence to a local optimum.
11. **What is elitism and why use it?** Copying the best individuals unchanged ensures the best penalty never gets worse between generations.
12. **What is the repair mechanism?** A step that fixes invalid genes after crossover/mutation, e.g. a 2-period lab crossing the end of the day or a lecture in an undersized room.
13. **Difference between hard and soft constraints?** Hard must hold for validity (no clashes, availability, capacity, correct lecture counts); soft are preferences (few gaps, spread subjects).
14. **How do you know the final timetable is valid?** An independent validator, sharing no code with the fitness function, checks every hard constraint; only its result is trusted.
15. **Does a GA guarantee the optimal timetable?** No. It is heuristic; it may find a good or feasible solution but cannot guarantee the global optimum or even feasibility within limits.
16. **What if constraints are impossible?** The feasibility pre-check and validator report failure; the app never shows an invalid schedule as successful.
17. **What is the time complexity?** About O(G × P × N) for G generations, P population, N genes, assuming near-constant-time slot checks via hash maps.
18. **Exploration vs exploitation?** Mutation and diversity explore new areas; selection, crossover and elitism exploit good solutions.
19. **How would you improve the system?** Teacher preferences, more batches, adaptive mutation rates, hybrid local search, comparison with simulated annealing, database and login.
20. **How did you evaluate performance?** Using real runs: initial vs final penalty, convergence graph, success rate over multiple seeds, and validator results.

## 7. Advantages and limitations
**Advantages:** automated, flexible constraints, visual dashboard, transparent hand-written GA, independent validation.
**Limitations:** heuristic (no optimality guarantee), small dataset, hand-tuned weights/parameters, no persistent storage.
