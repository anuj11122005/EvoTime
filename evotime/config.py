"""
config.py — Central Configuration and Hyperparameters for EvoTime.

This module defines all constants and hyperparameter settings for the
timetable generation system. It includes college schedule bounds (days,
periods per day), Genetic Algorithm parameters (population size, mutation rate,
elitism), and penalty weights for hard and soft constraints.

Why keep this in a separate file?
In a Genetic Algorithm project, hyperparameter tuning is common. Keeping all
parameters here allows us to experiment with different mutation rates,
crossover rates, or constraint penalties without touching the algorithm logic.
"""

from dataclasses import dataclass, field
from typing import List


# ============================================================================
# College Schedule Structure
# ============================================================================

# Working days in a typical college week (indices 0 to 4 correspond to Mon to Fri)
DAYS: List[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
NUM_DAYS: int = len(DAYS)  # 5 working days

# Periods available per day (indices 0 to 4 correspond to Periods 1 to 5)
PERIODS_PER_DAY: int = 5

# Standard daily period times (1-hour each, except lunch break)
PERIOD_TIMES: List[str] = [
    "09:00 - 10:00",
    "10:00 - 11:00",
    "11:15 - 12:15",
    "12:15 - 01:15",
    "02:00 - 03:00",
]

# Human-readable labels for each daily period, dynamically generated from PERIOD_TIMES
PERIOD_LABELS: List[str] = [
    f"Period {i + 1} ({t})" for i, t in enumerate(PERIOD_TIMES)
]

# Total slots available per week per batch (5 days * 5 periods = 25 slots)
TOTAL_SLOTS_PER_WEEK: int = NUM_DAYS * PERIODS_PER_DAY


# ============================================================================
# Genetic Algorithm Default Hyperparameters
# ============================================================================

DEFAULT_POPULATION_SIZE: int = 50       # Number of candidate timetables in each generation
DEFAULT_MAX_GENERATIONS: int = 250      # Maximum number of evolutionary cycles before stopping (tuned from 200)
DEFAULT_CROSSOVER_RATE: float = 0.8     # Probability of mating two parent timetables
DEFAULT_MUTATION_RATE: float = 0.08     # Probability of mutating each individual gene
DEFAULT_ELITISM_COUNT: int = 2          # Top best timetables carried forward unchanged
DEFAULT_TOURNAMENT_K: int = 4           # Number of candidates competing in tournament selection (tuned from 3)
DEFAULT_RANDOM_SEED: int = 42           # Fixed seed for reproducibility of experimental runs


# ============================================================================
# Penalty Weights (Fitness Evaluation) & Termination
# ============================================================================

# Hard constraint violation penalty:
# Clashes are fatal in a real timetable (e.g., Professor X cannot be in two rooms at once).
# High penalty ensures any schedule with 1 hard clash is ranked worse than one with
# dozens of minor soft preference issues.
HARD_PENALTY_WEIGHT: int = 1000

# Base multiplier for soft constraint violations
SOFT_PENALTY_WEIGHT: int = 10

# Default early termination scalar penalty threshold:
# GA stops early if hard_total == 0 and penalty <= TERMINATION_PENALTY_THRESHOLD
TERMINATION_PENALTY_THRESHOLD: float = 300.0

# Individual weights for specific soft constraints (relative to each other):
# 1. Teacher has > 3 consecutive lectures without a break (causes teacher fatigue)
WEIGHT_TEACHER_CONSECUTIVE: int = 1

# 2. Gaps (idle free periods) between classes in a student batch's day
WEIGHT_BATCH_GAPS: int = 1

# 3. Teacher daily workload variance (e.g. 5 lectures on Monday and 0 on Tuesday)
WEIGHT_TEACHER_WORKLOAD_VARIANCE: int = 1

# 4. Undesirable patterns (e.g., a batch coming to college for only 1 isolated lecture in a day,
#    or 2-period practicals scheduled at the very last period of the day)
WEIGHT_UNDESIRABLE_PATTERNS: int = 1


# ============================================================================
# Config Dataclass (for runtime customization from UI or CLI)
# ============================================================================

@dataclass
class GAConfig:
    """
    Holds runtime configuration parameters for the Genetic Algorithm.
    Allows easy tweaking from Streamlit sidebar controls or CLI arguments.
    """
    population_size: int = DEFAULT_POPULATION_SIZE
    max_generations: int = DEFAULT_MAX_GENERATIONS
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    mutation_rate: float = DEFAULT_MUTATION_RATE
    elitism_count: int = DEFAULT_ELITISM_COUNT
    tournament_k: int = DEFAULT_TOURNAMENT_K
    seed: int = DEFAULT_RANDOM_SEED

    hard_weight: int = HARD_PENALTY_WEIGHT
    soft_weight: int = SOFT_PENALTY_WEIGHT
    termination_penalty_threshold: float = TERMINATION_PENALTY_THRESHOLD

    weight_teacher_consecutive: int = WEIGHT_TEACHER_CONSECUTIVE
    weight_batch_gaps: int = WEIGHT_BATCH_GAPS
    weight_teacher_workload_variance: int = WEIGHT_TEACHER_WORKLOAD_VARIANCE
    weight_undesirable_patterns: int = WEIGHT_UNDESIRABLE_PATTERNS
