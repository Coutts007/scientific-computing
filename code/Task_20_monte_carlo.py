"""
BIT 2118 Scientific Computing - Task S (Task 20)
Random Numbers and Monte Carlo Methods
=========================================================================

Program file : code/Task_20_monte_carlo.py
Run with     : python code/Task_20_monte_carlo.py
Output graphs: figures/Task_20_monte_carlo.png
               figures/Task_20_project_risk.png  (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Monte Carlo methods estimate quantities by repeated random sampling.
If an event has probability p, then in N independent simulations the
observed fraction p_hat = (number of times it happened) / N approaches p
as N grows (law of large numbers).

Accuracy: p_hat has standard error
        SE = sqrt( p_hat (1 - p_hat) / N ),
and an approximate 95 % confidence interval is p_hat +/- 1.96 SE. The
error shrinks like 1/sqrt(N): 100 times more simulations give only 10
times more accuracy - but this rate does NOT depend on how complicated
the model is, which is why Monte Carlo is used when no formula exists.

Classic example: estimate pi by sampling points uniformly in the unit
square and counting the fraction inside the quarter circle x^2+y^2 <= 1:
        pi ~ 4 x (inside / total).

Pseudo-random numbers come from a deterministic generator started from a
seed; fixing the seed (default_rng(42)) makes results repeatable.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
Estimate the probability that a network server becomes overloaded when
incoming demand is uncertain: generate many random demand scenarios,
check whether each exceeds capacity, and divide the number of overloads
by the number of simulations.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    capacity    = 1000 requests per second
    demand      ~ Normal(mean 850, standard deviation 120)
    N           = 100 000 simulations, seed 42
Operations:
    rng.normal (random demand), comparison demand > capacity (Boolean
    array), mean of Booleans = fraction overloaded, cumulative sum for
    the running estimate, standard error / confidence interval, and the
    exact normal-distribution answer 1 - Phi((1000 - 850)/120) for
    checking (scipy.stats.norm).

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def overload_estimate(rng, capacity, mean, std, n):
    """Monte Carlo estimate of P(demand > capacity) and its standard error."""
    demand = rng.normal(mean, std, n)
    overloaded = demand > capacity
    p_hat = overloaded.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat, se, overloaded


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

rng = np.random.default_rng(42)

capacity = 1000.0
mean_demand = 850.0
std_demand = 120.0
N = 100_000

# Random demand scenarios
demand = rng.normal(mean_demand, std_demand, N)

overloaded = demand > capacity
overload_probability = overloaded.mean()

# Standard error and 95 % confidence interval of the estimate
standard_error = np.sqrt(overload_probability * (1 - overload_probability)
                         / N)
ci_low = overload_probability - 1.96 * standard_error
ci_high = overload_probability + 1.96 * standard_error

# Exact answer (possible here because demand is normal) for checking
z = (capacity - mean_demand) / std_demand
exact_probability = 1 - norm.cdf(z)

print("=" * 65)
print("PART 1 - Original solution: server-overload probability")
print("=" * 65)
print(f"Estimated overload probability: {overload_probability:.4%}")
print(f"Number of overload cases: {overloaded.sum()} out of {N}")
print(f"Standard error: {standard_error:.4%}")
print(f"95 % confidence interval: {ci_low:.3%} to {ci_high:.3%}")
print(f"Exact probability 1 - Phi({z:.2f}) = {exact_probability:.4%}")
print("Exact value inside the confidence interval:",
      ci_low <= exact_probability <= ci_high)

# Running probability to show convergence
running_probability = np.cumsum(overloaded) / np.arange(1, N + 1)

# Classic example from the concept section: estimating pi
points = rng.random((N, 2))
inside = (points**2).sum(axis=1) <= 1.0
pi_estimate = 4 * inside.mean()
print(f"\npi estimate from {N:,} random points: {pi_estimate:.5f} "
      f"(error {abs(pi_estimate - np.pi):.5f})")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: server capacity upgraded from 1000 to 1100 requests/s.
#   Expected: overload needs demand more than 2.08 standard deviations
#   above the mean instead of 1.25, so the probability falls from about
#   10.6 % to about 1.9 %. Rarer events need more simulations for the same
#   RELATIVE accuracy.
#
# Change 2: the number of simulations N is varied from 100 to 1 000 000
#   (several independent repeats for each N).
#   Expected: the spread of the estimates shrinks like 1/sqrt(N): each
#   100x increase in N makes the typical error about 10x smaller.
#
# Change 3: the random seed is changed (42 -> 7, 2024).
#   Expected: slightly different estimates, all within the confidence
#   interval - the answer is random, but its uncertainty is predictable.

p_up, se_up, _ = overload_estimate(np.random.default_rng(42), 1100.0,
                                   mean_demand, std_demand, N)
exact_up = 1 - norm.cdf((1100 - mean_demand) / std_demand)

sizes = np.array([100, 1_000, 10_000, 100_000, 1_000_000])
REPEATS = 30
rng_study = np.random.default_rng(123)
rms_errors = []
for n in sizes:
    estimates = [overload_estimate(rng_study, capacity, mean_demand,
                                   std_demand, n)[0] for _ in range(REPEATS)]
    rms_errors.append(np.sqrt(np.mean((np.array(estimates)
                                       - exact_probability)**2)))
rms_errors = np.array(rms_errors)
theory_se = np.sqrt(exact_probability * (1 - exact_probability) / sizes)

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print("Change 1 - capacity 1000 -> 1100:")
print(f"  estimate {p_up:.4%} +/- {1.96 * se_up:.3%}  (exact {exact_up:.4%})")
print(f"  relative standard error {se_up / p_up:.1%} vs "
      f"{standard_error / overload_probability:.1%} before")
print()
print(f"Change 2 - number of simulations ({REPEATS} repeats each):")
print(f"{'N':>10}{'RMS error':>12}{'theory SE':>12}")
for n, e, t in zip(sizes, rms_errors, theory_se):
    print(f"{n:>10,}{e:>12.5f}{t:>12.5f}")
print()
print("Change 3 - different seeds (N = 100,000):")
for seed in (42, 7, 2024):
    p_s, se_s, _ = overload_estimate(np.random.default_rng(seed), capacity,
                                     mean_demand, std_demand, N)
    print(f"  seed {seed:>4}: {p_s:.4%} +/- {1.96 * se_s:.3%}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Left graph : running estimate against number of simulations (log x),
#              with the exact value and the +/-1.96 SE band.
# Right graph: RMS error against N on log-log axes with slope -1/2.

indices = np.unique(
    np.logspace(1, np.log10(N), 200).astype(int)
) - 1

fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(12, 4.5))
n_axis = indices + 1
band = 1.96 * np.sqrt(exact_probability * (1 - exact_probability) / n_axis)
ax_r.fill_between(n_axis, exact_probability - band, exact_probability + band,
                  color="gray", alpha=0.2, label="Exact +/- 1.96 SE")
ax_r.semilogx(
    n_axis,
    running_probability[indices],
    label="Running estimate"
)
ax_r.axhline(exact_probability, color="black", linestyle="--",
             label=f"Exact {exact_probability:.2%}")
ax_r.set_ylim(0, 0.25)
ax_r.set_xlabel("Number of simulations")
ax_r.set_ylabel("Estimated overload probability")
ax_r.set_title("Monte Carlo estimate of server-overload risk")
ax_r.legend(fontsize=8)
ax_r.grid(True)

ax_e.loglog(sizes, rms_errors, "o-", label="Observed RMS error")
ax_e.loglog(sizes, theory_se, "k--", label="Theory sqrt(p(1-p)/N)")
ax_e.set_xlabel("Number of simulations N")
ax_e.set_ylabel("Error in probability")
ax_e.set_title("Error shrinks like 1/sqrt(N)")
ax_e.legend()
ax_e.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_20_monte_carlo.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The server is overloaded in about {overload_probability:.1%} of "
      "demand scenarios - roughly")
print("  1 period in 10. With 95 % confidence the true value lies between "
      f"{ci_low:.2%} and {ci_high:.2%}.")
print("  That is a high risk for a production service; extra capacity or")
print("  auto-scaling is justified.")
print(f"* Upgrading to 1100 req/s cuts the risk to about {p_up:.1%}.")
print("* The simulation agrees with the exact normal-distribution answer,")
print("  which validates the method; Monte Carlo becomes essential when the")
print("  model is too complex for a formula (see Part 4).")
print("* Graph (left): the estimate jumps around for small N, then settles")
print("  inside the grey band around the exact value, which narrows as N")
print("  grows. Graph (right): errors fall along a line of slope -1/2,")
print("  matching the theory: 100x more simulations -> 10x smaller error.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: schedule risk of a software project.
#
# Problem:
#   A team must deliver a mobile banking app. The client wants it in 60
#   working days. Each task's duration is uncertain. What is the
#   probability of finishing on time, and what deadline can be promised
#   with 90 % confidence?
#
# Project structure (task: optimistic, most likely, pessimistic days):
#   1. Requirements          :  5,  8, 14
#   then in PARALLEL:
#   2a. Backend API           : 15, 22, 35
#   2b. Mobile front-end      : 12, 18, 30
#   then (needs both 2a and 2b finished):
#   3. Integration & testing  : 10, 14, 25
#   4. Deployment & review    :  3,  5, 10
#   Total = T1 + max(T2a, T2b) + T3 + T4
#
# Assumptions:
#   * Each duration follows a triangular distribution (min, mode, max),
#     a common choice when only expert three-point estimates exist.
#   * Task durations are independent.
#   * N = 200 000 simulated projects, seed 2025.
#
# Why Monte Carlo: the max() of two random parallel tasks and the skewed
# distributions mean there is no simple formula for the total. Simply
# adding the "most likely" values (8 + 22 + 14 + 5 = 49 days) ignores
# both the skew and the parallel-path effect and is over-optimistic.

DEADLINE = 60
N_PROJECTS = 200_000
tasks = {
    "Requirements": (5, 8, 14),
    "Backend API": (15, 22, 35),
    "Mobile front-end": (12, 18, 30),
    "Integration & testing": (10, 14, 25),
    "Deployment & review": (3, 5, 10),
}

rng_proj = np.random.default_rng(2025)
durations = {name: rng_proj.triangular(*abc, size=N_PROJECTS)
             for name, abc in tasks.items()}

total = (durations["Requirements"]
         + np.maximum(durations["Backend API"],
                      durations["Mobile front-end"])
         + durations["Integration & testing"]
         + durations["Deployment & review"])

naive_plan = sum(tasks[t][1] for t in ("Requirements", "Backend API",
                                       "Integration & testing",
                                       "Deployment & review"))
p_on_time = (total <= DEADLINE).mean()
se_on_time = np.sqrt(p_on_time * (1 - p_on_time) / N_PROJECTS)
p50, p90 = np.percentile(total, [50, 90])
frontend_critical = (durations["Mobile front-end"]
                     > durations["Backend API"]).mean()

print()
print("=" * 65)
print("PART 4 - Modified case: software project schedule risk")
print("=" * 65)
print(f"Plan using most-likely durations : {naive_plan} days")
print(f"Mean simulated duration          : {total.mean():.1f} days")
print(f"Median (50 % chance)             : {p50:.1f} days")
print(f"90 % confidence finish           : {p90:.1f} days")
print(f"P(finish within {DEADLINE} days)       : {p_on_time:.2%} "
      f"(+/- {1.96 * se_on_time:.2%})")
print(f"P(finish within {naive_plan} days)       : "
      f"{(total <= naive_plan).mean():.2%}")
print(f"Front-end is the longer parallel task in {frontend_critical:.1%} "
      "of runs")

fig2, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_h.hist(total, bins=100, density=True, color="C0", alpha=0.7)
for value, label, colour in ((naive_plan, f"Naive plan {naive_plan} d", "C2"),
                             (DEADLINE, f"Deadline {DEADLINE} d", "C3"),
                             (p90, f"90 % point {p90:.0f} d", "k")):
    ax_h.axvline(value, color=colour, linestyle="--", label=label)
ax_h.set_xlabel("Total project duration (days)")
ax_h.set_ylabel("Probability density")
ax_h.set_title("Distribution of project duration")
ax_h.legend(fontsize=8)
ax_h.grid(True, alpha=0.4)

sorted_total = np.sort(total)
cumulative = np.arange(1, N_PROJECTS + 1) / N_PROJECTS
ax_c.plot(sorted_total, cumulative)
ax_c.axvline(DEADLINE, color="C3", linestyle="--")
ax_c.axhline(p_on_time, color="C3", linestyle=":")
ax_c.annotate(f"{p_on_time:.0%} on time", (DEADLINE, p_on_time),
              xytext=(8, -15), textcoords="offset points", fontsize=8)
ax_c.set_xlabel("Deadline (days)")
ax_c.set_ylabel("Probability of finishing by then")
ax_c.set_title("Chance of meeting any deadline")
ax_c.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_20_project_risk.png", dpi=120)

print()
print("Interpretation:")
print(f"* The 'most likely' plan of {naive_plan} days is met only "
      f"{(total <= naive_plan).mean():.0%} of the time: the task")
print("  distributions are skewed towards delays, and with two parallel")
print("  tasks the project waits for whichever is slower.")
print(f"* There is about a {p_on_time:.0%} chance of meeting the {DEADLINE}-day "
      "deadline. To be 90 % sure,")
print(f"  the team should promise about {np.ceil(p90):.0f} days or reduce scope.")
print(f"* The front-end finishes later than the backend in {frontend_critical:.0%}"
      " of runs, so")
print("  both teams need monitoring - the critical path is itself random.")
print(f"* Graph (left): the outcomes centre near {p50:.0f} days, well to the "
      f"right of the")
print(f"  naive {naive_plan}-day plan; only the small area left of the green "
      "line is on time")
print("  for that plan. Graph (right): read off the probability of finishing")
print("  by any deadline.")

plt.show()
