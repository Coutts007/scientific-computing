"""
BIT 2118 Scientific Computing - Task R (Task 19)
Numerical Optimization
=========================================================================

Program file : code/Task_19_optimization.py
Run with     : python code/Task_19_optimization.py
Output graphs: figures/Task_19_optimization.png
               figures/Task_19_autoscaling.png   (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Optimization finds the inputs x that minimise (or maximise) an
objective function f(x), possibly subject to constraints:
        minimise f(x)
        subject to g_i(x) >= 0   (inequality constraints)
                   bounds such as x >= 0
An optimum x* has an objective value no larger than any nearby feasible
point. Maximising f is the same as minimising -f.

Linear programming (LP): objective and constraints are linear. The
feasible region is a polygon and the optimum lies at a corner (vertex);
cost contours are parallel straight lines.

Nonlinear optimization: at an unconstrained smooth minimum f'(x*) = 0.
With constraints, the optimum is either inside the feasible region
(constraint inactive) or on its boundary (constraint "binding").

scipy.optimize.minimize with method "SLSQP" (Sequential Least SQuares
Programming) handles bounds and equality/inequality constraints by
solving a sequence of quadratic approximations.

Integer requirements (whole servers) make the problem harder: rounding
the continuous answer is not always optimal, so nearby integer points
are searched (or an integer solver such as scipy.optimize.milp is used).

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A cloud service chooses how many units of two server types to run so
that operating cost is minimal while total processing capacity is at
least a required level.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Unknowns: x = units of Type A server, y = units of Type B server
Inputs:
    cost per unit      : A = 12, B = 18
    capacity per unit  : A = 100, B = 170
    required capacity  : at least 1000
    starting guess     : (5, 3)
Problem:
    minimise   12x + 18y
    subject to 100x + 170y >= 1000,  x >= 0,  y >= 0
Operations:
    minimize(method="SLSQP"), brute-force search over integer pairs,
    scipy.optimize.milp as an exact integer check.

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import (Bounds, LinearConstraint, milp, minimize,
                            minimize_scalar)

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def solve_allocation(cost_a=12.0, cost_b=18.0, cap_a=100.0, cap_b=170.0,
                     required=1000.0):
    """Continuous optimum (SLSQP) and best integer allocation.

    Returns (SLSQP result, (cost, a, b, capacity) of the best integer pair).
    """
    def cost(z):
        x, y = z
        return cost_a * x + cost_b * y

    def capacity_constraint(z):
        x, y = z
        return cap_a * x + cap_b * y - required      # must be >= 0

    constraints = {
        "type": "ineq",
        "fun": capacity_constraint
    }
    bounds = [(0, None), (0, None)]

    solution = minimize(
        cost,
        x0=[5, 3],
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    # If whole servers are required, test nearby integer combinations.
    best_integer = None
    for a in range(0, 20):
        for b in range(0, 20):
            capacity = cap_a * a + cap_b * b
            if capacity >= required:
                total_cost = cost_a * a + cost_b * b
                if best_integer is None or total_cost < best_integer[0]:
                    best_integer = (total_cost, a, b, capacity)
    return solution, best_integer


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

solution, best_integer = solve_allocation()
x, y = solution.x

# Exact mixed-integer check with scipy.optimize.milp:
# minimise c @ z subject to 100 a + 170 b >= 1000, a, b integer >= 0
milp_result = milp(c=[12, 18],
                   constraints=LinearConstraint([[100, 170]], lb=1000),
                   integrality=[1, 1], bounds=Bounds(0, np.inf))

print("=" * 65)
print("PART 1 - Original solution: cheapest server mix")
print("=" * 65)
print("Optimization success:", solution.success)
print(f"Type A allocation: {x:.3f}")
print(f"Type B allocation: {y:.3f}")
print(f"Total capacity: {100*x + 170*y:.2f}")
print(f"Minimum planning cost: {solution.fun:.2f}")
print()
print("Cost per unit of capacity: A = "
      f"{12 / 100:.4f}, B = {18 / 170:.4f}  -> B is cheaper per unit")
print("Best nearby integer solution (cost, A, B, capacity):", best_integer)
print(f"milp check: A = {abs(milp_result.x[0]):.0f}, "
      f"B = {abs(milp_result.x[1]):.0f}, "
      f"cost = {milp_result.fun:.0f}")
print(f"Rounding the continuous answer up (A = 0, B = {np.ceil(y):.0f}) "
      f"costs {18 * np.ceil(y):.0f}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the price of Type B rises from 18 to 21.
#   Expected: B's cost per unit of capacity becomes 21/170 = 0.1235,
#   now MORE than A's 0.12, so the optimum jumps to the other corner of
#   the feasible region: all Type A (x = 10, y = 0, cost 120). In linear
#   problems small price changes can switch the answer completely.
#
# Change 2: the required capacity rises from 1000 to 1500 (original
#   prices).
#   Expected: the optimal mix keeps the same shape (only Type B, as it is
#   still cheaper per unit) but scales up by 1.5: y = 8.82, cost 158.8.
#   The integer answer must round B up to 9 units (capacity 1530).

sol_b21, int_b21 = solve_allocation(cost_b=21.0)
sol_1500, int_1500 = solve_allocation(required=1500.0)

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print(f"{'Case':<26}{'A':>8}{'B':>8}{'Cost':>9}   Best integer (cost,A,B,cap)")
for label, sol, best in (("Original", solution, best_integer),
                         ("Change 1: B costs 21", sol_b21, int_b21),
                         ("Change 2: need 1500", sol_1500, int_1500)):
    print(f"{label:<26}{sol.x[0]:>8.3f}{sol.x[1]:>8.3f}{sol.fun:>9.2f}"
          f"   {best}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Graph: the feasible region (shaded), cost contour lines, the
# continuous optimum, all feasible integer points and the best one.

xs = np.linspace(0, 12, 300)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.fill_between(xs, np.maximum((1000 - 100 * xs) / 170, 0), 8,
                color="C2", alpha=0.12, label="Feasible (capacity >= 1000)")
ax.plot(xs, (1000 - 100 * xs) / 170, "C2", label="100x + 170y = 1000")
for level in (90, 105.88, 120, 140):
    ax.plot(xs, (level - 12 * xs) / 18, "k:", linewidth=0.8)
    ax.annotate(f"cost {level:g}", (0.3, (level - 12 * 0.3) / 18 + 0.1),
                fontsize=7)
grid_a, grid_b = np.meshgrid(np.arange(0, 12), np.arange(0, 8))
feasible = 100 * grid_a + 170 * grid_b >= 1000
ax.scatter(grid_a[feasible], grid_b[feasible], s=12, color="gray",
           label="Feasible integer mixes")
ax.plot(*solution.x, "r*", markersize=15, label="Continuous optimum")
ax.plot(best_integer[1], best_integer[2], "bs", markersize=9,
        fillstyle="none", mew=2, label="Best integer mix")
ax.plot(*sol_b21.x, "m^", markersize=10, label="Optimum if B costs 21")
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.set_xlabel("Type A servers (x)")
ax.set_ylabel("Type B servers (y)")
ax.set_title("Minimum-cost server allocation")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_19_optimization.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The cheapest plan uses only Type B: {y:.2f} units, cost "
      f"{solution.fun:.2f}, providing exactly")
print("  the required 1000 capacity. B is chosen because it delivers")
print("  capacity more cheaply (0.106 per unit vs 0.120 for A).")
print("* The capacity constraint is binding: any cheaper plan would fall")
print("  below 1000, so the optimum sits on the constraint line.")
print(f"* With whole servers the best mix costs {best_integer[0]:.0f} "
      f"(A = {best_integer[1]}, B = {best_integer[2]}), slightly more than")
print("  the continuous optimum. milp confirms this integer answer.")
print("* Raising B's price to 21 flips the answer to all Type A - in linear")
print("  problems the optimum jumps between corners rather than moving")
print("  gradually.")
print("* Graph: the green region contains every mix with enough capacity.")
print("  Dotted lines are equal-cost lines; the optimum is where the lowest")
print("  cost line just touches the region - the corner on the y-axis (red")
print("  star). The square marks the cheapest whole-server mix.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: auto-scaling web servers - balancing server cost against
# customer waiting time (a NONLINEAR optimization).
#
# Problem:
#   An online shop receives lambda = 900 requests per second at peak.
#   Each server handles mu = 100 requests per second. More servers cost
#   more money, but fewer servers make customers wait longer, which loses
#   sales. How many servers minimise the total hourly cost?
#
# Model:
#   * Requests are spread evenly over n servers; each behaves like an
#     M/M/1 queue, with average response time
#           T(n) = 1 / (mu - lambda / n)    seconds,  needs n > lambda/mu
#   * Hourly cost = server cost + waiting cost:
#           f(n) = c_s n + c_w (lambda x 3600) T(n)
#     c_s = KES 50 per server-hour,
#     c_w = KES 0.02 per second a request waits (estimated lost sales).
#   * Service-level agreement (SLA) variant: average response time must be
#     at most 15 ms, i.e. constraint 0.015 - T(n) >= 0.
#
# Assumptions:
#   * Random (Poisson) arrivals and service times (M/M/1 per server).
#   * n is treated as continuous first, then nearby whole numbers are
#     checked, as in Part 1.
#
# Methods: minimize_scalar (bounded) for the unconstrained problem,
# minimize with SLSQP for the SLA-constrained problem.

LAMBDA, MU = 900.0, 100.0
COST_SERVER, COST_WAIT = 50.0, 0.02
SLA_SECONDS = 0.015
n_min = LAMBDA / MU                       # below this the queue explodes


def response_time(n):
    return 1.0 / (MU - LAMBDA / n)


def hourly_cost(n):
    """Total hourly cost (KES) of running n servers."""
    return COST_SERVER * n + COST_WAIT * LAMBDA * 3600 * response_time(n)


unconstrained = minimize_scalar(hourly_cost, bounds=(n_min + 0.01, 60),
                                method="bounded")
n_star = unconstrained.x

sla = minimize(lambda z: hourly_cost(z[0]), x0=[30.0], method="SLSQP",
               bounds=[(n_min + 0.01, 60)],
               constraints={"type": "ineq",
                            "fun": lambda z: SLA_SECONDS - response_time(z[0])})
n_sla = sla.x[0]

# Best whole numbers of servers
candidates = np.arange(int(n_min) + 1, 61)
costs = np.array([hourly_cost(n) for n in candidates])
best_int = candidates[np.argmin(costs)]
# A tiny tolerance is needed: 27 servers gives exactly 15 ms, but
# floating-point round-off can make it appear a hair above the limit.
sla_ok = candidates[[response_time(n) <= SLA_SECONDS + 1e-12
                     for n in candidates]]
sla_costs = np.array([hourly_cost(n) for n in sla_ok])
best_int_sla = sla_ok[np.argmin(sla_costs)]

print()
print("=" * 65)
print("PART 4 - Modified case: auto-scaling web servers")
print("=" * 65)
print(f"Minimum servers for a stable queue: n > {n_min:.0f}")
print(f"Unconstrained optimum: n = {n_star:.2f}, cost KES "
      f"{unconstrained.fun:,.0f}/hour, response {response_time(n_star) * 1000:.1f} ms")
print(f"  best whole number: {best_int} servers, cost KES "
      f"{hourly_cost(best_int):,.0f}/hour, response "
      f"{response_time(best_int) * 1000:.1f} ms")
print(f"With SLA (<= {SLA_SECONDS * 1000:.0f} ms): n = {n_sla:.2f}, cost KES "
      f"{hourly_cost(n_sla):,.0f}/hour, success = {sla.success}")
print(f"  best whole number: {best_int_sla} servers, cost KES "
      f"{hourly_cost(best_int_sla):,.0f}/hour, response "
      f"{response_time(best_int_sla) * 1000:.1f} ms")
print(f"  extra cost of the SLA: KES "
      f"{hourly_cost(best_int_sla) - hourly_cost(best_int):,.0f}/hour")

n_plot = np.linspace(n_min + 0.5, 45, 400)
server_part = COST_SERVER * n_plot
wait_part = COST_WAIT * LAMBDA * 3600 * response_time(n_plot)
fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
ax2.plot(n_plot, server_part, "--", label="Server cost")
ax2.plot(n_plot, wait_part, "--", label="Waiting cost")
ax2.plot(n_plot, server_part + wait_part, "k", linewidth=2,
         label="Total cost")
ax2.plot(n_star, unconstrained.fun, "r*", markersize=14,
         label=f"Optimum n = {n_star:.1f}")
ax2.axvspan(n_min, LAMBDA / (MU - 1 / SLA_SECONDS), color="red", alpha=0.08,
            label="Violates 15 ms SLA")
ax2.plot(n_sla, hourly_cost(n_sla), "bs", markersize=9,
         label=f"SLA optimum n = {n_sla:.1f}")
ax2.set_ylim(0, 6000)
ax2.set_xlabel("Number of servers n")
ax2.set_ylabel("Cost (KES per hour)")
ax2.set_title("Server cost vs customer waiting cost")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_19_autoscaling.png", dpi=120)

print()
print("Interpretation:")
print(f"* About {best_int} servers minimise total cost: fewer servers save "
      "server fees but")
print("  waiting time (and lost sales) rises steeply as servers approach")
print(f"  full load; more servers cost more than the waiting they save.")
print(f"* At the optimum each server is only {LAMBDA / best_int / MU:.0%} "
      "busy - running servers")
print("  near 100 % utilisation looks efficient but is very expensive in")
print("  waiting time.")
print(f"* The 15 ms SLA is a BINDING constraint: it forces {best_int_sla} "
      f"servers, above the cost")
print("  optimum, and the solution sits exactly on the constraint boundary.")
print("* Graph: server cost rises in a straight line, waiting cost falls in")
print("  a curve; the total is U-shaped with its minimum at the red star.")
print("  The red band is excluded by the SLA, pushing the choice right to")
print("  the blue square.")

plt.show()
