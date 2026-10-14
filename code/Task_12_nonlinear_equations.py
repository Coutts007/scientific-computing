"""
BIT 2118 Scientific Computing - Task K (Task 12)
Polynomial and Nonlinear Equation Solving
=========================================================================

Program file : code/Task_12_nonlinear_equations.py
Run with     : python code/Task_12_nonlinear_equations.py
Output graphs: figures/Task_12_nonlinear_equations.png
               figures/Task_12_trilateration.png  (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
A polynomial equation has the form a_n x^n + ... + a_1 x + a_0 = 0. A
degree-n polynomial has exactly n roots (counting complex ones and
repeats); np.roots finds them all as the eigenvalues of the polynomial's
"companion matrix".

A nonlinear SYSTEM has several equations F(z) = 0 that are not all
linear in the unknowns z = (x, y, ...). Solvers such as
scipy.optimize.root use multidimensional Newton-type methods:
        J(z_k) dz = -F(z_k),     z_{k+1} = z_k + dz
where J is the Jacobian matrix of partial derivatives. A nonlinear system
can have zero, one or many solutions; the solver returns the one its
starting guess leads to, so several guesses are needed to find them all.
Results are checked by evaluating the residuals F(z), which should be
close to zero.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
Find the intersection points of a circle and a parabola (two nonlinear
equations in x and y), using scipy.optimize.root from more than one
initial guess, and verify that the residuals are close to zero.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    Circle   : x^2 + y^2 = 4    (centre (0, 0), radius 2)
    Parabola : y = x^2 - 1
    Initial guesses: [1, 1] and [-1, 1]
Operations:
    residual functions F1 = x^2 + y^2 - 4, F2 = y - (x^2 - 1)
    scipy.optimize.root (hybrid Powell / Newton method)
    np.roots on the single-variable polynomial obtained by substitution
    (x^2 = y + 1  ->  y^2 + y - 3 = 0) as an independent check.

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, root

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

np.set_printoptions(precision=6, suppress=True)


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Find intersections of:
# Circle: x^2 + y^2 = r^2   (r = 2 in the original problem)
# Parabola: y = x^2 - 1

def equations(z, radius=2.0):
    """Residuals of the two curves; both are zero at an intersection."""
    x, y = z
    return [
        x**2 + y**2 - radius**2,
        y - (x**2 - 1)
    ]


def polynomial_check(radius):
    """Intersections from algebra: substitute x^2 = y + 1 into the circle.

    (y + 1) + y^2 = r^2  ->  y^2 + y + (1 - r^2) = 0.
    Each real root y with y + 1 >= 0 gives x = +/- sqrt(y + 1).
    """
    y_roots = np.roots([1.0, 1.0, 1.0 - radius**2])
    points = set()
    for y in y_roots:
        if abs(y.imag) < 1e-12 and y.real + 1 >= -1e-12:
            x = np.sqrt(max(y.real + 1, 0.0))
            points.add((round(float(x), 4), round(float(y.real), 4)))
            points.add((round(float(-x), 4), round(float(y.real), 4)))
    return y_roots, sorted(points)


print("=" * 65)
print("PART 1 - Original solution: circle radius 2 and parabola")
print("=" * 65)
for guess in ([1.0, 1.0], [-1.0, 1.0]):
    solution = root(equations, guess)

    print("Initial guess:", guess)
    print("Intersection:", solution.x)
    print("Residuals:", np.array(equations(solution.x)))
    print("Converged:", solution.success)
    print()

y_roots, exact_points = polynomial_check(2.0)
print("Polynomial check y^2 + y - 3 = 0, roots:", y_roots.real)
print("  y = -2.3028 would need x^2 = -1.30 (impossible), so only")
print("  the root y = 1.3028 gives real points:", exact_points)
print()
print("np.roots on the example x^2 - 5x + 6:", np.roots([1, -5, 6]))


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change: the circle radius is changed from 2 to 1 and to 0.5.
#
# Expected effect (from y^2 + y + (1 - r^2) = 0):
#   * r = 1   : y^2 + y = 0 -> y = 0 or y = -1. y = 0 gives (+/-1, 0);
#               y = -1 gives x = 0, the point (0, -1), where the circle
#               just TOUCHES the bottom of the parabola (tangent). Three
#               intersections in total. At a tangent point the Jacobian
#               is singular, which can make convergence slower or less
#               precise (here the solver still copes well).
#   * r = 0.5 : y^2 + y + 0.75 = 0 has discriminant 1 - 3 < 0, so there
#               are no real intersections. The solver cannot reach zero
#               residual and must report failure (or large residuals).
# The method is unchanged; only the geometry changes the answer.
#
# A third starting guess (0.1, -1.2), near the bottom of the parabola,
# is also tried. For r = 2 there is no intersection near it, and the
# solver gets stuck where the residual is smallest locally but not zero,
# showing that a bad guess can fail even when solutions exist.

print()
print("=" * 65)
print("PART 2 - Effect of changing the circle radius")
print("=" * 65)
guesses = ([1.0, 1.0], [-1.0, 1.0], [0.1, -1.2])
for radius in (2.0, 1.0, 0.5):
    print(f"Radius {radius}:")
    for guess in guesses:
        sol = root(equations, guess, args=(radius,))
        res = np.linalg.norm(equations(sol.x, radius))
        print(f"  guess {guess!s:<12} -> ({sol.x[0]: .6f}, {sol.x[1]: .6f})"
              f"  |residual| = {res:.1e}  success = {sol.success}")
    _, pts = polynomial_check(radius)
    print(f"  exact real intersections: {pts if pts else 'none'}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Graph: the parabola and the three circles, with the exact intersection
# points marked.

theta = np.linspace(0, 2 * np.pi, 400)
x_par = np.linspace(-2.5, 2.5, 400)

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot(x_par, x_par**2 - 1, "k", label="Parabola y = x^2 - 1")
for radius, colour in ((2.0, "C0"), (1.0, "C1"), (0.5, "C2")):
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=colour,
            label=f"Circle r = {radius}")
    _, pts = polynomial_check(radius)
    for px, py in pts:
        ax.plot(px, py, "o", color=colour, markersize=8,
                markeredgecolor="black")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Circle-parabola intersections")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_12_nonlinear_equations.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print("* For radius 2 the curves cross at (+/-1.5175, 1.3028). The two")
print("  initial guesses led the solver to the two different points; one")
print("  guess alone would have missed one of them.")
print("* Residuals of 1e-13 or smaller show both equations are satisfied;")
print("  the polynomial (substitution) method gives the same points.")
print("* For radius 2 the guess (0.1, -1.2) failed (success = False,")
print("  residual ~1): no intersection lies near it, so the solver stalled.")
print("  Always check success and residuals rather than trusting the output.")
print("* The algebra produced a second root y = -2.30 that is NOT a real")
print("  intersection (x would be imaginary): every root must be checked")
print("  against the original problem.")
print("* Radius 1 gives three points, including a tangent point at (0, -1).")
print("  Radius 0.5 gives none: the solver reports failure / non-zero")
print("  residual, which is the correct signal that no solution exists.")
print("* Graph: dots mark where each circle meets the parabola. The small")
print("  green circle lies entirely inside the parabola's curve and never")
print("  meets it.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: locating a phone from cell-tower distances (trilateration).
#
# Problem:
#   A phone measures its distance to cell towers from signal travel
#   time. Each distance d_i to a tower at (x_i, y_i) gives a circle
#       (x - x_i)^2 + (y - y_i)^2 = d_i^2.
#   The phone is where the circles meet - a nonlinear system, just like
#   the circle-parabola problem.
#
# Assumptions:
#   * Flat 2-D area, coordinates in km.
#   * Towers: T1 (0, 0), T2 (6, 0), T3 (2, 5).
#   * True phone position (3.0, 2.0), used only to generate distances
#     and to check the answer.
#   * Step 1: exact distances, two towers only.
#   * Step 2: real distances contain noise (about +/-50 m), so with three
#     towers the circles do not meet at exactly one point. There are
#     then 3 equations for 2 unknowns, so the best answer minimises the
#     sum of squared residuals (scipy.optimize.least_squares).
#
# Input data: tower positions, measured distances (computed below).
# Operations: root() with two guesses, least_squares() with 3 towers,
# residual norms, position error.

towers = np.array([[0.0, 0.0], [6.0, 0.0], [2.0, 5.0]])
true_position = np.array([3.0, 2.0])
exact_dist = np.linalg.norm(towers - true_position, axis=1)

rng = np.random.default_rng(7)
noisy_dist = exact_dist + rng.normal(0, 0.05, size=3)    # +/-50 m noise


def tower_residuals(z, tower_xy, dists):
    """Distance from z to each tower minus the measured distance."""
    return np.linalg.norm(tower_xy - z, axis=1) - dists


# Step 1: two towers, exact distances -> two possible positions
two_tower_solutions = []
for guess in ([3.0, 3.0], [3.0, -3.0]):
    sol = root(tower_residuals, guess, args=(towers[:2], exact_dist[:2]))
    two_tower_solutions.append(sol.x)

# Step 2: three towers, noisy distances -> least-squares best fit
fit = least_squares(tower_residuals, x0=[1.0, 1.0],
                    args=(towers, noisy_dist))
position_error_m = np.linalg.norm(fit.x - true_position) * 1000

print()
print("=" * 65)
print("PART 4 - Modified case: phone location by trilateration")
print("=" * 65)
print("Tower positions (km):\n", towers)
print("Exact distances (km):", exact_dist)
print("Noisy distances (km):", noisy_dist)
print()
print("Step 1 - two towers, exact distances:")
for s in two_tower_solutions:
    print(f"  solution ({s[0]:.4f}, {s[1]:.4f}) km, distance to T3 = "
          f"{np.linalg.norm(s - towers[2]):.4f} km "
          f"(measured {exact_dist[2]:.4f})")
print()
print("Step 2 - three towers, noisy distances (least squares):")
print(f"  estimated position ({fit.x[0]:.4f}, {fit.x[1]:.4f}) km")
print(f"  residuals (km): {fit.fun}")
print(f"  error from true position: {position_error_m:.1f} m")

# Graph for Part 4
fig2, ax2 = plt.subplots(figsize=(6.5, 6))
for k, ((tx, ty), d) in enumerate(zip(towers, noisy_dist), start=1):
    ax2.plot(tx + d * np.cos(theta), ty + d * np.sin(theta),
             label=f"Tower T{k} range")
    ax2.plot(tx, ty, "^", color=f"C{k - 1}", markersize=11,
             markeredgecolor="black")
for s in two_tower_solutions:
    ax2.plot(*s, "x", color="gray", markersize=10, mew=2)
ax2.plot(*true_position, "k*", markersize=14, label="True position")
ax2.plot(*fit.x, "ro", label="Least-squares estimate")
ax2.annotate("two-tower\nambiguity", two_tower_solutions[1],
             xytext=(10, 0), textcoords="offset points", fontsize=8)
ax2.set_aspect("equal")
ax2.set_xlim(-1.5, 7.5)
ax2.set_ylim(-3.5, 6.5)
ax2.set_xlabel("x (km)")
ax2.set_ylabel("y (km)")
ax2.set_title("Trilateration from three cell towers")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_12_trilateration.png", dpi=120)

print()
print("Interpretation:")
print("* With only two towers the two circles cross at TWO points,")
print("  (3, 2) and its mirror image (3, -2); the system really has two")
print("  solutions and different guesses find different ones. The third")
print("  tower's distance matches only (3, 2), resolving the ambiguity.")
print("* With noisy distances the three circles no longer pass through a")
print("  single point (residuals are not zero). Least squares picks the")
print(f"  point that fits all three best, here within {position_error_m:.0f} m"
      " of the truth,")
print("  smaller than the 50 m noise level, because the errors of the three")
print("  distances partly cancel out.")
print("* Graph: triangles are towers, coloured circles their measured")
print("  ranges, grey crosses the two-tower solutions, the star the true")
print("  position and the red dot the estimate.")

plt.show()
