"""
BIT 2118 Scientific Computing - Task I (Task 10)
Root Finding: Bisection and Fixed-Point Methods
=========================================================================

Program file : code/Task_10_bisection_fixed_point.py
Run with     : python code/Task_10_bisection_fixed_point.py
Output graphs: figures/Task_10_bisection_fixed_point.png
               figures/Task_10_break_even.png   (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
A root of f(x) is a value r with f(r) = 0.

Bisection:
    Start with [a, b] where f(a) and f(b) have opposite signs; by the
    Intermediate Value Theorem a continuous f has a root inside. Take
    the midpoint m = (a + b)/2 and keep the half where the sign changes.
    The interval halves every step, so after k steps the error is at
    most (b - a)/2^k. Reaching a tolerance tol needs about
        k = ceil( log2((b - a) / tol) )   iterations.
    Slow but guaranteed (linear convergence, factor 1/2).

Fixed-point iteration:
    Rewrite f(x) = 0 as x = g(x) and repeat x_{n+1} = g(x_n).
    It converges near the fixed point r if |g'(r)| < 1, and the error
    shrinks by roughly a factor |g'(r)| per step. If |g'(r)| > 1 it
    diverges. The same equation can be rewritten in many ways; only some
    of them converge.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
Determine the temperature at which a nonlinear heat-balance equation
becomes zero (heat gain equals heat loss). Bracket a sign change and
apply bisection; also solve with fixed-point iteration.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    f(T) = T - 25 - 10 exp(-T/20)
        25 : ambient temperature (C)
        10 : heat-gain coefficient (C)
        20 : temperature scale of the gain term (C)
    bracket [20, 40], xtol = 1e-10 for bisection
    g(T) = 25 + 10 exp(-T/20), starting guess T0 = 30, tolerance 1e-10
Operations:
    scipy.optimize.root_scalar(method="bisect"), a hand-written
    bisection (to record each step), fixed-point loop, exponentials.

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def bisection(f, a, b, tol=1e-10, max_iter=200):
    """Bisection written out by hand; returns (root, list of midpoints)."""
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    midpoints = []
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        midpoints.append(m)
        if f(a) * f(m) <= 0:     # sign change in the left half
            b = m
        else:                    # sign change in the right half
            a = m
        if (b - a) / 2 < tol:
            break
    return 0.5 * (a + b), midpoints


def fixed_point(g, x0, tol=1e-10, max_iter=100):
    """Iterate x_{n+1} = g(x_n); returns (last value, history, converged)."""
    x = x0
    history = [x]
    for _ in range(max_iter):
        x_new = g(x)
        history.append(x_new)
        if not np.isfinite(x_new):          # blew up
            return x_new, history, False
        if abs(x_new - x) < tol:
            return x_new, history, True
        x = x_new
    return x, history, False


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Heat-balance model:
# f(T) = 0 means heat gain equals heat loss.
def heat_balance(T, ambient=25.0, gain=10.0):
    return T - ambient - gain * np.exp(-T / 20)


# Bisection: bracket a sign change (f(20) < 0 < f(40))
solution = root_scalar(
    heat_balance,
    bracket=[20, 40],
    method="bisect",
    xtol=1e-10
)
root_manual, midpoints = bisection(heat_balance, 20, 40)


# Fixed-point form:
# T = 25 + 10*exp(-T/20)
def g(T):
    return 25 + 10 * np.exp(-T / 20)


T_fp, history, _ = fixed_point(g, 30.0)
root = solution.root

# g'(T) = -0.5 exp(-T/20): its size at the root predicts the speed
g_prime = -0.5 * np.exp(-root / 20)

print("=" * 65)
print("PART 1 - Original solution: equilibrium temperature")
print("=" * 65)
print(f"f(20) = {heat_balance(20):.3f}, f(40) = {heat_balance(40):.3f} "
      "-> sign change, root inside")
print(f"Bisection equilibrium temperature: {root:.6f} C "
      f"({solution.iterations} iterations, SciPy)")
print(f"Hand-written bisection           : {root_manual:.6f} C "
      f"({len(midpoints)} iterations)")
print(f"Predicted bisection iterations   : "
      f"{int(np.ceil(np.log2(20 / 1e-10)))}")
print(f"Fixed-point equilibrium temperature: {history[-1]:.6f} C")
print("Fixed-point iterations:", len(history) - 1)
print(f"|g'(root)| = {abs(g_prime):.4f} (< 1, so fixed point converges)")
print(f"f(root) = {heat_balance(root):.2e}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the heat-gain coefficient is raised from 10 to 40 (a stronger
#   heat source).
#   Expected: f(T) is lower for every T, so f crosses zero at a higher T
#   - the equilibrium temperature rises. The bracket [20, 40] still
#   contains a sign change.
#
# Change 2: the same equation is rearranged differently for fixed point:
#   T - 25 = 10 exp(-T/20)  ->  T = h(T) = -20 ln((T - 25) / 10).
#   Expected: divergence. h'(T) = -20 / (T - 25); at the root
#   T - 25 is only about 2.4, so |h'(root)| is about 8 > 1 and every step
#   moves FURTHER away from the root.
#
# Change 3: the bisection tolerance is loosened from 1e-10 to 1e-3.
#   Expected: about log2(20/1e-3) = 15 iterations instead of about
#   log2(20/1e-10) = 38. (The hand-written version stops one step
#   earlier, at 14 and 37, because it returns the midpoint of its final
#   interval, which is already within half the interval width.)

sol_hot = root_scalar(lambda T: heat_balance(T, gain=40.0),
                      bracket=[20, 40], method="bisect", xtol=1e-10)


def h(T):
    """Alternative (divergent) fixed-point form of the same equation."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return -20 * np.log((T - 25) / 10)


_, history_bad, converged_bad = fixed_point(h, 27.4, max_iter=10)
h_prime = -20 / (root - 25)

_, midpoints_loose = bisection(heat_balance, 20, 40, tol=1e-3)

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print(f"Change 1 - gain 10 -> 40: equilibrium {root:.4f} C -> "
      f"{sol_hot.root:.4f} C")
print(f"Change 2 - rearranged g: |h'(root)| = {abs(h_prime):.2f} (> 1)")
print("  Iterates from T0 = 27.4:",
      ", ".join(f"{v:.3f}" for v in history_bad))
print(f"  Converged: {converged_bad} (values move away, then become "
      "undefined)")
print(f"Change 3 - tolerance 1e-10 -> 1e-3: {len(midpoints)} -> "
      f"{len(midpoints_loose)} bisection iterations")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Left graph : f(T) for both gains; the root is where each curve crosses 0.
# Right graph: error |x_k - root| per iteration for bisection and fixed
#              point, on a log scale.

temps = np.linspace(20, 40, 300)
fig, (ax_f, ax_e) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_f.plot(temps, heat_balance(temps), label="gain = 10 (original)")
ax_f.plot(temps, heat_balance(temps, gain=40.0), "--", label="gain = 40")
ax_f.axhline(0, color="black", linewidth=0.8)
ax_f.axvline(root, color="C0", linestyle=":")
ax_f.axvline(sol_hot.root, color="C1", linestyle=":")
ax_f.set_xlabel("Temperature (C)")
ax_f.set_ylabel("Heat-balance function f(T)")
ax_f.set_title("Equilibrium temperature")
ax_f.legend()
ax_f.grid(True)

bis_err = np.abs(np.array(midpoints) - root)
fp_err = np.abs(np.array(history) - root)
ax_e.semilogy(range(1, len(bis_err) + 1), np.maximum(bis_err, 1e-16),
              marker=".", label="Bisection")
ax_e.semilogy(range(len(fp_err)), np.maximum(fp_err, 1e-16),
              marker="s", label="Fixed point g(T)")
ax_e.semilogy(range(1, len(bis_err) + 1),
              20 / 2 ** np.arange(1, len(bis_err) + 1), "k:",
              label="Bisection bound 20/2^k")
ax_e.set_xlabel("Iteration")
ax_e.set_ylabel("|estimate - root| (C)")
ax_e.set_title("Convergence of the two methods")
ax_e.legend()
ax_e.grid(True, which="both", alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_10_bisection_fixed_point.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The object settles at {root:.4f} C, where heat gained equals heat")
print("  lost. It is a little above the 25 C ambient because of the")
print("  extra heat input, which weakens as temperature rises.")
print("* Bisection and fixed point agree to about 10 decimal places; fixed")
print(f"  point needed only {len(history) - 1} iterations because |g'| = "
      f"{abs(g_prime):.2f} is small; bisection")
print(f"  needed {len(midpoints)} because it only halves the interval each "
      "time.")
print(f"* A stronger heat source (gain 40) raises equilibrium to "
      f"{sol_hot.root:.2f} C.")
print("* Rearranging the equation badly makes fixed point diverge, while")
print("  bisection always works once a sign change is bracketed.")
print("* Graph (left): each curve crosses zero at its equilibrium; the")
print("  larger gain shifts the curve down and the crossing to the right.")
print("  Graph (right): bisection's error follows the 20/2^k bound in a")
print("  zig-zag; fixed point's error drops in a much steeper straight line.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: break-even points of a small bakery.
#
# Problem:
#   A bakery sells q loaves per day. To sell more it must lower its price,
#   so the price per loaf is p(q) = 80 - 0.02 q (KES). Daily costs are a
#   fixed KES 15,000 (rent, wages, power) plus KES 30 per loaf for
#   ingredients. Find the sales levels at which profit is zero
#   (break-even points).
#
# Model:
#   Revenue R(q) = q (80 - 0.02 q)
#   Cost    C(q) = 15000 + 30 q
#   Profit  P(q) = R(q) - C(q) = -0.02 q^2 + 50 q - 15000
#   P(q) = 0 has two roots: a lower break-even (start making money) and
#   an upper one (price too low, losses again).
#
# Assumptions:
#   * Linear demand: every extra 50 loaves lowers the price by KES 1.
#   * Costs per loaf are constant; everything baked is sold.
#   * Because P is quadratic the exact roots are known from the formula
#         q = (-b +/- sqrt(b^2 - 4ac)) / (2a),
#     so the numerical answers can be checked.
#
# Methods:
#   * Bisection on [0, 1250] and [1250, 2500] (1250 = profit-maximising q,
#     where P'(q) = 0, so each interval holds exactly one root).
#   * Fixed point q = g(q) = (15000 + 0.02 q^2) / 50, derived from P = 0.
#     g'(q) = 0.0008 q, so it converges to the lower root (g' < 1) but
#     not to the upper root (g' > 1).

FIXED_COST, UNIT_COST = 15_000.0, 30.0
P0, SLOPE = 80.0, 0.02


def profit(q):
    """Daily profit (KES) when q loaves are sold."""
    revenue = q * (P0 - SLOPE * q)
    cost = FIXED_COST + UNIT_COST * q
    return revenue - cost


def g_bakery(q):
    """Fixed-point form of P(q) = 0."""
    return (FIXED_COST + SLOPE * q ** 2) / (P0 - UNIT_COST)


q_peak = (P0 - UNIT_COST) / (2 * SLOPE)          # P'(q) = 0
lower = root_scalar(profit, bracket=[0, q_peak], method="bisect",
                    xtol=1e-8)
upper = root_scalar(profit, bracket=[q_peak, 2 * q_peak], method="bisect",
                    xtol=1e-8)

# Exact roots from the quadratic formula for checking
a_c, b_c, c_c = -SLOPE, P0 - UNIT_COST, -FIXED_COST
disc = np.sqrt(b_c ** 2 - 4 * a_c * c_c)
exact = sorted([(-b_c + disc) / (2 * a_c), (-b_c - disc) / (2 * a_c)])

q_fp, hist_fp, conv_fp = fixed_point(g_bakery, 0.0, tol=1e-8, max_iter=200)
q_fp_up, hist_fp_up, conv_fp_up = fixed_point(g_bakery, exact[1] + 1.0,
                                              tol=1e-8, max_iter=15)

print()
print("=" * 65)
print("PART 4 - Modified case: break-even points of a bakery")
print("=" * 65)
print(f"Profit-maximising output: {q_peak:.0f} loaves, max profit "
      f"KES {profit(q_peak):,.0f}/day")
print(f"Lower break-even (bisection): {lower.root:.4f} loaves "
      f"(exact {exact[0]:.4f})")
print(f"Upper break-even (bisection): {upper.root:.4f} loaves "
      f"(exact {exact[1]:.4f})")
print(f"Fixed point from q0 = 0      : {q_fp:.4f} loaves, "
      f"{len(hist_fp) - 1} iterations, g' = {0.0008 * exact[0]:.3f}")
print(f"Fixed point from q0 = upper+1: converged = {conv_fp_up}, "
      f"g' = {0.0008 * exact[1]:.3f}; last value {hist_fp_up[-1]:.3g}")
print(f"Price at break-even: KES {P0 - SLOPE * exact[0]:.2f} and "
      f"KES {P0 - SLOPE * exact[1]:.2f}")

q_plot = np.linspace(0, 2600, 400)
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(q_plot, profit(q_plot) / 1000, label="Profit P(q)")
ax2.axhline(0, color="black", linewidth=0.8)
for r, name in ((lower.root, "lower"), (upper.root, "upper")):
    ax2.plot(r, 0, "ro")
    ax2.annotate(f"{name} break-even\n{r:.0f} loaves", (r, 0),
                 textcoords="offset points", xytext=(0, -35), ha="center",
                 bbox=dict(boxstyle="round", fc="white", ec="gray"),
                 fontsize=8)
ax2.plot(q_peak, profit(q_peak) / 1000, "g^", label="Maximum profit")
ax2.fill_between(q_plot, profit(q_plot) / 1000, 0,
                 where=profit(q_plot) > 0, alpha=0.15, color="green",
                 label="Profitable range")
ax2.set_xlabel("Loaves sold per day (q)")
ax2.set_ylabel("Profit (thousand KES/day)")
ax2.set_title("Bakery break-even analysis")
ax2.legend(loc="lower center", fontsize=8)
ax2.grid(True)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_10_break_even.png", dpi=120)

print()
print("Interpretation:")
print(f"* The bakery must sell at least {np.ceil(exact[0]):.0f} loaves a day "
      "to cover its costs.")
print(f"* Profit peaks at {q_peak:.0f} loaves (KES {profit(q_peak):,.0f}/day)."
      " Beyond that, the price")
print("  cut needed to sell more loaves outweighs the extra sales.")
print(f"* Above {np.floor(exact[1]):.0f} loaves the price (about KES "
      f"{P0 - SLOPE * exact[1]:.0f}) is so low that")
print("  the bakery makes a loss again.")
print("* Bisection found both roots because each bracket was chosen to")
print("  contain one sign change. Fixed point found only the lower root:")
print("  near the upper root |g'| > 1, so iterates run away - the same")
print("  behaviour seen in Part 2.")
print("* Graph: the green area is the profitable range between the two")
print("  red break-even points; the triangle marks maximum profit.")

plt.show()
