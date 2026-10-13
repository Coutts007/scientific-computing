"""
BIT 2118 Scientific Computing - Task J (Task 11)
Root Finding: Newton-Raphson and Secant Methods
=========================================================================

Program file : code/Task_11_newton_secant.py
Run with     : python code/Task_11_newton_secant.py
Output graphs: figures/Task_11_newton_secant.png
               figures/Task_11_diode.png        (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Newton-Raphson follows the tangent line at the current guess down to the
x-axis:
        x_{n+1} = x_n - f(x_n) / f'(x_n)
Near a simple root it converges QUADRATICALLY: the number of correct
digits roughly doubles each step. It needs the derivative f'(x) and a
good starting guess; from a poor guess it can jump to another root or
diverge.

The secant method replaces f'(x_n) by the slope through the last two
points:
        x_{n+1} = x_n - f(x_n) (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
It needs no derivative and converges superlinearly (order about 1.618),
a little slower than Newton per step but with only one new function
evaluation per step.

Both are usually far faster than bisection (Task I), which only gains
one binary digit per step, but neither is guaranteed to converge.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A financial application must compute the interest rate that makes the
net present value (NPV) of an investment zero - the internal rate of
return (IRR).
        NPV(r) = sum_t CF_t / (1 + r)^t
        NPV'(r) = sum_t -t CF_t / (1 + r)^(t+1)

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    cash_flows = [-10000, 3000, 3500, 4000, 2500]
        year 0: investment of 10,000 (negative = money out)
        years 1-4: money received back
    Newton: x0 = 0.10, xtol = 1e-12
    Secant: x0 = 0.05, x1 = 0.20, xtol = 1e-12
Operations:
    discounting (powers of 1 + r), its derivative, Newton and secant
    updates (scipy.optimize.root_scalar, plus hand-written versions that
    record every step).

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Overflow warnings are expected in the divergence demonstration in
# Part 2; they are handled explicitly, so hide the console noise.
warnings.filterwarnings("ignore", category=RuntimeWarning)


def newton(f, fprime, x0, tol=1e-12, max_iter=200):
    """Newton-Raphson; returns (root, list of iterates, converged)."""
    xs = [x0]
    x = x0
    for _ in range(max_iter):
        try:
            x_new = x - f(x) / fprime(x)
        except (OverflowError, ZeroDivisionError):
            # Plain Python floats raise instead of returning inf when a
            # power such as (1 + r)**t becomes too large: treat as divergence.
            xs.append(np.inf)
            return np.inf, xs, False
        xs.append(x_new)
        if not np.isfinite(x_new):
            return x_new, xs, False
        if abs(x_new - x) < tol:
            return x_new, xs, True
        x = x_new
    return x, xs, False


def secant(f, x0, x1, tol=1e-12, max_iter=200):
    """Secant method; returns (root, list of iterates, converged)."""
    xs = [x0, x1]
    for _ in range(max_iter):
        f0, f1 = f(xs[-2]), f(xs[-1])
        if f1 == f0:                        # flat secant: cannot continue
            return xs[-1], xs, abs(f1) < tol
        x_new = xs[-1] - f1 * (xs[-1] - xs[-2]) / (f1 - f0)
        xs.append(x_new)
        if abs(x_new - xs[-2]) < tol:
            return x_new, xs, True
    return xs[-1], xs, False


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Initial investment followed by annual cash inflows
cash_flows = [-10000, 3000, 3500, 4000, 2500]


def npv(rate, flows=cash_flows):
    """Net present value: each cash flow discounted back to year 0."""
    return sum(
        cf / (1 + rate)**t
        for t, cf in enumerate(flows)
    )


def d_npv(rate, flows=cash_flows):
    """Derivative of NPV with respect to the rate (for Newton)."""
    return sum(
        -t * cf / (1 + rate)**(t + 1)
        for t, cf in enumerate(flows)
        if t > 0
    )


newton_sp = root_scalar(
    npv,
    fprime=d_npv,
    x0=0.10,
    method="newton",
    xtol=1e-12
)

secant_sp = root_scalar(
    npv,
    x0=0.05,
    x1=0.20,
    method="secant",
    xtol=1e-12
)

irr = newton_sp.root
_, newton_steps, _ = newton(npv, d_npv, 0.10)
_, secant_steps, _ = secant(npv, 0.05, 0.20)
bisect_sp = root_scalar(npv, bracket=[0.0, 1.0], method="bisect",
                        xtol=1e-12)

print("=" * 65)
print("PART 1 - Original solution: internal rate of return")
print("=" * 65)
print(f"IRR using Newton-Raphson: {newton_sp.root:.6%} "
      f"({newton_sp.iterations} iterations)")
print(f"IRR using Secant method : {secant_sp.root:.6%} "
      f"({secant_sp.iterations} iterations)")
print(f"IRR using bisection     : {bisect_sp.root:.6%} "
      f"({bisect_sp.iterations} iterations, for comparison)")
print(f"NPV at Newton solution  : {npv(newton_sp.root):.6f}")
print()
print("Newton iterates (correct digits roughly double each step):")
for k, r in enumerate(newton_steps):
    print(f"  step {k}: r = {r:.15f}   error = {abs(r - irr):.2e}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the final-year inflow is raised from 2,500 to 5,000.
#   Expected: the project returns more money for the same investment, so
#   NPV is higher at every rate and the IRR (where NPV = 0) increases.
#
# Change 2: Newton's starting guess is changed from 10 % to other values.
#   Expected: from guesses near 10 % (or below) Newton still finds the
#   IRR, using more steps the further away it starts. From a large guess
#   such as 100 % the tangent is almost flat (NPV changes very slowly at
#   high rates), so the first step overshoots far to the left. It then
#   lands on a mathematically valid but meaningless root below -100 %,
#   or blows up. Newton is fast but only reliable from a good start.

flows_changed = [-10000, 3000, 3500, 4000, 5000]
irr_changed = root_scalar(lambda r: npv(r, flows_changed),
                          fprime=lambda r: d_npv(r, flows_changed),
                          x0=0.10, method="newton", xtol=1e-12).root

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print(f"Change 1 - final inflow 2,500 -> 5,000: IRR {irr:.4%} -> "
      f"{irr_changed:.4%}")
print()
print("Change 2 - Newton from different starting guesses:")
print(f"{'x0':>8}{'Result':>22}{'Steps':>7}  Comment")
for x0 in (0.10, -0.5, -0.9, 1.0, 2.0):
    root, steps, ok = newton(npv, d_npv, x0)
    if not ok or not np.isfinite(root):
        comment, shown = "diverged", "overflow"
    elif abs(root - irr) < 1e-8:
        comment, shown = "found the IRR", f"{root:.6%}"
    else:
        comment, shown = "wrong root (rate < -100 %)", f"{root:.4%}"
    print(f"{x0:>8.0%}{shown:>22}{len(steps) - 1:>7}  {comment}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Left graph : NPV against the discount rate for both cash-flow sets.
# Right graph: error per iteration for Newton, secant and bisection.

rates = np.linspace(0.0, 0.40, 300)
fig, (ax_npv, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_npv.plot(rates * 100, [npv(r) for r in rates], label="Original flows")
ax_npv.plot(rates * 100, [npv(r, flows_changed) for r in rates], "--",
            label="Final inflow 5,000")
ax_npv.axhline(0, color="black", linewidth=0.8)
ax_npv.plot(irr * 100, 0, "o", color="C0")
ax_npv.plot(irr_changed * 100, 0, "o", color="C1")
ax_npv.annotate(f"IRR {irr:.2%}", (irr * 100, 0), xytext=(-80, -20),
                textcoords="offset points")
ax_npv.annotate(f"IRR {irr_changed:.2%}", (irr_changed * 100, 0),
                xytext=(5, 15), textcoords="offset points")
ax_npv.set_xlabel("Discount rate (%)")
ax_npv.set_ylabel("Net present value")
ax_npv.set_title("NPV profile")
ax_npv.legend()
ax_npv.grid(True)

# Bisection history for the comparison (hand-made, records midpoints)
a, b, bis = 0.0, 1.0, []
while (b - a) / 2 > 1e-12:
    m = (a + b) / 2
    bis.append(m)
    a, b = (a, m) if npv(a) * npv(m) <= 0 else (m, b)

for name, xs, mk in (("Newton", newton_steps, "o"),
                     ("Secant", secant_steps, "s"),
                     ("Bisection", bis, ".")):
    err = np.maximum(np.abs(np.array(xs) - irr), 1e-17)
    ax_err.semilogy(err, marker=mk, label=name)
ax_err.set_xlabel("Iteration")
ax_err.set_ylabel("|r_k - IRR|")
ax_err.set_title("Convergence speed")
ax_err.legend()
ax_err.grid(True, which="both", alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_11_newton_secant.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The IRR is {irr:.4%}: the project earns the same as a savings")
print(f"  account paying {irr:.2%} a year. It is worth doing if money can")
print("  be borrowed (or would otherwise earn) less than that rate.")
print(f"* NPV at the solution is {npv(irr):.1e}, i.e. zero up to round-off.")
print(f"* Newton needed {newton_sp.iterations} steps, secant "
      f"{secant_sp.iterations}, bisection {bisect_sp.iterations}. Newton's")
newton_errors = ", ".join(f"{abs(r - irr):.0e}" for r in newton_steps[:4])
print(f"  errors went {newton_errors}: each error is roughly the square of")
print("  the previous one (times a constant) - quadratic convergence.")
print(f"* A larger final inflow raised the IRR to {irr_changed:.2%}.")
print("* Starting guesses matter: from 100 % Newton returned a root below")
print("  -100 %, which has no financial meaning; from 200 % it diverged.")
print("  In practice, start Newton from a sensible guess or fall back on")
print("  bisection when it fails.")
print("* Graph (left): NPV falls as the rate rises; the IRR is where the")
print("  curve crosses zero. Raising the final inflow lifts the curve and")
print("  moves the crossing right. Graph (right): Newton and secant curves")
print("  plunge steeply (bending downward), bisection falls in a straight")
print("  line - one binary digit per step.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: operating point of a diode circuit.
#
# Problem:
#   A silicon diode is connected in series with a resistor R = 1 kOhm to a
#   5 V supply. Find the diode voltage V_D and the current I.
#
# Model:
#   Shockley diode equation : I = I_s (exp(V_D / V_T) - 1)
#   Kirchhoff (resistor)     : I = (V_s - V_D) / R
#   Setting them equal gives one nonlinear equation in V_D:
#       f(V) = I_s (exp(V / V_T) - 1) - (V_s - V) / R = 0
#       f'(V) = (I_s / V_T) exp(V / V_T) + 1 / R
#   There is no algebraic formula for V, so a numerical root is needed.
#
# Assumptions:
#   * Saturation current I_s = 1e-12 A, ideality factor 1.
#   * Thermal voltage V_T = 25.85 mV (room temperature, 300 K).
#   * Ideal supply and resistor; no series resistance in the diode.
#
# Input data: V_s = 5 V, R = 1000 Ohm, I_s = 1e-12 A, V_T = 0.025852 V
#
# Methods: Newton (derivative known) and secant (no derivative) from a
# sensible guess (0.6 V, the typical silicon turn-on voltage) and from
# a poor guess (0 V).

V_S, R_LOAD = 5.0, 1000.0
I_S, V_T = 1e-12, 0.025852


def diode_f(V):
    """Diode current minus resistor current; zero at the operating point."""
    return I_S * np.expm1(V / V_T) - (V_S - V) / R_LOAD


def diode_fprime(V):
    """Derivative of diode_f with respect to V."""
    return I_S / V_T * np.exp(V / V_T) + 1 / R_LOAD


vd_good, hist_good, _ = newton(diode_f, diode_fprime, 0.6)
vd_poor, hist_poor, _ = newton(diode_f, diode_fprime, 0.0)
vd_sec, hist_sec, _ = secant(diode_f, 0.6, 0.7)
current = (V_S - vd_good) / R_LOAD

print()
print("=" * 65)
print("PART 4 - Modified case: diode-resistor circuit operating point")
print("=" * 65)
print(f"Newton from 0.6 V : V_D = {vd_good:.6f} V, "
      f"{len(hist_good) - 1} iterations")
print(f"Secant from 0.6/0.7 V: V_D = {vd_sec:.6f} V, "
      f"{len(hist_sec) - 2} iterations")
print(f"Newton from 0.0 V : V_D = {vd_poor:.6f} V, "
      f"{len(hist_poor) - 1} iterations")
print(f"  first step from 0 V jumped to {hist_poor[1]:.3f} V")
print(f"Circuit current I = {current * 1000:.4f} mA")
print(f"Check - diode current {I_S * np.expm1(vd_good / V_T) * 1000:.4f} mA "
      f"= resistor current {current * 1000:.4f} mA")
print(f"Power: diode {vd_good * current * 1000:.3f} mW, resistor "
      f"{current ** 2 * R_LOAD * 1000:.3f} mW")

# Graph: the two I-V curves (intersection = operating point) and the
# Newton iterates from the poor guess.
v_plot = np.linspace(0, 0.8, 400)
fig2, (ax_iv, ax_it) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_iv.plot(v_plot, I_S * np.expm1(v_plot / V_T) * 1000, label="Diode current")
ax_iv.plot(v_plot, (V_S - v_plot) / R_LOAD * 1000,
           label="Resistor (load line)")
ax_iv.plot(vd_good, current * 1000, "ko",
           label=f"Operating point ({vd_good:.3f} V, {current*1000:.2f} mA)")
ax_iv.set_ylim(0, 10)
ax_iv.set_xlabel("Diode voltage V_D (V)")
ax_iv.set_ylabel("Current (mA)")
ax_iv.set_title("Diode circuit: load-line intersection")
ax_iv.legend(fontsize=8)
ax_iv.grid(True)

ax_it.plot(hist_poor, marker=".", label="Newton from 0.0 V")
ax_it.plot(hist_good, marker="o", label="Newton from 0.6 V")
ax_it.axhline(vd_good, color="gray", linestyle="--", label="Solution")
ax_it.set_xlabel("Iteration")
ax_it.set_ylabel("V_D estimate (V)")
ax_it.set_title("Effect of the starting guess")
ax_it.legend()
ax_it.grid(True)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_11_diode.png", dpi=120)

print()
print("Interpretation:")
print(f"* The diode settles at {vd_good:.3f} V and {current * 1000:.2f} mA: "
      "almost all of the 5 V")
print("  supply is dropped across the resistor, which limits the current.")
print("* From the sensible 0.6 V guess Newton converged in a handful of")
print("  steps and secant in a few more, without needing f'(V).")
print(f"* From 0 V the diode curve is almost flat, so the first tangent")
print(f"  shot out to {hist_poor[1]:.1f} V. There exp(V/V_T) is enormous and")
print("  Newton can only walk back about one V_T (26 mV) per step, so it")
print(f"  needed {len(hist_poor) - 1} iterations. Same equation, same method - "
      "the starting")
print("  guess made a ~30x difference. Engineers therefore start diode")
print("  calculations near 0.6-0.7 V.")
print("* Graph (left): the operating point is where the steep diode curve")
print("  meets the resistor's straight load line. Graph (right): the poor")
print("  guess overshoots, then creeps down in a straight line.")

plt.show()
