"""
BIT 2118 Scientific Computing - Task Q (Task 18)
Numerical Solution of ODEs Using SciPy
=========================================================================

Program file : code/Task_18_scipy_ode.py
Run with     : python code/Task_18_scipy_ode.py
Output graphs: figures/Task_18_scipy_ode.png
               figures/Task_18_suspension.png    (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
scipy.integrate.solve_ivp solves an initial value problem
        dy/dt = f(t, y),   y(t0) = y0
over an interval [t0, t_end]. Its default method, RK45 (Runge-Kutta-
Fehlberg/Dormand-Prince), uses six slope evaluations per step to build a
5th-order estimate, plus an embedded 4th-order estimate. The difference
between them estimates the local error, and the step size is adjusted
automatically so that the error stays below
        atol + rtol * |y|.
Large steps are taken where the solution is smooth and small steps where
it changes quickly. This is far more accurate and efficient than the
fixed-step Euler method of Task P.

Higher-order ODEs are rewritten as first-order SYSTEMS. For example
x'' = g(x, x') becomes y = [x, v] with  x' = v,  v' = g(x, v).

RC charging circuit (Kirchhoff's voltage law):
        dV/dt = (Vs - V) / (R C),   V(0) = V0
Exact solution: V(t) = Vs + (V0 - Vs) e^{-t/(RC)}.  tau = RC is the time
constant: after one tau the capacitor reaches 63.2 % of the way to Vs;
after 5 tau it is above 99 %.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
Model the voltage in an RC circuit as it charges over time. Write the
circuit ODE as a Python function, pass it to solve_ivp with the initial
voltage and time interval, and plot the numerical solution.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    Vs = 5 V (supply), R = 1000 ohm, C = 0.001 F  ->  tau = RC = 1 s
    V0 = 0 V, time interval 0-5 s, 200 output times,
    rtol = 1e-9, atol = 1e-11 (tight tolerances)
Operations:
    solve_ivp (RK45), exact exponential for validation, maximum error.

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# RC charging circuit:
# dV/dt = (Vs - V)/(R*C)
Vs = 5.0          # supply voltage (V)
R = 1000.0        # resistance (ohms)
C = 0.001         # capacitance (farads)
V0 = 0.0          # initial capacitor voltage


def rc_circuit(t, V, R=R, C=C):
    """Right-hand side of the RC ODE. V is a length-1 array."""
    return [(Vs - V[0]) / (R * C)]


def exact_rc(t, R=R, C=C):
    """Exact RC charging curve."""
    return Vs * (1 - np.exp(-t / (R * C)))


t_eval = np.linspace(0, 5, 200)

solution = solve_ivp(
    rc_circuit,
    (0, 5),
    [V0],
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

# Exact RC charging curve for validation
exact = exact_rc(solution.t)
tau = R * C

print("=" * 65)
print("PART 1 - Original solution: RC circuit charging, tau = 1 s")
print("=" * 65)
print("Solver success:", solution.success)
print(f"Capacitor voltage at 5 s: {solution.y[0, -1]:.4f} V")
print(f"Maximum error against exact solution: "
      f"{np.max(np.abs(solution.y[0] - exact)):.3e} V")
print(f"Right-hand-side evaluations used: {solution.nfev}")
print(f"Voltage at t = tau = {tau:.0f} s: "
      f"{np.interp(tau, solution.t, solution.y[0]):.4f} V "
      f"({np.interp(tau, solution.t, solution.y[0]) / Vs:.1%} of Vs)")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: capacitance changed from 0.001 F to 0.002 F and 0.0005 F
#   (tau = 2 s and 0.5 s).
#   Expected: larger C stores more charge, so it charges more slowly.
#   At 5 s: tau = 2 -> 5(1 - e^-2.5) = 4.59 V; tau = 0.5 -> 5(1 - e^-10)
#   = 4.9998 V. The curve shape is the same, stretched in time by tau.
#
# Change 2: the solver tolerances are loosened to the SciPy defaults
#   (rtol = 1e-3, atol = 1e-6).
#   Expected: far fewer function evaluations (bigger steps) but a much
#   larger error - accuracy is traded for speed. Error should still be
#   at most around 1e-3 relative.
#
# Change 3: a different method, 'Radau' (implicit, for stiff problems),
#   with the original tight tolerances.
#   Expected: same answer to within tolerance; different amount of work.

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print("Change 1 - capacitance:")
curves = {}
for C_new in (0.0005, 0.001, 0.002):
    sol = solve_ivp(rc_circuit, (0, 5), [V0], t_eval=t_eval,
                    args=(R, C_new), rtol=1e-9, atol=1e-11)
    curves[C_new] = sol
    print(f"  C = {C_new:.4f} F, tau = {R * C_new:.1f} s: V(5 s) = "
          f"{sol.y[0, -1]:.4f} V  (exact {exact_rc(5.0, R, C_new):.4f} V)")

print()
print("Change 2 and 3 - tolerances and method:")
print(f"{'Method':<8}{'rtol':>8}{'atol':>8}{'evaluations':>13}"
      f"{'steps':>7}{'max error (V)':>15}")
for method, rtol, atol in (("RK45", 1e-9, 1e-11), ("RK45", 1e-3, 1e-6),
                           ("Radau", 1e-9, 1e-11)):
    sol = solve_ivp(rc_circuit, (0, 5), [V0], method=method, t_eval=t_eval,
                    rtol=rtol, atol=atol)
    err = np.max(np.abs(sol.y[0] - exact_rc(sol.t)))
    n_steps = len(solve_ivp(rc_circuit, (0, 5), [V0], method=method,
                            rtol=rtol, atol=atol).t) - 1
    print(f"{method:<8}{rtol:>8.0e}{atol:>8.0e}{sol.nfev:>13}"
          f"{n_steps:>7}{err:>15.2e}")

# Steps chosen by the adaptive solver (no t_eval -> its own step points)
adaptive = solve_ivp(rc_circuit, (0, 5), [V0], rtol=1e-6, atol=1e-9)
step_sizes = np.diff(adaptive.t)


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Left graph : charging curves for the three capacitances, with the
#              63.2 % level at t = tau marked for the original circuit.
# Right graph: step sizes chosen by the adaptive solver.

fig, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_v.plot(solution.t, exact, "k--", linewidth=2.5, alpha=0.5,
          label="Exact (C = 0.001)")
for C_new, sol in curves.items():
    ax_v.plot(sol.t, sol.y[0], label=f"solve_ivp, tau = {R * C_new:g} s")
ax_v.axhline(0.632 * Vs, color="gray", linestyle=":", linewidth=0.8)
ax_v.axvline(tau, color="gray", linestyle=":", linewidth=0.8)
ax_v.annotate("63.2 % at t = tau", (tau, 0.632 * Vs), xytext=(15, -25),
              textcoords="offset points", fontsize=8)
ax_v.set_xlabel("Time (s)")
ax_v.set_ylabel("Capacitor voltage (V)")
ax_v.set_title("RC circuit charging")
ax_v.legend(fontsize=8)
ax_v.grid(True)

ax_h.plot(adaptive.t[1:], step_sizes, "o-")
ax_h.set_xlabel("Time (s)")
ax_h.set_ylabel("Step size chosen (s)")
ax_h.set_title("Adaptive step sizes (RK45, rtol = 1e-6)")
ax_h.grid(True)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_18_scipy_ode.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The capacitor charges from 0 V towards the 5 V supply, reaching "
      f"{solution.y[0, -1]:.3f} V")
print("  after 5 s (5 time constants) - effectively fully charged.")
print("* It charges fastest at first, when the voltage difference across")
print("  the resistor (and so the current) is largest, then more slowly.")
print("* The solver matches the exact formula to ~1e-10 V with tight")
print("  tolerances. Default tolerances use far fewer evaluations but are")
print("  less accurate - choose tolerances to suit the application.")
print("* Doubling C doubles tau: the curve is stretched in time, so at 5 s")
print("  the larger capacitor is still only at about 92 % of Vs.")
print("* Graph (left): the dotted lines show the capacitor at 63.2 % of 5 V")
print("  after one time constant. Graph (right): the solver starts with")
print(f"  tiny steps ({step_sizes[0]:.1e} s) and lengthens them to "
      f"{step_sizes.max():.2f} s as the curve flattens -")
print("  this automatic step control is solve_ivp's main advantage over")
print("  fixed-step Euler.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: car suspension driving up a kerb (mechanical system).
#
# Problem:
#   A car wheel suddenly rises by 5 cm when it drives up a kerb. How does
#   the car body move, and how does the shock absorber's condition (new
#   vs worn) affect comfort and safety?
#
# Model (quarter-car, single mass-spring-damper):
#   Body mass m on a spring k and damper c attached to the wheel. With
#   body position x and wheel (road) position u:
#       m x'' = -k (x - u) - c (x' - u')
#   The wheel climbs the kerb smoothly over T_RISE = 0.05 s:
#       u(t)  = 0.05 (1 - cos(pi t / T_RISE)) / 2   for t < T_RISE,
#       u(t)  = 0.05                                 afterwards,
#   so the wheel velocity u'(t) is finite and the damper force c u'
#   (which pushes the body up while the wheel is rising) is included.
#   Second-order ODE -> first-order system with y = [x, v]:
#       x' = v
#       v' = (-k (x - u) - c (v - u')) / m
#
# Assumptions:
#   * m = 300 kg (a quarter of a 1200 kg car), k = 20 000 N/m.
#   * New shock absorber c = 1800 N s/m, worn one c = 400 N s/m.
#   * Damping ratio zeta = c / (2 sqrt(k m)) -> about 0.37 (new) and
#     0.08 (worn). Natural frequency sqrt(k/m)/(2 pi) ~ 1.3 Hz.
#   * The 5 cm rise takes 0.05 s (the wheel rolling over a rounded kerb
#     edge); tyre flexibility is ignored.
#   * max_step is limited to 5 ms so the solver cannot step over the short
#     0.05 s rise without "seeing" it.
#
# Outputs: peak overshoot above the new 5 cm level, settling time (when
# the motion stays within +/- 2 mm of 5 cm), and the peak acceleration
# felt by passengers. An event function stops nothing but records when
# the body first crosses the 5 cm level.

M_BODY, K_SPRING = 300.0, 20_000.0
STEP_HEIGHT = 0.05
SETTLE_BAND = 0.002
T_RISE = 0.05


def road(t):
    """Wheel height u(t) and wheel velocity u'(t) while climbing the kerb."""
    if t < T_RISE:
        u = STEP_HEIGHT * (1 - np.cos(np.pi * t / T_RISE)) / 2
        du = STEP_HEIGHT * np.pi / (2 * T_RISE) * np.sin(np.pi * t / T_RISE)
        return u, du
    return STEP_HEIGHT, 0.0


def suspension(t, y, c):
    """First-order system for the body position x and velocity v."""
    x, v = y
    u, du = road(t)
    a = (-K_SPRING * (x - u) - c * (v - du)) / M_BODY
    return [v, a]


def crosses_level(t, y, c):
    """Event: body reaches the new road level (x = 5 cm)."""
    return y[0] - STEP_HEIGHT


crosses_level.direction = 1      # only count upward crossings

t_span = (0, 5)
t_out = np.linspace(*t_span, 2000)
cases = {"New shock absorber": 1800.0, "Worn shock absorber": 400.0}
outcomes = {}

print()
print("=" * 65)
print("PART 4 - Modified case: car suspension driving up a 5 cm kerb")
print("=" * 65)
print(f"Natural frequency: {np.sqrt(K_SPRING / M_BODY) / (2 * np.pi):.2f} Hz")
print(f"{'Case':<22}{'zeta':>6}{'overshoot':>11}{'settle (s)':>12}"
      f"{'first reach (s)':>17}{'peak accel':>12}")
for name, c in cases.items():
    sol = solve_ivp(suspension, t_span, [0.0, 0.0], args=(c,),
                    t_eval=t_out, events=crosses_level, rtol=1e-8,
                    atol=1e-10, max_step=0.005)
    x = sol.y[0]
    accel = np.array([suspension(t, y, c)[1] for t, y in zip(sol.t, sol.y.T)])
    overshoot = x.max() - STEP_HEIGHT
    outside = np.where(np.abs(x - STEP_HEIGHT) > SETTLE_BAND)[0]
    settle_time = sol.t[outside[-1] + 1] if outside[-1] + 1 < len(sol.t) \
        else np.nan
    zeta = c / (2 * np.sqrt(K_SPRING * M_BODY))
    outcomes[name] = (sol, accel, overshoot, settle_time)
    print(f"{name:<22}{zeta:>6.2f}{overshoot * 1000:>9.1f}mm"
          f"{settle_time:>12.2f}{sol.t_events[0][0]:>17.3f}"
          f"{np.abs(accel).max():>9.2f}m/s2")

fig2, (ax_x, ax_a) = plt.subplots(1, 2, figsize=(12, 4.5))
for name, (sol, accel, _, settle) in outcomes.items():
    ax_x.plot(sol.t, sol.y[0] * 100, label=name)
    ax_a.plot(sol.t, accel / 9.81, label=name)
ax_x.axhline(STEP_HEIGHT * 100, color="black", linestyle=":",
             label="Kerb height 5 cm")
ax_x.axhspan((STEP_HEIGHT - SETTLE_BAND) * 100,
             (STEP_HEIGHT + SETTLE_BAND) * 100, color="gray", alpha=0.2,
             label="+/- 2 mm settling band")
ax_x.set_xlabel("Time (s)")
ax_x.set_ylabel("Car body height (cm)")
ax_x.set_title("Body motion after hitting a kerb")
ax_x.legend(fontsize=8)
ax_x.grid(True)
ax_a.set_xlabel("Time (s)")
ax_a.set_ylabel("Body acceleration (g)")
ax_a.set_title("Acceleration felt by passengers")
ax_a.legend(fontsize=8)
ax_a.grid(True)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_18_suspension.png", dpi=120)

new = outcomes["New shock absorber"]
worn = outcomes["Worn shock absorber"]
print()
print("Interpretation:")
print(f"* With a new shock absorber the body overshoots by {new[2] * 1000:.0f}"
      f" mm and settles within")
print(f"  {new[3]:.1f} s - one or two gentle bounces.")
print(f"* With a worn one it overshoots by {worn[2] * 1000:.0f} mm "
      f"({worn[2] / STEP_HEIGHT:.0%} of the kerb height)")
if np.isnan(worn[3]):
    print("  and is still bouncing after 5 s.")
else:
    print(f"  and keeps bouncing for {worn[3]:.1f} s. Long bouncing reduces "
          "tyre grip and")
    print("  comfort, which is why worn shock absorbers fail vehicle "
          "inspections.")
print(f"* The initial jolt is LARGER with the new damper "
      f"({np.abs(new[1]).max() / 9.81:.2f} g vs "
      f"{np.abs(worn[1]).max() / 9.81:.2f} g), because a")
print("  stiffer damper passes the fast upward wheel movement straight to the")
print("  body. Firm damping gives control (little bouncing) at the cost of a")
print("  sharper first jolt - a real design trade-off.")
print("* solve_ivp handled the second-order equation after rewriting it as")
print("  two first-order equations; the event function located the moment")
print("  the body first reaches the new level without a fine time grid.")
print("* Graph (left): the worn damper's curve oscillates long after the new")
print("  one has entered the grey settling band. Graph (right): acceleration")
print("  in units of g - the new damper gives one sharp jolt, then calm.")

plt.show()
