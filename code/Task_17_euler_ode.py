"""
BIT 2118 Scientific Computing - Task P (Task 17)
Ordinary Differential Equations - Euler Method
=========================================================================

Program file : code/Task_17_euler_ode.py
Run with     : python code/Task_17_euler_ode.py
Output graphs: figures/Task_17_euler_ode.png
               figures/Task_17_sir_epidemic.png  (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
An ordinary differential equation (ODE) describes how a quantity
changes:  dy/dt = f(t, y),  with a known starting value y(t0) = y0.

Euler's method follows the tangent (the current slope) for one small
step h and repeats:
        t_{n+1} = t_n + h
        y_{n+1} = y_n + h f(t_n, y_n)
It is a FIRST-ORDER method: the global error at a fixed end time is
roughly proportional to h (halve h -> halve the error), so accurate
answers need many small steps.

Stability: for a decay equation dy/dt = -k (y - y_inf) each Euler step
multiplies the distance from equilibrium by (1 - h k).
    0 < h k < 1   -> smooth decay (always below/above the exact curve)
    1 < h k < 2   -> overshoots and oscillates, but still decays
    h k > 2       -> |1 - h k| > 1: the error GROWS - numerically
                     unstable, even though the real system is stable.

Newton's law of cooling:  dT/dt = -k (T - T_ambient)
Exact solution:           T(t) = T_amb + (T0 - T_amb) e^{-k t}

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
Predict the cooling of a hot object using Newton's law of cooling by
choosing an initial temperature and time step, repeatedly applying
Euler's update, and plotting temperature against time.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    T_ambient = 25 C, T_initial = 90 C, k = 0.12 per minute,
    h = 1 minute, end_time = 30 minutes
Operations:
    Euler update T_{n+1} = T_n + h (-k (T_n - T_amb)) repeated 30 times,
    exact exponential solution for validation, absolute error.

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Newton's law of cooling:
# dT/dt = -k(T - T_ambient)
T_ambient = 25.0
T_initial = 90.0
k = 0.12
h = 1.0  # minutes
end_time = 30.0


def cooling_rate(T, k=k):
    """Right-hand side f(T) of the cooling ODE (C per minute)."""
    return -k * (T - T_ambient)


def euler_cooling(h, k=k, end=end_time):
    """Run Euler's method; return the time grid and temperatures."""
    n_steps = int(round(end / h))
    t = np.linspace(0, n_steps * h, n_steps + 1)
    T = np.zeros_like(t)
    T[0] = T_initial
    for i in range(n_steps):
        T[i + 1] = T[i] + h * cooling_rate(T[i], k)
    return t, T


def exact_cooling(t, k=k):
    """Exact solution of Newton's law of cooling."""
    return T_ambient + (T_initial - T_ambient) * np.exp(-k * t)


time = np.arange(0, end_time + h, h)
temperature = np.zeros_like(time)
temperature[0] = T_initial

for i in range(len(time) - 1):
    temperature[i + 1] = (
        temperature[i] + h * cooling_rate(temperature[i])
    )

# Exact solution for validation
exact = exact_cooling(time)

print("=" * 65)
print("PART 1 - Original solution: cooling with h = 1 minute")
print("=" * 65)
print("First two Euler steps by hand:")
print(f"  T1 = 90 + 1 x (-0.12 x (90 - 25)) = {temperature[1]:.2f} C")
print(f"  T2 = {temperature[1]:.2f} + 1 x (-0.12 x ({temperature[1]:.2f}"
      f" - 25)) = {temperature[2]:.4f} C")
print(f"Euler temperature after 30 min: {temperature[-1]:.2f} C")
print(f"Exact temperature after 30 min: {exact[-1]:.2f} C")
print(f"Final absolute error: {abs(temperature[-1] - exact[-1]):.4f} C")
print(f"Largest error over the run: {np.max(np.abs(temperature - exact)):.4f}"
      f" C at t = {time[np.argmax(np.abs(temperature - exact))]:.0f} min")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the step size h is varied: 0.25, 0.5, 1, 2, 5 minutes.
#   Expected: Euler is first order, so the largest error over the run
#   should roughly double each time h doubles (error ~ C h). The table
#   prints the ratio of each error to the one for the previous (half as
#   large) step, so values near 2 confirm first-order behaviour.
#
# Change 2: very large steps, h = 10 and h = 20 minutes (h k = 1.2, 2.4).
#   Expected: with h k = 1.2 the factor (1 - h k) = -0.2 is negative, so
#   the temperature jumps BELOW 25 C and oscillates around it - physically
#   impossible for a cooling object. With h k = 2.4 the factor is -1.4,
#   |factor| > 1, so the oscillation grows each step: the method is
#   unstable. The stability limit is h < 2/k = 16.7 min.
#
# Change 3: the cooling constant k is raised from 0.12 to 0.25 per minute
#   (e.g. a fan blowing on the object).
#   Expected: faster cooling; and with the same h = 1 the error is
#   larger because the curve bends more sharply (error grows with k^2 h).

step_sizes = [0.25, 0.5, 1.0, 2.0, 5.0]
max_errors = []
for hs in step_sizes:
    t_h, T_h = euler_cooling(hs)
    max_errors.append(np.max(np.abs(T_h - exact_cooling(t_h))))

t_10, T_10 = euler_cooling(10.0, end=60.0)
t_20, T_20 = euler_cooling(20.0, end=120.0)

t_fan, T_fan = euler_cooling(1.0, k=0.25)
fan_error = np.max(np.abs(T_fan - exact_cooling(t_fan, k=0.25)))

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print("Change 1 - step size:")
print(f"{'h (min)':>9}{'steps':>7}{'max error (C)':>15}{'ratio':>8}")
for j, (hs, err) in enumerate(zip(step_sizes, max_errors)):
    ratio = f"{err / max_errors[j - 1]:.2f}" if j else "-"
    print(f"{hs:>9}{int(end_time / hs):>7}{err:>15.4f}{ratio:>8}")
print()
print("Change 2 - very large steps (stability limit h < 2/k = "
      f"{2 / k:.1f} min):")
print("  h = 10:", np.round(T_10, 2))
print("  h = 20:", np.round(T_20, 1))
print()
print("Change 3 - k = 0.25 (fan), h = 1:")
print(f"  temperature after 30 min {T_fan[-1]:.2f} C "
      f"(k = 0.12 gave {temperature[-1]:.2f} C)")
print(f"  largest Euler error {fan_error:.3f} C "
      f"(k = 0.12 gave {max_errors[2]:.3f} C)")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Left graph : Euler (several h) and exact cooling curves.
# Middle     : unstable / oscillating Euler with very large h.
# Right      : largest error against h on log-log axes (slope 1).

t_fine = np.linspace(0, 120, 500)
fig, (ax_c, ax_u, ax_e) = plt.subplots(1, 3, figsize=(15, 4.5))
ax_c.plot(t_fine[t_fine <= 30], exact_cooling(t_fine[t_fine <= 30]), "k",
          linewidth=2, label="Exact solution")
for hs in (1.0, 5.0):
    t_h, T_h = euler_cooling(hs)
    ax_c.plot(t_h, T_h, marker="o", markersize=4, label=f"Euler h = {hs:g}")
ax_c.axhline(T_ambient, color="gray", linestyle=":", label="Ambient 25 C")
ax_c.set_xlabel("Time (minutes)")
ax_c.set_ylabel("Temperature (C)")
ax_c.set_title("Cooling of a hot object")
ax_c.legend(fontsize=8)
ax_c.grid(True)

ax_u.plot(t_fine, exact_cooling(t_fine), "k", linewidth=2, label="Exact")
ax_u.plot(t_10, T_10, "o-", label="Euler h = 10 (hk = 1.2)")
ax_u.plot(t_20, T_20, "s-", label="Euler h = 20 (hk = 2.4)")
ax_u.axhline(T_ambient, color="gray", linestyle=":")
ax_u.set_ylim(-150, 200)
ax_u.set_xlabel("Time (minutes)")
ax_u.set_ylabel("Temperature (C)")
ax_u.set_title("Too-large steps: oscillation and instability")
ax_u.legend(fontsize=8)
ax_u.grid(True)

ax_e.loglog(step_sizes, max_errors, "o-", label="Euler max error")
ax_e.loglog(step_sizes, max_errors[0] * np.array(step_sizes) / step_sizes[0],
            "k:", label="slope 1 (error ~ h)")
ax_e.set_xticks(step_sizes)
ax_e.set_xticklabels([f"{v:g}" for v in step_sizes])
ax_e.minorticks_off()
ax_e.set_xlabel("Step size h (minutes)")
ax_e.set_ylabel("Largest error (C)")
ax_e.set_title("First-order convergence")
ax_e.legend(fontsize=8)
ax_e.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_17_euler_ode.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The object cools from 90 C to about {exact[-1]:.1f} C in 30 minutes,"
      " quickly at first")
print("  and then more slowly as it approaches room temperature (25 C):")
print("  the cooling rate is proportional to the temperature difference.")
print(f"* Euler with h = 1 min is within {max_errors[2]:.2f} C of the exact "
      "answer. It runs a little")
print("  BELOW the exact curve because it uses the slope at the start of each")
print("  step, which is steeper than the average slope over the step.")
print("* Doubling h roughly doubles the error (ratio column ~2): first-order")
print("  convergence. Ten times more accuracy needs ten times more steps.")
print("  At h = 5 the ratio is 3: such a large step (h k = 0.6) is outside")
print("  the range where the simple 'error ~ h' rule holds.")
print("* Steps larger than 2/k = 16.7 min make Euler unstable, and steps")
print("  above 1/k = 8.3 min already give non-physical oscillation below")
print("  room temperature. Step size must suit the system's time scale.")
print("* With a fan (k = 0.25) the object cools faster and Euler's error")
print("  with the same h grows, because the curve bends more sharply.")
print("* Graphs: left - Euler with h = 5 cools visibly too fast, h = 1 nearly")
print("  matches; middle - large steps zig-zag around 25 C (h = 10) or")
print("  explode (h = 20); right - errors lie on a slope-1 line.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: spread of flu in a boarding school (SIR epidemic model).
#
# Problem:
#   One student returns to a 1000-student boarding school with flu.
#   The school wants to know how many students may fall ill at the same
#   time (to plan sick-bay beds) and when the peak will occur.
#
# Model (a SYSTEM of three ODEs, Euler applied to each equation):
#   S = susceptible, I = infected, R = recovered, S + I + R = N
#       dS/dt = -beta S I / N
#       dI/dt =  beta S I / N - gamma I
#       dR/dt =  gamma I
#
# Assumptions:
#   * Closed population N = 1000 (no one enters or leaves).
#   * beta = 0.5 per day (contact/transmission rate),
#     gamma = 1/4 per day (average illness lasts 4 days),
#     so the basic reproduction number R0 = beta/gamma = 2.
#   * Recovered students are immune for the rest of the term.
#   * Initial state: S = 999, I = 1, R = 0. Simulate 60 days.
#
# Methods: Euler with h = 1 day and h = 0.25 day; a very fine Euler run
# (h = 0.001 day) is used as the reference "exact" answer because the SIR
# model has no simple formula for I(t).
#
# Outputs: peak number ill, day of peak, total ever infected, and the
# effect of step size on these numbers.

N = 1000.0
BETA, GAMMA = 0.5, 0.25


def sir_rates(state):
    """Return (dS/dt, dI/dt, dR/dt) for the current (S, I, R)."""
    S, I, R = state
    new_infections = BETA * S * I / N
    recoveries = GAMMA * I
    return np.array([-new_infections, new_infections - recoveries,
                     recoveries])


def euler_sir(h, days=60.0):
    """Euler's method applied to the whole state vector at once."""
    n_steps = int(round(days / h))
    t = np.linspace(0, days, n_steps + 1)
    y = np.zeros((n_steps + 1, 3))
    y[0] = [N - 1, 1.0, 0.0]
    for n in range(n_steps):
        y[n + 1] = y[n] + h * sir_rates(y[n])
    return t, y


results = {h_s: euler_sir(h_s) for h_s in (1.0, 0.25, 0.001)}
t_ref, y_ref = results[0.001]

print()
print("=" * 65)
print("PART 4 - Modified case: SIR flu epidemic in a boarding school")
print("=" * 65)
print(f"R0 = beta/gamma = {BETA / GAMMA:.1f}")
print(f"{'Step h (days)':>14}{'Peak ill':>10}{'Peak day':>10}"
      f"{'Total infected':>16}{'S+I+R at end':>14}")
for h_s, (t_s, y_s) in results.items():
    peak = np.argmax(y_s[:, 1])
    total = N - y_s[-1, 0]
    print(f"{h_s:>14}{y_s[peak, 1]:>10.1f}{t_s[peak]:>10.2f}"
          f"{total:>16.1f}{y_s[-1].sum():>14.1f}")

t_1, y_1 = results[1.0]
peak_ref = np.argmax(y_ref[:, 1])

fig2, (ax_s, ax_z) = plt.subplots(1, 2, figsize=(12, 4.5))
for col, name, colour in ((0, "Susceptible", "C0"), (1, "Infected", "C3"),
                          (2, "Recovered", "C2")):
    ax_s.plot(t_ref, y_ref[:, col], color=colour, label=f"{name} (reference)")
    ax_s.plot(t_1, y_1[:, col], "o", color=colour, markersize=3,
              label=f"{name} (Euler h = 1)")
ax_s.set_xlabel("Day")
ax_s.set_ylabel("Students")
ax_s.set_title("SIR model of a school flu outbreak")
ax_s.legend(fontsize=7, loc="center right")
ax_s.grid(True)

for h_s, (t_s, y_s) in results.items():
    ax_z.plot(t_s, y_s[:, 1], label=f"h = {h_s:g} day")
ax_z.set_xlim(10, 35)
ax_z.set_xlabel("Day")
ax_z.set_ylabel("Students ill at the same time")
ax_z.set_title("Infected curve: effect of step size")
ax_z.legend()
ax_z.grid(True)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_17_sir_epidemic.png", dpi=120)

print()
print("Interpretation:")
print(f"* About {y_ref[peak_ref, 1]:.0f} students are ill at the same time at "
      f"the peak, around day {t_ref[peak_ref]:.0f}.")
print(f"  The sick bay must plan for roughly {y_ref[peak_ref, 1]:.0f} beds, "
      "or reduce contacts")
print("  (lower beta) to flatten the curve.")
print(f"* In total about {N - y_ref[-1, 0]:.0f} of 1000 students catch flu; "
      f"the rest are")
print("  protected because the epidemic burns out once too few susceptible")
print("  students are left (herd immunity), not because everyone was ill.")
print("* Euler with h = 1 day gets the overall shape right but shifts the")
print("  peak's size and timing compared with the fine reference; h = 0.25")
print("  is already close. S + I + R stays exactly 1000 for every h because")
print("  Euler's three updates add up to zero - the model conserves people.")
print("* Graph (left): susceptibles fall, recoveries rise, and the infected")
print("  curve forms a single wave. Graph (right): coarser steps change the")
print("  height and timing of the peak.")

plt.show()
