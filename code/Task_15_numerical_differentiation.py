"""
BIT 2118 Scientific Computing - Task N (Task 15)
Numerical Differentiation
=========================================================================

Program file : code/Task_15_numerical_differentiation.py
Run with     : python code/Task_15_numerical_differentiation.py
Output graphs: figures/Task_15_numerical_differentiation.png
               figures/Task_15_tank_flow.png     (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
A derivative f'(x) measures how fast a quantity changes. When only
sampled values are available, it is approximated from nearby samples
separated by a step h:

    Forward difference : f'(x) ~ [f(x+h) - f(x)] / h          error O(h)
    Backward difference: f'(x) ~ [f(x) - f(x-h)] / h          error O(h)
    Central difference : f'(x) ~ [f(x+h) - f(x-h)] / (2h)     error O(h^2)
    Second derivative  : f''(x) ~ [f(x+h) - 2f(x) + f(x-h)] / h^2

From Taylor series, the forward error is about (h/2) f''(x) and the
central error about (h^2/6) f'''(x), so central differences are usually
much more accurate for the same h.

Two error sources pull in opposite directions:
    * truncation error shrinks as h gets smaller;
    * round-off / measurement noise is DIVIDED by h, so it grows as h
      gets smaller.
There is therefore an optimal step size, and noisy data must be smoothed
before differentiating.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A GPS device records position at closely spaced times. Estimate the
instantaneous vehicle speed using central differences for interior
points (forward/backward at the ends) and compare step sizes.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    time     = 0, 1, ..., 6 s          (h = 1 s)
    position = 0.0, 4.8, 10.2, 16.1, 22.7, 30.0, 38.0 m
Operations:
    forward difference at t = 0, central differences at t = 1..5,
    backward difference at t = 6; np.gradient (does the same thing in
    one call); second differences for acceleration.

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def finite_difference_speed(t, x):
    """Forward diff at the start, central inside, backward at the end."""
    v = np.zeros_like(x)
    v[0] = (x[1] - x[0]) / (t[1] - t[0])
    for i in range(1, len(t) - 1):
        v[i] = (x[i + 1] - x[i - 1]) / (t[i + 1] - t[i - 1])
    v[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    return v


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# GPS samples: time in seconds and position in metres
time = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
position = np.array([0.0, 4.8, 10.2, 16.1, 22.7, 30.0, 38.0])

speed = np.zeros_like(position)

# Forward difference at the beginning (no earlier point exists)
speed[0] = (position[1] - position[0]) / (time[1] - time[0])

# Central differences for interior points
for i in range(1, len(time) - 1):
    speed[i] = (
        position[i + 1] - position[i - 1]
    ) / (
        time[i + 1] - time[i - 1]
    )

# Backward difference at the end (no later point exists)
speed[-1] = (
    position[-1] - position[-2]
) / (
    time[-1] - time[-2]
)

# np.gradient performs exactly the same scheme in one call
speed_np = np.gradient(position, time)

# Acceleration: second central difference at interior points
h = time[1] - time[0]
accel = (position[2:] - 2 * position[1:-1] + position[:-2]) / h**2

print("=" * 65)
print("PART 1 - Original solution: speed from GPS positions (h = 1 s)")
print("=" * 65)
for t, x, v in zip(time, position, speed):
    print(f"t={t:.0f}s, position={x:.1f}m, estimated speed={v:.2f}m/s")
print("np.gradient gives the same speeds:", np.allclose(speed, speed_np))
print("Acceleration at t = 1..5 s (m/s^2):", np.round(accel, 2))
print(f"Mean acceleration: {accel.mean():.2f} m/s^2")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the GPS sampling interval is changed from 1 s to 2 s (only
#   t = 0, 2, 4, 6 are kept).
#   Expected: fewer speed estimates, each an average over a longer
#   window. Central-difference error is about (h^2/6) x'''(t). The
#   acceleration here is almost constant, so position is nearly a
#   quadratic in t, x''' ~ 0, and the interior speeds should barely
#   change. The one-sided end values have error about (h/2) x'', so
#   doubling h should roughly double their error.
#
# Change 2: step size study on a function with a KNOWN derivative,
#   f(x) = sin(x) at x = 1 (true f'(1) = cos(1)), for h from 1 down to
#   1e-15, using forward and central differences.
#   Expected: error first falls as h shrinks (forward ~ h, central ~ h^2)
#   and then RISES again for very small h, because round-off in
#   f(x+h) - f(x) (about 1e-16) is divided by a tiny h. Best h is about
#   1e-8 for forward and about 1e-5 for central differences.
#
# Change 3: random GPS noise of +/- 0.3 m is added to the positions.
#   Expected: noise is amplified by differentiation. A central
#   difference subtracts two noisy values and divides by 2h, giving a
#   speed error with standard deviation sqrt(2) x 0.3 / 2 ~ 0.21 m/s;
#   the one-sided end values divide by h only, so ~ 0.42 m/s there.

t2, x2 = time[::2], position[::2]
speed_h2 = finite_difference_speed(t2, x2)

hs = np.logspace(-15, 0, 61)
true_derivative = np.cos(1.0)
forward_err = np.abs((np.sin(1 + hs) - np.sin(1)) / hs - true_derivative)
central_err = np.abs((np.sin(1 + hs) - np.sin(1 - hs)) / (2 * hs)
                     - true_derivative)
best_fwd = hs[np.argmin(forward_err)]
best_cen = hs[np.argmin(central_err)]

rng = np.random.default_rng(3)
noisy_position = position + rng.normal(0, 0.3, position.size)
speed_noisy = finite_difference_speed(time, noisy_position)

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print("Change 1 - sampling every 2 s instead of 1 s:")
for t, v in zip(t2, speed_h2):
    v1 = speed[int(t)]
    print(f"  t={t:.0f}s: h=2 s -> {v:.2f} m/s   (h=1 s gave {v1:.2f} m/s)")
print()
print("Change 2 - derivative of sin(x) at x = 1 (true value "
      f"{true_derivative:.10f}):")
print(f"{'h':>10}{'forward error':>16}{'central error':>16}")
for target in (1e-1, 1e-3, 1e-5, 1e-8, 1e-11, 1e-14):
    k = np.argmin(np.abs(hs - target))
    print(f"{hs[k]:>10.0e}{forward_err[k]:>16.2e}{central_err[k]:>16.2e}")
print(f"  best h: forward {best_fwd:.0e} (error {forward_err.min():.1e}),"
      f" central {best_cen:.0e} (error {central_err.min():.1e})")
print()
print("Change 3 - positions with +/-0.3 m GPS noise:")
print("  clean speeds:", np.round(speed, 2))
print("  noisy speeds:", np.round(speed_noisy, 2))
print(f"  largest speed error: {np.max(np.abs(speed_noisy - speed)):.2f} m/s")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Left graph : estimated speed against time (h = 1 s, h = 2 s, noisy).
# Right graph: error against step size h for the sin(x) test on log-log
#              axes, with reference slopes h and h^2.

fig, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_v.plot(time, speed, marker="o", label="h = 1 s (original)")
ax_v.plot(t2, speed_h2, marker="s", linestyle="--", label="h = 2 s")
ax_v.plot(time, speed_noisy, marker="^", linestyle=":",
          label="h = 1 s, +/-0.3 m noise")
ax_v.set_xlabel("Time (s)")
ax_v.set_ylabel("Estimated speed (m/s)")
ax_v.set_title("Speed estimated from GPS position data")
ax_v.legend()
ax_v.grid(True)

ax_h.loglog(hs, np.maximum(forward_err, 1e-17), label="Forward difference")
ax_h.loglog(hs, np.maximum(central_err, 1e-17), label="Central difference")
ax_h.loglog(hs, 0.5 * hs, "k:", linewidth=0.8, label="slope h (reference)")
ax_h.loglog(hs, 0.1 * hs**2, "k--", linewidth=0.8,
            label="slope h^2 (reference)")
ax_h.set_ylim(1e-13, 1e1)
ax_h.set_xlabel("Step size h")
ax_h.set_ylabel("|error| in derivative of sin at x = 1")
ax_h.set_title("Truncation vs round-off error")
ax_h.legend(fontsize=8)
ax_h.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_15_numerical_differentiation.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The vehicle speeds up steadily from about {speed[0]:.1f} m/s to "
      f"{speed[-1]:.1f} m/s")
print(f"  (about {speed[-1] * 3.6:.0f} km/h) in 6 s. The second differences "
      f"give an average")
print(f"  acceleration of {accel.mean():.2f} m/s^2 - a gentle, normal "
      "acceleration for a car.")
print("* The end values use one-sided differences, which are less accurate:")
print("  each one-sided value is really the AVERAGE speed over the first or")
print("  last second, so it lags the true speed at that instant.")
print("* Sampling every 2 s left the interior speeds almost unchanged")
print("  (acceleration is nearly constant, which central differences handle")
print("  exactly) but moved the end values by 0.3-0.35 m/s: one-sided")
print("  differences suffer most from a larger step.")
print("* Graph (right): both errors first fall along their reference lines")
print("  (forward ~ h, central ~ h^2), then rise again as h gets tiny,")
print("  because round-off is divided by h. Central differences reach a")
print(f"  far smaller minimum error ({central_err.min():.0e} vs "
      f"{forward_err.min():.0e}) at a larger h.")
print("* Graph (left): small GPS noise makes the speed curve jumpy - the")
print("  derivative amplifies noise, so real GPS speeds must be smoothed.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: estimating water inflow rate from a tank-level sensor.
#
# Problem:
#   A water-supply tank has a level sensor that reports the water depth
#   every 30 s. The utility wants the net flow rate into the tank
#   (inflow minus outflow) to detect pump failures or leaks. Flow is
#   the rate of change of volume:
#       Q(t) = dV/dt = A * dh/dt     (A = tank cross-section area)
#
# Assumptions:
#   * Cylindrical tank, cross-section A = 20 m^2.
#   * True level over one hour (used to create data and to check):
#       h(t) = 2 + 0.6 sin(2 pi t / 3600) + 0.00015 t   (metres)
#     i.e. a slow fill/drain cycle from changing demand plus a small
#     steady rise.
#   * The sensor has random noise with standard deviation 5 mm.
#   * Flow is reported in litres per second (1 m^3/s = 1000 L/s).
#
# Methods:
#   1. Central differences (np.gradient) on the raw noisy level data.
#   2. Savitzky-Golay filter (scipy.signal.savgol_filter): fits a local
#      quadratic to each window of 15 samples (7 minutes) by least
#      squares and differentiates that polynomial - smoothing and
#      differentiation in one step.
#
# Input data: 121 level readings (0-3600 s every 30 s).
# Output: flow-rate estimates, RMS error of each method.

AREA = 20.0                     # m^2
DT = 30.0                       # s between readings
t_tank = np.arange(0, 3600 + DT, DT)


def true_level(t):
    return 2 + 0.6 * np.sin(2 * np.pi * t / 3600) + 0.00015 * t


def true_flow_lps(t):
    """Exact Q = A dh/dt in litres per second (for checking)."""
    dhdt = 0.6 * (2 * np.pi / 3600) * np.cos(2 * np.pi * t / 3600) + 0.00015
    return AREA * dhdt * 1000


level_noisy = true_level(t_tank) + rng.normal(0, 0.005, t_tank.size)

flow_raw = AREA * np.gradient(level_noisy, DT) * 1000
flow_sg = AREA * savgol_filter(level_noisy, window_length=15, polyorder=2,
                               deriv=1, delta=DT) * 1000
flow_true = true_flow_lps(t_tank)

rms_raw = np.sqrt(np.mean((flow_raw - flow_true)**2))
rms_sg = np.sqrt(np.mean((flow_sg - flow_true)**2))

print()
print("=" * 65)
print("PART 4 - Modified case: tank inflow rate from a level sensor")
print("=" * 65)
print(f"Readings: {t_tank.size} (every {DT:.0f} s), sensor noise 5 mm")
print(f"True flow range: {flow_true.min():.1f} to {flow_true.max():.1f} L/s")
print(f"RMS error, raw central differences: {rms_raw:.2f} L/s")
print(f"RMS error, Savitzky-Golay         : {rms_sg:.2f} L/s")
print()
print(f"{'Time (min)':>10}{'True':>9}{'Raw':>9}{'Smoothed':>10}   (L/s)")
for k in range(0, t_tank.size, 20):
    print(f"{t_tank[k] / 60:>10.0f}{flow_true[k]:>9.2f}{flow_raw[k]:>9.2f}"
          f"{flow_sg[k]:>10.2f}")

fig2, (ax_l, ax_q) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_l.plot(t_tank / 60, level_noisy, ".", markersize=4, label="Sensor level")
ax_l.plot(t_tank / 60, true_level(t_tank), label="True level")
ax_l.set_xlabel("Time (min)")
ax_l.set_ylabel("Water depth (m)")
ax_l.set_title("Tank level measurements")
ax_l.legend()
ax_l.grid(True)

ax_q.plot(t_tank / 60, flow_raw, color="lightgray", marker=".",
          label="Raw central differences")
ax_q.plot(t_tank / 60, flow_sg, linewidth=2, label="Savitzky-Golay")
ax_q.plot(t_tank / 60, flow_true, "k--", label="True flow")
ax_q.axhline(0, color="black", linewidth=0.6)
ax_q.set_xlabel("Time (min)")
ax_q.set_ylabel("Net inflow Q (L/s)")
ax_q.set_title("Flow rate = A dh/dt")
ax_q.legend(fontsize=8)
ax_q.grid(True)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_15_tank_flow.png", dpi=120)

print()
print("Interpretation:")
print("* Positive Q means the tank is filling, negative means it is")
print(f"  draining. The true flow swings between about "
      f"{flow_true.max():+.0f} and {flow_true.min():+.0f} L/s over")
print("  the hour as demand changes.")
print(f"* A 5 mm sensor noise looks tiny on the level graph, but raw central")
print(f"  differences divide it by 2 x 30 s and multiply by 20 m^2, giving")
print(f"  flow errors of {rms_raw:.1f} L/s RMS - too jumpy to detect a leak.")
print(f"* Savitzky-Golay smoothing cut the error to {rms_sg:.1f} L/s, about "
      f"{rms_raw / rms_sg:.0f}x better, while")
print("  still following the real rise and fall of the flow.")
print("* Graph (right): grey raw estimates scatter widely; the smoothed")
print("  curve tracks the dashed true flow. Differentiate measured data only")
print("  after smoothing or fitting it.")

plt.show()
