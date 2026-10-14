"""
BIT 2118 Scientific Computing - Task L (Task 13)
Interpolation
=========================================================================

Program file : code/Task_13_interpolation.py
Run with     : python code/Task_13_interpolation.py
Output graphs: figures/Task_13_interpolation.png
               figures/Task_13_thermistor.png    (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Interpolation estimates values BETWEEN known data points (x_i, y_i). The
interpolating function passes exactly through every data point.

Linear interpolation joins neighbouring points with straight lines. For
x_i <= x <= x_{i+1}:
        y(x) = y_i + (y_{i+1} - y_i) (x - x_i) / (x_{i+1} - x_i)
Simple and safe, but the slope jumps at every data point (corners).

Polynomial interpolation fits a single polynomial of degree n-1 through
all n points. With many points it can oscillate wildly between them
(Runge's phenomenon).

Cubic spline: a separate cubic polynomial on each interval, joined so
that the value, slope and curvature are continuous at every data point.
This gives a smooth curve without the wild oscillation of one big
polynomial. scipy's CubicSpline uses the "not-a-knot" end condition by
default.

Extrapolation (estimating OUTSIDE the data range) is much less reliable
than interpolation because nothing constrains the curve there.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A sensor records temperature every hour, but an application needs
estimates every 10 minutes. Build interpolation functions from the
hourly measurements and evaluate them at the intermediate times.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    hours       = 8, 9, ..., 14 (hour of day)
    temperature = 18.0, 20.0, 22.5, 24.0, 25.0, 25.5, 25.0 (C)
    query times = every 10 minutes (1/6 hour) from 8:00 to 14:00
    target time = 10.5 (10:30)
Operations:
    scipy.interpolate.interp1d(kind="linear") - piecewise linear
    scipy.interpolate.CubicSpline              - piecewise cubic
    np.arange to build the 10-minute time grid

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def clock(hour):
    """Format a decimal hour such as 10.5 as '10:30'."""
    h = int(hour)
    m = int(round((hour - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h:02d}:{m:02d}"


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Hourly temperature measurements
hours = np.array([8, 9, 10, 11, 12, 13, 14], dtype=float)
temperature = np.array([18.0, 20.0, 22.5, 24.0, 25.0, 25.5, 25.0])

linear = interp1d(hours, temperature, kind="linear")
cubic = CubicSpline(hours, temperature)

# Estimate every 10 minutes = 1/6 hour: 6 hours x 6 steps + 1 = 37 times.
# np.linspace is used instead of np.arange(8, 14 + 1/6, 1/6) because
# arange with a fractional step can, through round-off, produce a last
# value slightly above 14, which interp1d would reject as out of range.
ten_minute_times = np.linspace(8, 14, 37)

linear_estimates = linear(ten_minute_times)
cubic_estimates = cubic(ten_minute_times)

target_time = 10.5  # 10:30

print("=" * 65)
print("PART 1 - Original solution: hourly -> 10-minute temperatures")
print("=" * 65)
print(f"Number of 10-minute estimates: {len(ten_minute_times)}")
print(f"Linear estimate at 10:30: {float(linear(target_time)):.2f} C")
print(f"Cubic estimate at 10:30: {float(cubic(target_time)):.2f} C")
print()
print("Both pass exactly through the measurements:",
      np.allclose(linear(hours), temperature)
      and np.allclose(cubic(hours), temperature))
print()
print(f"{'Time':<7}{'Linear':>9}{'Spline':>9}{'Diff':>8}")
for t, lin, cub in list(zip(ten_minute_times, linear_estimates,
                            cubic_estimates))[12:19]:      # 10:00-11:00
    print(f"{clock(t):<7}{lin:>9.3f}{cub:>9.3f}{cub - lin:>8.3f}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the 11:00 measurement is removed (e.g. the sensor missed a
#   reading) and both methods are used to estimate it from the others.
#   Expected: linear interpolation simply averages 10:00 and 12:00
#   (22.5 + 25.0)/2 = 23.75 C, which is below the real 24.0 C because
#   temperature rises along a curve that bends downwards (concave). The
#   spline follows the curvature of the neighbouring points and should
#   land closer to 24.0 C. This "leave one out" test measures how well
#   each method fills a gap.
#
# Change 2: the target time is moved OUTSIDE the data, to 15:00
#   (extrapolation).
#   Expected: interp1d refuses by default (raises an error), which is a
#   safety feature. CubicSpline extends its last cubic piece, giving a
#   number that may be unrealistic because no data constrains it.

mask = hours != 11
linear_gap = interp1d(hours[mask], temperature[mask], kind="linear")
cubic_gap = CubicSpline(hours[mask], temperature[mask])

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print("Change 1 - 11:00 reading removed, then estimated (true 24.0 C):")
print(f"  Linear estimate: {float(linear_gap(11)):.3f} C "
      f"(error {float(linear_gap(11)) - 24.0:+.3f})")
print(f"  Spline estimate: {float(cubic_gap(11)):.3f} C "
      f"(error {float(cubic_gap(11)) - 24.0:+.3f})")
print(f"  Spline 10:30 estimate with gap: {float(cubic_gap(10.5)):.3f} C "
      f"(was {float(cubic(10.5)):.3f} C)")

print()
print("Change 2 - extrapolate to 15:00:")
try:
    linear(15.0)
except ValueError as error:
    print(f"  interp1d (linear): refused - {error}")
print(f"  CubicSpline: {float(cubic(15.0)):.2f} C, and at 17:00: "
      f"{float(cubic(17.0)):.2f} C")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Left graph : hourly data, linear and spline curves (original problem),
#              the estimates at 10:30 marked.
# Right graph: the gap-filling test and the extrapolation beyond 14:00.

fig, (ax_main, ax_test) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_main.scatter(hours, temperature, color="black", zorder=3,
                label="Hourly measurements")
ax_main.plot(ten_minute_times, linear_estimates, label="Linear")
ax_main.plot(ten_minute_times, cubic_estimates, label="Cubic spline")
ax_main.plot([target_time] * 2,
             [float(linear(target_time)), float(cubic(target_time))],
             "rx", markersize=9, mew=2, label="Estimates at 10:30")
ax_main.set_xlabel("Hour of day")
ax_main.set_ylabel("Temperature (C)")
ax_main.set_title("Temperature interpolation")
ax_main.legend()
ax_main.grid(True)

t_ext = np.linspace(8, 17, 300)
t_in = np.linspace(8, 14, 300)
ax_test.scatter(hours[mask], temperature[mask], color="black", zorder=3,
                label="Data used (11:00 removed)")
ax_test.scatter([11], [24.0], marker="*", s=150, color="gold",
                edgecolors="black", zorder=4, label="True 11:00 value")
ax_test.plot(t_in, linear_gap(t_in), label="Linear without 11:00")
ax_test.plot(t_in, cubic_gap(t_in), label="Spline without 11:00")
ax_test.plot(t_ext[t_ext >= 14], cubic(t_ext[t_ext >= 14]), "r--",
             label="Spline extrapolation")
ax_test.axvspan(14, 17, color="red", alpha=0.07)
ax_test.set_xlabel("Hour of day")
ax_test.set_ylabel("Temperature (C)")
ax_test.set_title("Gap filling and extrapolation")
ax_test.legend(fontsize=8)
ax_test.grid(True)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_13_interpolation.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* At 10:30 the linear estimate is {float(linear(10.5)):.2f} C and "
      f"the spline {float(cubic(10.5)):.2f} C.")
print("  Linear assumes steady warming within each hour; the spline follows")
print("  the curving trend of the whole morning. The two differ by at most")
print(f"  {np.max(np.abs(cubic_estimates - linear_estimates)):.2f} C, so "
      "either is fine for a display, but the spline")
print("  is smoother and has no sudden changes in warming rate on the hour.")
print("* Filling the missing 11:00 value, the spline was closer to the true")
print("  reading than linear interpolation, because it uses the curvature")
print("  of nearby points.")
print("* Beyond 14:00 the spline keeps following its last cubic piece and")
print(f"  predicts {float(cubic(17.0)):.1f} C at 17:00 - a guess with no "
      "data behind it.")
print("  Interpolation is trustworthy inside the data range only.")
print("* Graph (left): both curves pass through every black dot; the spline")
print("  is a smooth curve, linear is a chain of straight segments. Graph")
print("  (right): the star is the removed true value; the red shaded zone")
print("  is extrapolation, where estimates become unreliable.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: thermistor sensor calibration with a lookup table.
#
# Problem:
#   A cheap NTC thermistor's resistance falls steeply and NON-linearly as
#   temperature rises. The manufacturer's datasheet gives resistance only
#   every 10 C (a lookup table). A microcontroller reads resistance and
#   must convert it to temperature for any reading in between.
#
# Assumptions:
#   * The thermistor follows the Beta model (used here as the "truth" to
#     generate the table and check the answers):
#         R(T) = R0 exp( B (1/T - 1/T0) ),  T in kelvin
#     with R0 = 10 kOhm at T0 = 25 C and B = 3950 K (common values).
#   * Datasheet table: 0, 10, 20, ..., 100 C.
#   * Test readings: 400 resistances across the whole range.
#
# Input data: (R, T) table points; measured resistance readings.
#
# Methods compared (temperature as a function of resistance):
#   1. Linear interpolation in R
#   2. Cubic spline in R
#   3. Cubic spline in ln(R). Because R depends exponentially on 1/T,
#      ln(R) is close to linear in 1/T - a much smoother relationship -
#      so interpolating against ln(R) should be far more accurate. This
#      shows that choosing the right variable can matter more than
#      choosing the interpolation method.
#
# Note: interp1d / CubicSpline need increasing x values. R decreases
# with T, so the table is reversed (sorted by increasing R) first.

R0, T0_K, BETA = 10_000.0, 298.15, 3950.0


def beta_resistance(temp_c):
    """True thermistor resistance (ohm) at temperature temp_c (C)."""
    T = temp_c + 273.15
    return R0 * np.exp(BETA * (1 / T - 1 / T0_K))


def beta_temperature(resistance):
    """Exact inverse of the Beta model: temperature (C) from resistance."""
    inv_T = 1 / T0_K + np.log(resistance / R0) / BETA
    return 1 / inv_T - 273.15


table_T = np.arange(0, 101, 10, dtype=float)
table_R = beta_resistance(table_T)

order = np.argsort(table_R)                 # increasing resistance
R_sorted, T_sorted = table_R[order], table_T[order]

lin_R = interp1d(R_sorted, T_sorted, kind="linear")
spline_R = CubicSpline(R_sorted, T_sorted)
spline_lnR = CubicSpline(np.log(R_sorted), T_sorted)

readings = np.geomspace(table_R.min(), table_R.max(), 400)
true_T = beta_temperature(readings)
estimates = {
    "Linear in R": lin_R(readings),
    "Spline in R": spline_R(readings),
    "Spline in ln(R)": spline_lnR(np.log(readings)),
}

print()
print("=" * 65)
print("PART 4 - Modified case: thermistor calibration lookup table")
print("=" * 65)
print("Datasheet table:")
print(f"{'T (C)':>8}" + "".join(f"{t:>8.0f}" for t in table_T))
print(f"{'R (kOhm)':>8}" + "".join(f"{r / 1000:>8.2f}" for r in table_R))
print()
print(f"{'Method':<18}{'Max error (C)':>15}{'Mean error (C)':>16}")
for name, est in estimates.items():
    err = np.abs(est - true_T)
    print(f"{name:<18}{err.max():>15.2e}{err.mean():>16.2e}")

example_R = 5_000.0
print()
print(f"Example reading {example_R / 1000:.1f} kOhm: true "
      f"{float(beta_temperature(example_R)):.3f} C")
for name, fn in (("Linear in R", lambda r: lin_R(r)),
                 ("Spline in R", lambda r: spline_R(r)),
                 ("Spline in ln(R)", lambda r: spline_lnR(np.log(r)))):
    print(f"  {name:<16}: {float(fn(example_R)):.3f} C")

fig2, (ax_curve, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_curve.plot(readings / 1000, true_T, "k", linewidth=3, alpha=0.3,
              label="True (Beta model)")
for name, est in estimates.items():
    ax_curve.plot(readings / 1000, est, label=name)
ax_curve.scatter(table_R / 1000, table_T, color="black", zorder=3,
                 label="Datasheet points")
ax_curve.set_xscale("log")
ax_curve.set_xlabel("Resistance (kOhm, log scale)")
ax_curve.set_ylabel("Temperature (C)")
ax_curve.set_title("Thermistor: temperature from resistance")
ax_curve.legend(fontsize=8)
ax_curve.grid(True, which="both", alpha=0.4)

for name, est in estimates.items():
    ax_err.semilogy(true_T, np.maximum(np.abs(est - true_T), 1e-6),
                    label=name)
ax_err.set_xlabel("True temperature (C)")
ax_err.set_ylabel("Absolute error (C)")
ax_err.set_title("Interpolation error across the range")
ax_err.legend(fontsize=8)
ax_err.grid(True, which="both", alpha=0.4)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_13_thermistor.png", dpi=120)

errs = {name: np.abs(est - true_T).max() for name, est in estimates.items()}
print()
print("Interpretation:")
print(f"* Linear interpolation in R is off by 0.4 to "
      f"{errs['Linear in R']:.2f} C between EVERY pair of")
print("  table points, because the true curve bends throughout the range.")
print(f"* A spline in R is accurate to about 0.01 C in the middle of the")
print(f"  range, but still off by up to {errs['Spline in R']:.2f} C between "
      "0 and 10 C, where")
print("  the curve bends most sharply and the table points are far apart.")
print(f"* A spline in ln(R) reduces it to {errs['Spline in ln(R)']:.1e} C - "
      "far below any")
print("  sensor's accuracy - from only 11 table entries. Transforming the")
print("  input so the relationship is nearly linear made the biggest gain.")
print("* Graph (left): all methods pass through the datasheet points; on")
print("  the log scale the true curve is gentle. Graph (right): error is")
print("  zero at each table point (dips) and largest between them; the")
print("  ln(R) spline line sits far below the others.")

plt.show()
