"""
BIT 2118 Scientific Computing - Task O (Task 16)
Numerical Integration
=========================================================================

Program file : code/Task_16_numerical_integration.py
Run with     : python code/Task_16_numerical_integration.py
Output graphs: figures/Task_16_numerical_integration.png
               figures/Task_16_wind_energy.png   (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
An integral is an accumulated quantity, or the area under a curve:
        E = integral from a to b of P(t) dt
With samples f_0, f_1, ..., f_n at equal spacing h:

Trapezoidal rule - join the samples with straight lines and add the
trapezoid areas:
        T = h [ f_0/2 + f_1 + f_2 + ... + f_{n-1} + f_n/2 ]
    error ~ -(b - a) h^2 f''/12   (halving h cuts the error ~4x)

Simpson's rule - fit a parabola through each pair of intervals
(needs an even number of intervals for the classic form):
        S = h/3 [ f_0 + 4 f_1 + 2 f_2 + 4 f_3 + ... + 4 f_{n-1} + f_n ]
    error ~ -(b - a) h^4 f''''/180  (halving h cuts the error ~16x)

Adaptive quadrature (scipy.integrate.quad) evaluates a FUNCTION (not
fixed samples) and keeps subdividing where the integrand is hard until
an error estimate is below a tolerance.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A smart meter records power at regular intervals. Estimate the total
electrical energy consumed by integrating power over time with the
trapezoidal and Simpson's rules.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    time_hours = 0, 2, 4, ..., 12   (h = 2 h, 6 intervals)
    power_kw   = 0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9
Operations:
    np.trapezoid                - trapezoidal rule
    scipy.integrate.simpson     - Simpson's rule
    kW x h = kWh (energy units)
    scipy.integrate.quad        - adaptive reference in Part 2

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, simpson

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Smart-meter readings taken every two hours
time_hours = np.array([0, 2, 4, 6, 8, 10, 12], dtype=float)
power_kw = np.array([0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9])

energy_trapezoid = np.trapezoid(power_kw, time_hours)
energy_simpson = simpson(power_kw, x=time_hours)

# The same trapezoidal rule written out from the formula, as a check
h = time_hours[1] - time_hours[0]
manual_trap = h * (power_kw[0] / 2 + power_kw[1:-1].sum() + power_kw[-1] / 2)
manual_simp = h / 3 * (power_kw[0] + 4 * power_kw[1:-1:2].sum()
                       + 2 * power_kw[2:-1:2].sum() + power_kw[-1])

TARIFF_KES = 25.0     # assumed electricity price per kWh

print("=" * 65)
print("PART 1 - Original solution: smart-meter energy over 12 hours")
print("=" * 65)
print(f"Energy using trapezoidal rule: {energy_trapezoid:.3f} kWh "
      f"(by hand: {manual_trap:.3f})")
print(f"Energy using Simpson's rule:   {energy_simpson:.3f} kWh "
      f"(by hand: {manual_simp:.3f})")
print(f"Difference between estimates: "
      f"{abs(energy_simpson - energy_trapezoid):.3f} kWh")
print(f"Average power: {energy_simpson / 12:.3f} kW")
print(f"Cost at KES {TARIFF_KES:.0f}/kWh: KES {energy_simpson * TARIFF_KES:.2f}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the meter is set to record every 4 hours instead of every 2
#   (only t = 0, 4, 8, 12 are kept).
#   Expected: coarser sampling misses the 6:00 peak (1.8 kW). With only
#   4 samples each rule has to guess the shape between readings, so the
#   two rules should disagree much more than they did with 2-hour data.
#   (4 samples = 3 intervals, an odd number; scipy's simpson handles the
#   last interval with a special correction.)
#
# Change 2: the 6:00 reading is raised from 1.8 kW to 2.6 kW (e.g. an
#   electric oven running).
#   Expected: extra energy = (extra height) x (weight of that sample):
#     trapezoid weight = h = 2 h       ->  +0.8 x 2       = +1.6 kWh
#     Simpson weight   = 4h/3 = 2.67 h ->  +0.8 x 8/3     = +2.13 kWh
#   (Simpson gives odd-numbered samples a larger weight.)
#
# Change 3: convergence study on a smooth load profile with a KNOWN
#   exact integral:  P(t) = 1 + 0.5 sin(pi t / 12)  kW over 0-12 h
#   (load rises to a midday peak and falls again),
#   exact energy = 12 + 0.5 x (12/pi) x [1 - cos(pi)] = 12 + 12/pi kWh.
#   Expected: trapezoid error falls ~4x and Simpson ~16x each time the
#   number of intervals doubles, until Simpson hits round-off.
#   (A profile covering exactly whole periods of a sine/cosine would be a
#   poor test: the trapezoidal rule is then exact by symmetry.)

t4, p4 = time_hours[::2], power_kw[::2]
trap_4h = np.trapezoid(p4, t4)
simp_4h = simpson(p4, x=t4)

power_peak = power_kw.copy()
power_peak[3] = 2.6
trap_peak = np.trapezoid(power_peak, time_hours)
simp_peak = simpson(power_peak, x=time_hours)


def smooth_load(t):
    """Smooth daily load profile used for the convergence study."""
    return 1 + 0.5 * np.sin(np.pi * t / 12)


EXACT = 12 + 12 / np.pi
quad_value, quad_error_estimate = quad(smooth_load, 0, 12)
intervals = np.array([2, 4, 8, 16, 32, 64, 128, 256])
trap_errors, simp_errors = [], []
for n in intervals:
    t = np.linspace(0, 12, n + 1)
    trap_errors.append(abs(np.trapezoid(smooth_load(t), t) - EXACT))
    simp_errors.append(abs(simpson(smooth_load(t), x=t) - EXACT))
trap_errors, simp_errors = np.array(trap_errors), np.array(simp_errors)

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print("Change 1 - readings every 4 h instead of 2 h:")
print(f"  trapezoid {energy_trapezoid:.3f} -> {trap_4h:.3f} kWh,  "
      f"Simpson {energy_simpson:.3f} -> {simp_4h:.3f} kWh")
print("Change 2 - 6:00 reading 1.8 -> 2.6 kW:")
print(f"  trapezoid +{trap_peak - energy_trapezoid:.3f} kWh, "
      f"Simpson +{simp_peak - energy_simpson:.3f} kWh")
print(f"Change 3 - smooth profile, exact energy {EXACT:.10f} kWh:")
print(f"  quad: {quad_value:.10f} kWh (its own error estimate "
      f"{quad_error_estimate:.1e})")
print(f"{'intervals':>10}{'trap error':>13}{'ratio':>7}"
      f"{'Simpson error':>15}{'ratio':>7}")
for k, n in enumerate(intervals):
    tr = trap_errors[k - 1] / trap_errors[k] if k else np.nan
    sr = simp_errors[k - 1] / simp_errors[k] if k and simp_errors[k] else np.nan
    print(f"{n:>10}{trap_errors[k]:>13.2e}{tr:>7.1f}"
          f"{simp_errors[k]:>15.2e}{sr:>7.1f}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Left graph : the meter readings with the trapezoids shaded - the
#              shaded area IS the trapezoidal energy estimate - and the
#              Simpson parabolas drawn through each pair of intervals.
# Right graph: error against number of intervals (log-log) with slopes
#              of -2 (trapezoid) and -4 (Simpson).

fig, (ax_p, ax_e) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_p.fill_between(time_hours, power_kw, alpha=0.25,
                  label=f"Trapezoid area = {energy_trapezoid:.2f} kWh")
ax_p.plot(time_hours, power_kw, "o-", color="C0", label="Meter readings")
for i in range(0, len(time_hours) - 2, 2):
    xs = time_hours[i:i + 3]
    coeffs = np.polyfit(xs, power_kw[i:i + 3], 2)
    xf = np.linspace(xs[0], xs[-1], 50)
    ax_p.plot(xf, np.polyval(coeffs, xf), "--", color="C1",
              label="Simpson parabolas" if i == 0 else None)
ax_p.plot(t4, p4, "s:", color="C2", label="Every 4 h (change 1)")
ax_p.set_xlabel("Time (hours)")
ax_p.set_ylabel("Power (kW)")
ax_p.set_title("Energy = area under the power curve")
ax_p.legend(fontsize=8, loc="lower center")
ax_p.grid(True)

ax_e.loglog(intervals, trap_errors, "o-", label="Trapezoid")
ax_e.loglog(intervals, np.maximum(simp_errors, 1e-16), "s-",
            label="Simpson")
ax_e.loglog(intervals, trap_errors[0] * (intervals[0] / intervals)**2, "k:",
            label="slope -2")
ax_e.loglog(intervals, simp_errors[0] * (intervals[0] / intervals)**4,
            "k--", label="slope -4")
ax_e.set_xlabel("Number of intervals")
ax_e.set_ylabel("|error| (kWh)")
ax_e.set_title("Convergence on a smooth load profile")
ax_e.legend()
ax_e.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_16_numerical_integration.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The household used about {energy_simpson:.1f} kWh in 12 hours "
      f"(average {energy_simpson / 12:.2f} kW),")
print(f"  costing about KES {energy_simpson * TARIFF_KES:.0f} at "
      f"KES {TARIFF_KES:.0f}/kWh.")
print("* Trapezoid and Simpson differ by only "
      f"{abs(energy_simpson - energy_trapezoid):.3f} kWh, so either is good")
print("  enough for billing with this sampling rate.")
print("* With 4-hour readings the 6:00 peak is missed and the two rules")
print(f"  now disagree by {abs(simp_4h - trap_4h):.2f} kWh instead of "
      f"{abs(energy_simpson - energy_trapezoid):.2f}: trapezoid fell to "
      f"{trap_4h:.2f}")
print(f"  and Simpson rose to {simp_4h:.2f}. With so few samples neither "
      "can be trusted -")
print("  the sampling interval matters more than the choice of rule.")
print("* Graph (left): the shaded area under the straight segments is the")
print("  trapezoidal energy; the dashed parabolas show Simpson's curved fit.")
print("  The 4-hour line cuts off the peak. Graph (right): trapezoid errors")
print("  follow slope -2 (error ~ h^2) and Simpson slope -4 (error ~ h^4),")
print("  so Simpson reaches any given accuracy with far fewer samples")
print("  (with 256 intervals: Simpson error ~5e-10 vs trapezoid ~5e-5).")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: expected annual energy from a wind turbine (probability +
# energy integral).
#
# Problem:
#   A 2 MW turbine is planned for a windy site. Wind speed varies, so the
#   expected power output is an integral over all wind speeds of
#       (turbine power at speed v) x (probability density of speed v):
#           E[P] = integral_0^inf P(v) f(v) dv
#   Annual energy = E[P] x 8760 hours.
#
# Assumptions:
#   * Wind speed follows a Weibull distribution (standard for wind
#     resource studies) with shape k = 2 and scale c = 7.5 m/s:
#         f(v) = (k/c) (v/c)^(k-1) exp(-(v/c)^k)
#   * Simplified power curve:
#         0 kW                         for v < 3 m/s   (cut-in)
#         2000 (v^3 - 27)/(12^3 - 27)  for 3 <= v < 12 (power ~ v^3)
#         2000 kW                      for 12 <= v <= 25 (rated)
#         0 kW                         for v > 25 m/s  (cut-out, safety)
#   * No losses or downtime.
#
# Methods:
#   * quad (adaptive) with the power-curve corners given as breakpoints
#     so it does not have to discover the jumps itself.
#   * Trapezoid and Simpson on a grid of wind speeds 0-30 m/s.
#   * Check: integral of f(v) over 0-inf must equal 1 (total probability).

K_SHAPE, C_SCALE = 2.0, 7.5
RATED_KW, V_IN, V_RATED, V_OUT = 2000.0, 3.0, 12.0, 25.0


def weibull_pdf(v):
    return (K_SHAPE / C_SCALE) * (v / C_SCALE)**(K_SHAPE - 1) \
        * np.exp(-(v / C_SCALE)**K_SHAPE)


def power_curve(v):
    v = np.asarray(v, dtype=float)
    p = np.where((v >= V_IN) & (v < V_RATED),
                 RATED_KW * (v**3 - V_IN**3) / (V_RATED**3 - V_IN**3), 0.0)
    return np.where((v >= V_RATED) & (v <= V_OUT), RATED_KW, p)


def integrand(v):
    return power_curve(v) * weibull_pdf(v)


total_probability, _ = quad(weibull_pdf, 0, np.inf)
mean_power_quad, _ = quad(integrand, 0, 30, points=[V_IN, V_RATED, V_OUT],
                          limit=200)

v_grid = np.linspace(0, 30, 61)              # every 0.5 m/s
mean_power_trap = np.trapezoid(integrand(v_grid), v_grid)
mean_power_simp = simpson(integrand(v_grid), x=v_grid)

# Same rules on a slightly different grid (60 points, spacing 0.508 m/s)
# whose nodes do NOT land on the 3, 12 and 25 m/s corners of the curve.
v_off = np.linspace(0, 30, 60)
mean_power_trap_off = np.trapezoid(integrand(v_off), v_off)
mean_power_simp_off = simpson(integrand(v_off), x=v_off)

HOURS_PER_YEAR = 8760
annual_mwh = mean_power_quad * HOURS_PER_YEAR / 1000
capacity_factor = mean_power_quad / RATED_KW

# Probability of the different operating zones (integrals of the pdf)
p_idle, _ = quad(weibull_pdf, 0, V_IN)
p_rated, _ = quad(weibull_pdf, V_RATED, V_OUT)
p_cutout, _ = quad(weibull_pdf, V_OUT, np.inf)

print()
print("=" * 65)
print("PART 4 - Modified case: expected wind-turbine energy")
print("=" * 65)
print(f"Check - total probability of Weibull pdf: {total_probability:.10f}")
print(f"Mean power, quad (adaptive)   : {mean_power_quad:.2f} kW")
print(f"Mean power, trapezoid (0.5 m/s grid): {mean_power_trap:.2f} kW")
print(f"Mean power, Simpson   (0.5 m/s grid): {mean_power_simp:.2f} kW")
print(f"Off-grid (0.508 m/s): trapezoid {mean_power_trap_off:.2f} kW, "
      f"Simpson {mean_power_simp_off:.2f} kW")
print(f"Annual energy: {annual_mwh:,.0f} MWh")
print(f"Capacity factor: {capacity_factor:.1%}")
print(f"Time with wind below cut-in (idle) : {p_idle:.1%}")
print(f"Time at full rated power            : {p_rated:.1%}")
print(f"Time above cut-out (shut down)      : {p_cutout:.4%}")

v_plot = np.linspace(0, 30, 600)
fig2, (ax_w, ax_i) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_w.plot(v_plot, weibull_pdf(v_plot), color="C0")
ax_w.set_xlabel("Wind speed (m/s)")
ax_w.set_ylabel("Probability density", color="C0")
ax_w.set_title("Wind distribution and power curve")
ax_w2 = ax_w.twinx()
ax_w2.plot(v_plot, power_curve(v_plot), color="C1")
ax_w2.set_ylabel("Turbine power (kW)", color="C1")
ax_w.grid(True, alpha=0.4)

ax_i.fill_between(v_plot, integrand(v_plot), alpha=0.35)
ax_i.plot(v_plot, integrand(v_plot))
for v_mark in (V_IN, V_RATED, V_OUT):
    ax_i.axvline(v_mark, color="gray", linestyle=":")
ax_i.set_xlabel("Wind speed (m/s)")
ax_i.set_ylabel("P(v) f(v)  (kW per m/s)")
ax_i.set_title(f"Area = mean power = {mean_power_quad:.0f} kW")
ax_i.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_16_wind_energy.png", dpi=120)

print()
print("Interpretation:")
print(f"* On average the turbine produces {mean_power_quad:.0f} kW - only "
      f"{capacity_factor:.0%} of its 2 MW rating -")
print(f"  about {annual_mwh:,.0f} MWh a year. A capacity factor in this "
      "range is typical for")
print("  onshore wind; investors use this number, not the rated power.")
print(f"* The wind is too weak to turn the blades {p_idle:.0%} of the time "
      f"and strong enough")
print(f"  for full power only {p_rated:.0%} of the time.")
trap_diff = abs(mean_power_trap - mean_power_quad) / mean_power_quad
simp_diff = abs(mean_power_simp - mean_power_quad) / mean_power_quad
print(f"* Trapezoid and Simpson differ from quad by {trap_diff:.3%} and "
      f"{simp_diff:.4%}.")
print("  The integrand has a corner at 12 m/s and a jump at 25 m/s, which")
print("  normally spoils Simpson's high accuracy. It works well here only")
print("  because the 0.5 m/s grid puts samples exactly on 3, 12 and 25 m/s:")
off = abs(mean_power_simp_off - mean_power_quad) / mean_power_quad
print(f"  on a 0.508 m/s grid Simpson's error grows to {off:.3%}, no better")
print("  than the trapezoid. quad was told the breakpoints explicitly - the")
print("  safest approach for functions with corners or jumps.")
print("* Graph (right): the shaded area is the expected power. Most of it")
print("  comes from 7-15 m/s winds - common enough AND strong enough.")

plt.show()
