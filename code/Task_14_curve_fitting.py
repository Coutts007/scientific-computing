"""
BIT 2118 Scientific Computing - Task M (Task 14)
Curve Fitting and Approximation
=========================================================================

Program file : code/Task_14_curve_fitting.py
Run with     : python code/Task_14_curve_fitting.py
Output graphs: figures/Task_14_curve_fitting.png
               figures/Task_14_growth_curve.png  (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Curve fitting chooses the parameters of a model y_hat = f(x; p) so that
it follows observed data as closely as possible. Unlike interpolation,
the curve does NOT have to pass through every point - real data contains
noise.

Least squares chooses the parameters p that minimise the sum of squared
residuals:
        S(p) = sum_i ( y_i - f(x_i; p) )^2
scipy.optimize.curve_fit does this (Levenberg-Marquardt algorithm for
nonlinear models) and also returns the covariance matrix of the
parameters; the square roots of its diagonal are the parameters'
standard errors (how uncertain each one is).

Measures of fit quality:
    RMSE = sqrt( mean( (y - y_hat)^2 ) )   typical size of an error,
                                           in the units of y
    R^2  = 1 - S / sum (y - mean(y))^2     share of the variation in y
                                           that the model explains (1 =
                                           perfect)
Residual plots should look like random scatter around zero; a pattern
(e.g. a U-shape) means the model shape is wrong.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
Measure server response time at different user loads and estimate a
mathematical relationship between load and response time. Fit the model,
plot observed and fitted values and inspect the residuals.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    users       = 50, 100, ..., 350 concurrent users
    response_ms = 82, 95, 115, 145, 185, 238, 305 milliseconds
    model       = quadratic  y = a x^2 + b x + c
Operations:
    curve_fit (least squares), RMSE, R^2, parameter standard errors from
    the covariance matrix, prediction at 400 users (extrapolation).

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def fit_quality(y, y_hat):
    """Return (RMSE, R^2) of predictions y_hat against observations y."""
    residuals = y - y_hat
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)
    return rmse, r2


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Concurrent users and measured server response time (milliseconds)
users = np.array([50, 100, 150, 200, 250, 300, 350], dtype=float)
response_ms = np.array([82, 95, 115, 145, 185, 238, 305], dtype=float)


# A simple quadratic model for increasing delay
def model(x, a, b, c):
    return a*x**2 + b*x + c


parameters, covariance = curve_fit(model, users, response_ms)
a, b, c = parameters
std_errors = np.sqrt(np.diag(covariance))

predicted = model(users, a, b, c)
rmse, r2 = fit_quality(response_ms, predicted)

print("=" * 65)
print("PART 1 - Original solution: quadratic response-time model")
print("=" * 65)
print(f"Fitted model: y = {a:.6f}x^2 {b:+.4f}x {c:+.2f}")
for name, value, se in zip("abc", parameters, std_errors):
    print(f"  {name} = {value:.6g} +/- {se:.2g} (standard error)")
print(f"RMSE: {rmse:.2f} ms")
print(f"R^2 : {r2:.5f}")
print(f"Predicted response at 400 users: {model(400, a, b, c):.2f} ms")
print()
print(f"{'Users':>6}{'Measured':>10}{'Fitted':>10}{'Residual':>10}")
for x, y, yh in zip(users, response_ms, predicted):
    print(f"{x:>6.0f}{y:>10.1f}{yh:>10.2f}{y - yh:>+10.2f}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the model is changed from quadratic to a straight line
#   y = m x + k.
#   Expected: a worse fit (larger RMSE, lower R^2) and a clear U-shaped
#   pattern in the residuals - the line is too high in the middle and too
#   low at both ends - because response time curves upward as load
#   grows. The straight line also under-predicts the 400-user load.
#
# Change 2: the last measurement (350 users) is changed from 305 ms to
#   400 ms, as if a garbage-collection pause made one test slow.
#   Expected: the fitted curve bends more steeply to chase that one
#   point, the RMSE rises, and the 400-user prediction jumps. Points at
#   the edge of the data have strong influence ("leverage") on
#   extrapolations.

def linear_model(x, m, k):
    return m * x + k


lin_params, _ = curve_fit(linear_model, users, response_ms)
lin_pred = linear_model(users, *lin_params)
lin_rmse, lin_r2 = fit_quality(response_ms, lin_pred)

response_outlier = response_ms.copy()
response_outlier[-1] = 400.0
out_params, _ = curve_fit(model, users, response_outlier)
out_rmse, out_r2 = fit_quality(response_outlier,
                               model(users, *out_params))

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print(f"{'Case':<30}{'RMSE (ms)':>10}{'R^2':>9}{'Pred @400':>11}")
print(f"{'Quadratic (original)':<30}{rmse:>10.2f}{r2:>9.4f}"
      f"{model(400, *parameters):>11.1f}")
print(f"{'Change 1: straight line':<30}{lin_rmse:>10.2f}{lin_r2:>9.4f}"
      f"{linear_model(400, *lin_params):>11.1f}")
print(f"{'Change 2: 350-user pt = 400 ms':<30}{out_rmse:>10.2f}"
      f"{out_r2:>9.4f}{model(400, *out_params):>11.1f}")
print()
print("Straight-line residuals (positive at both ends, negative in the")
print("middle - a U-shape):", np.round(response_ms - lin_pred, 1))


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Left graph : measurements, quadratic and straight-line fits, and the
#              fit with the changed 350-user point, extended to 400 users.
# Right graph: residuals of the quadratic and straight-line fits.

x_plot = np.linspace(users.min(), 400, 300)
fig, (ax_fit, ax_res) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_fit.scatter(users, response_ms, color="black", zorder=3,
               label="Measurements")
ax_fit.plot(x_plot, model(x_plot, a, b, c), label="Quadratic fit")
ax_fit.plot(x_plot, linear_model(x_plot, *lin_params), "--",
            label="Straight-line fit (change 1)")
ax_fit.plot(x_plot, model(x_plot, *out_params), ":",
            label="Quadratic, 350-user point = 400 ms")
ax_fit.scatter([350], [400], marker="x", color="C2", s=80)
ax_fit.axvspan(350, 400, color="gray", alpha=0.1)
ax_fit.set_xlabel("Concurrent users")
ax_fit.set_ylabel("Response time (ms)")
ax_fit.set_title("Server response-time curve fitting")
ax_fit.legend(fontsize=8)
ax_fit.grid(True)

width = 12
ax_res.bar(users - width / 2, response_ms - predicted, width,
           label="Quadratic")
ax_res.bar(users + width / 2, response_ms - lin_pred, width,
           label="Straight line")
ax_res.axhline(0, color="black", linewidth=0.8)
ax_res.set_xlabel("Concurrent users")
ax_res.set_ylabel("Residual = measured - fitted (ms)")
ax_res.set_title("Residuals")
ax_res.legend()
ax_res.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_14_curve_fitting.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The quadratic model explains {r2:.2%} of the variation; a typical")
print(f"  prediction error is only {rmse:.1f} ms.")
print(f"* The positive x^2 coefficient (a = {a:.5f}) means each extra user")
print("  adds more delay than the previous one: the server is approaching")
print("  saturation (queuing for CPU, database connections etc.).")
print(f"* Predicted response at 400 users is {model(400, a, b, c):.0f} ms. "
      "This is an extrapolation")
print("  beyond the tested range (grey band), so it should be confirmed by")
print("  a load test before relying on it.")
print(f"* A straight line fits worse (RMSE {lin_rmse:.1f} ms) and its "
      "residuals follow a")
print("  U-shaped pattern - proof that the relationship is curved.")
print("* One slow measurement at 350 users changed the 400-user prediction")
print(f"  from {model(400, *parameters):.0f} ms to "
      f"{model(400, *out_params):.0f} ms: always check for outliers.")
print("* Graph (left): the quadratic passes close to every point; the dotted")
print("  curve bends up sharply to chase the changed point. Graph (right):")
print("  quadratic residuals are small and scattered randomly; straight-line")
print("  residuals are large with a systematic pattern.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: forecasting the adoption of a new mobile app (growth curve).
#
# Problem:
#   A county launched a mobile service-payment app. Monthly counts of
#   registered users (thousands) are available for the first 18 months.
#   Planners need to know how many users to expect eventually (to size
#   the servers) and when growth will slow down.
#
# Model:
#   Adoption typically follows a logistic (S-shaped) curve:
#       N(t) = K / (1 + exp(-r (t - t0)))
#   K  : saturation level - the eventual number of users
#   r  : growth rate
#   t0 : month of fastest growth (the inflection point, where N = K/2)
#   For comparison, a pure exponential N(t) = A exp(g t) is also fitted;
#   it can describe the early months but never levels off.
#
# Assumptions:
#   * The monthly data are illustrative: generated from a logistic curve
#     with K = 250, r = 0.45, t0 = 12, plus random noise (+/- 4 %), so the
#     fitted values can be compared with the truth.
#   * The population that can adopt the app is fixed (no new market).
#   * Nonlinear least squares needs a starting guess p0; a sensible one is
#     K ~ 2 x latest value, r ~ 0.3, t0 ~ middle of the data.
#
# Input data: months 1-18 and registered users (thousands).
# Operations: curve_fit for both models, standard errors, RMSE, forecast
# to month 36.

TRUE_K, TRUE_R, TRUE_T0 = 250.0, 0.45, 12.0


def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def exponential(t, A, g):
    return A * np.exp(g * t)


rng = np.random.default_rng(11)
months = np.arange(1, 19, dtype=float)
users_k = logistic(months, TRUE_K, TRUE_R, TRUE_T0) \
    * (1 + rng.normal(0, 0.04, months.size))

p0 = [2 * users_k[-1], 0.3, months.mean()]
log_params, log_cov = curve_fit(logistic, months, users_k, p0=p0)
log_se = np.sqrt(np.diag(log_cov))
exp_params, _ = curve_fit(exponential, months, users_k, p0=[1.0, 0.3])

log_rmse, log_r2 = fit_quality(users_k, logistic(months, *log_params))
exp_rmse, exp_r2 = fit_quality(users_k, exponential(months, *exp_params))

K_fit, r_fit, t0_fit = log_params
month_90 = t0_fit + np.log(9) / r_fit      # N = 0.9 K  ->  solve for t

print()
print("=" * 65)
print("PART 4 - Modified case: logistic growth of app users")
print("=" * 65)
print("Observed users (thousands):", np.round(users_k, 1))
print()
print("Logistic fit:")
for name, val, se, true in zip(("K", "r", "t0"), log_params, log_se,
                               (TRUE_K, TRUE_R, TRUE_T0)):
    print(f"  {name:<3}= {val:8.3f} +/- {se:.3f}   (true value {true})")
print(f"  RMSE {log_rmse:.2f} thousand, R^2 {log_r2:.4f}")
print(f"Exponential fit: A = {exp_params[0]:.3f}, g = {exp_params[1]:.3f}"
      f", RMSE {exp_rmse:.2f} thousand, R^2 {exp_r2:.4f}")
print()
print(f"{'Month':>6}{'Logistic':>12}{'Exponential':>14}   (forecast, thousands)")
for m in (18, 24, 30, 36):
    print(f"{m:>6}{logistic(m, *log_params):>12.1f}"
          f"{exponential(m, *exp_params):>14.1f}")
print(f"90 % of saturation reached around month {month_90:.1f}")

t_plot = np.linspace(0, 36, 400)
fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
ax2.scatter(months, users_k, color="black", zorder=3,
            label="Observed (months 1-18)")
ax2.plot(t_plot, logistic(t_plot, *log_params), label="Logistic fit")
ax2.plot(t_plot, exponential(t_plot, *exp_params), "--",
         label="Exponential fit")
ax2.axhline(K_fit, color="C0", linestyle=":",
            label=f"Saturation K = {K_fit:.0f}k")
ax2.axvline(t0_fit, color="gray", linestyle=":",
            label=f"Fastest growth, month {t0_fit:.1f}")
ax2.axvspan(18, 36, color="gray", alpha=0.08)
ax2.set_ylim(0, 400)
ax2.set_xlabel("Months since launch")
ax2.set_ylabel("Registered users (thousands)")
ax2.set_title("App adoption: logistic vs exponential")
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_14_growth_curve.png", dpi=120)

print()
print("Interpretation:")
print(f"* The logistic model estimates saturation at about {K_fit:.0f} "
      f"thousand users (+/- {log_se[0]:.0f}k),")
print(f"  close to the true {TRUE_K:.0f}k, with fastest growth around month "
      f"{t0_fit:.1f}.")
print(f"  Servers should be sized for roughly {K_fit:.0f}k users, and growth")
print(f"  will have mostly levelled off by month {month_90:.0f}.")
print(f"* The exponential model fits the 18 months less well (RMSE "
      f"{exp_rmse:.1f}k vs {log_rmse:.1f}k)")
print("  and, worse, predicts endless growth: by month 36 it forecasts "
      f"{exponential(36, *exp_params):,.0f}k")
print(f"  users, far more than the market can supply. Its R^2 of "
      f"{exp_r2:.2f} sounds good,")
print("  yet the forecast is useless: a decent fit to past data does not")
print("  guarantee sensible forecasts - the model shape must match the")
print("  real process.")
print("* Graph: the shaded region is the forecast. The logistic curve")
print("  flattens towards K; the exponential curve shoots off the chart.")

plt.show()
