"""
BIT 2118 Scientific Computing - Task A (Task 02)
NumPy Arrays and Numerical Computation
=========================================================================

Program file : code/Task_02_numpy_arrays.py
Run with     : python code/Task_02_numpy_arrays.py

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
An array is an organised collection of numbers. A vector is a
one-dimensional array v = [v1, v2, ..., vn]; a matrix is a rectangular
array A = [aij]. NumPy applies an operation to every element of an array
at once (element-wise / vectorised), e.g. if x = [1, 2, 3] then
x**2 = [1, 4, 9], with no explicit Python loop.

Statistics used in this task:
    mean  = (x1 + x2 + ... + xn) / n
    min   = smallest element,  max = largest element
Linear unit conversion applied element-wise:
    F = 1.8 * C + 32          (Celsius -> Fahrenheit)

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A weather station records 24 hourly temperatures. Store them in an array,
calculate the daily mean, minimum and maximum, and convert every value
from Celsius to Fahrenheit.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    temperatures_c : 24 hourly readings in degrees Celsius (00:00-23:00)
Operations:
    np.mean()  -> arithmetic mean (sum of readings / 24)
    np.min()   -> smallest reading
    np.max()   -> largest reading
    1.8*C + 32 -> vectorised linear transformation of the whole array

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning / interpretation of the output   (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

import numpy as np


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# 24 hourly temperature readings in degrees Celsius.
# Index 0 is midnight (00:00) and index 23 is 23:00.
temperatures_c = np.array([
    18.5, 18.0, 17.8, 17.5, 17.9, 19.0,   # 00:00 - 05:00 (night, coolest)
    21.5, 23.8, 25.6, 27.1, 28.3, 29.0,   # 06:00 - 11:00 (morning warm-up)
    29.4, 29.1, 28.5, 27.6, 26.2, 24.8,   # 12:00 - 17:00 (afternoon peak)
    23.4, 22.1, 21.0, 20.2, 19.6, 19.0    # 18:00 - 23:00 (evening cooling)
])


def summarise(readings_c):
    """Return the mean, minimum and maximum of an array of readings.

    Each NumPy function reduces the whole array to a single number in one
    call, so no explicit loop over the 24 values is needed.
    """
    return np.mean(readings_c), np.min(readings_c), np.max(readings_c)


def celsius_to_fahrenheit(readings_c):
    """Convert an entire array from Celsius to Fahrenheit.

    The formula F = 1.8*C + 32 is applied element-wise: NumPy multiplies
    every element by 1.8 and adds 32 to every element in one expression.
    """
    return 1.8 * readings_c + 32


mean_temp, min_temp, max_temp = summarise(temperatures_c)

# Vectorised Celsius-to-Fahrenheit conversion of all 24 readings at once
temperatures_f = celsius_to_fahrenheit(temperatures_c)

print("=" * 65)
print("PART 1 - Original weather-station solution")
print("=" * 65)
print(f"Number of readings          : {temperatures_c.size}")
print(f"Daily mean temperature      : {mean_temp:.2f} C")
print(f"Minimum temperature         : {min_temp:.2f} C "
      f"(at {np.argmin(temperatures_c):02d}:00)")
print(f"Maximum temperature         : {max_temp:.2f} C "
      f"(at {np.argmax(temperatures_c):02d}:00)")
print("First six Fahrenheit readings:", temperatures_f[:6])


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change made: the 14:00 reading (index 14) is replaced with 35.0 C to
# simulate a sudden heat spike (or a faulty sensor value).
#
# Expected effect:
#   * The mean rises by (35.0 - 28.5) / 24 = 0.27 C, because one value
#     out of 24 increased by 6.5 C.
#   * The maximum changes from 29.4 C to 35.0 C (the new highest value).
#   * The minimum is unchanged, because the coolest hour was not modified.
# This shows that the mean is sensitive to a single extreme value
# (an outlier), while the min/max only change if the extreme is modified.

modified_c = temperatures_c.copy()     # copy so the original stays intact
modified_c[14] = 35.0                  # the single changed input value

new_mean, new_min, new_max = summarise(modified_c)

print()
print("=" * 65)
print("PART 2 - Effect of changing the 14:00 reading from 28.5 C to 35.0 C")
print("=" * 65)
print(f"{'Statistic':<10}{'Original':>12}{'Modified':>12}{'Change':>12}")
for name, old, new in (("Mean", mean_temp, new_mean),
                       ("Minimum", min_temp, new_min),
                       ("Maximum", max_temp, new_max)):
    print(f"{name:<10}{old:>12.2f}{new:>12.2f}{new - old:>+12.2f}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT  (Instruction D)
# =========================================================================
# No graph is required for this task; the interpretation is printed.

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* The average temperature over the day was {mean_temp:.2f} C.")
print(f"* The coolest time was early morning ({min_temp:.1f} C) and the")
print(f"  hottest was around midday ({max_temp:.1f} C), a daily range of")
print(f"  {max_temp - min_temp:.1f} C, which is a normal day/night cycle.")
print(f"* In Fahrenheit the day ranged from {temperatures_f.min():.1f} F to "
      f"{temperatures_f.max():.1f} F.")
print("* One abnormal reading (Part 2) shifted the mean and the maximum,")
print("  so sensor data should be checked for outliers before reporting.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: IoT soil-moisture monitoring on a farm.
#
# Problem:
#   Three soil-moisture sensors (one per farm plot) report volumetric
#   water content (%) every 4 hours for one day. The farmer wants the
#   average, lowest and highest moisture of each plot, the readings in
#   the range 0-1 (fraction) for an irrigation model, and to know which
#   plots need irrigation.
#
# Assumptions:
#   * Readings are taken at 00:00, 04:00, 08:00, 12:00, 16:00 and 20:00.
#   * Sensors are calibrated, so no outlier correction is needed.
#   * A plot needs irrigation when its DAILY MEAN moisture is below 25 %
#     (a typical threshold for maize in loam soil).
#
# Input data:
#   A 2-D array (matrix) with shape (3 plots, 6 readings). Each row is a
#   plot, each column is a reading time. This shows that NumPy handles
#   matrices exactly like vectors, operating along a chosen axis.
#
# Mathematical operations:
#   np.mean / np.min / np.max with axis=1 -> one result per row (plot)
#   moisture / 100                       -> element-wise percent->fraction
#   mean < threshold                     -> element-wise comparison giving
#                                           a Boolean array (True/False)

plots = ["Plot A", "Plot B", "Plot C"]
reading_times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]

moisture_pct = np.array([
    [32.0, 31.5, 30.8, 28.9, 27.5, 27.0],   # Plot A - recently irrigated
    [24.5, 24.1, 23.6, 21.8, 20.9, 20.5],   # Plot B - sandy, drains fast
    [28.0, 27.8, 27.1, 25.2, 24.0, 23.7],   # Plot C - drying out
])

IRRIGATION_THRESHOLD = 25.0   # percent

# axis=1 means "reduce across the columns", i.e. one value per plot (row)
plot_mean = moisture_pct.mean(axis=1)
plot_min = moisture_pct.min(axis=1)
plot_max = moisture_pct.max(axis=1)

# Vectorised unit conversion for every element of the 3 x 6 matrix
moisture_fraction = moisture_pct / 100.0

# Boolean mask: True where the plot's mean is below the threshold
needs_irrigation = plot_mean < IRRIGATION_THRESHOLD

print()
print("=" * 65)
print("PART 4 - Modified case: IoT soil-moisture sensors on a farm")
print("=" * 65)
print("Input matrix shape (plots, readings):", moisture_pct.shape)
print("Reading times:", ", ".join(reading_times))
print()
print(f"{'Plot':<8}{'Mean %':>9}{'Min %':>9}{'Max %':>9}  Irrigate?")
for i, plot in enumerate(plots):
    print(f"{plot:<8}{plot_mean[i]:>9.2f}{plot_min[i]:>9.2f}"
          f"{plot_max[i]:>9.2f}  {'YES' if needs_irrigation[i] else 'no'}")
print()
print("Moisture as fractions (input to irrigation model):")
print(np.round(moisture_fraction, 3))
print()
print("Overall farm mean moisture:", f"{moisture_pct.mean():.2f} %")

# Interpretation of the modified case
print()
print("Interpretation:")
print("* Moisture falls in every plot during the day because of")
print("  evaporation and plant uptake (each row decreases left to right).")
print(f"* Plot B has the lowest average ({plot_mean[1]:.2f} %) and is below")
print(f"  the {IRRIGATION_THRESHOLD:.0f} % threshold, so it should be "
      "irrigated first.")
print(f"* Plot C averages {plot_mean[2]:.2f} %, just above the threshold,")
print("  but its last readings are already below 25 %, so it will need")
print("  water soon. Plot A is well watered.")
print("* One NumPy call per statistic summarised all plots at once,")
print("  which scales to thousands of sensors without changing the code.")
