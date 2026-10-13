"""
BIT 2118 Scientific Computing - Task C (Task 04)
Floating-Point Arithmetic and Numerical Errors
=========================================================================

Program file : code/Task_04_numerical_errors.py
Run with     : python code/Task_04_numerical_errors.py
Output graph : figures/Task_04_numerical_errors.png

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Computers store most real numbers with a finite number of binary digits
(64-bit IEEE-754 "double" gives about 15-16 significant decimal digits),
so many values are only approximations. For example 0.1 has no exact
binary representation, just as 1/3 has no exact decimal representation.

    Absolute error = |true value - approximate value|
    Relative error = absolute error / |true value|

    Truncation error : from replacing an exact mathematical process with
                       an approximation (e.g. stopping a series early).
    Round-off error  : from the finite precision of the computer.

Catastrophic cancellation: subtracting two nearly equal numbers removes
their matching leading digits, leaving mostly round-off "noise". The
expression (sqrt(1+x) - 1)/x suffers from this when x is tiny. It can be
rewritten exactly (multiply top and bottom by sqrt(1+x) + 1) as
    1 / (sqrt(1+x) + 1),
which has no subtraction and is therefore numerically stable.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A program compares 0.1 + 0.2 with 0.3 and unexpectedly reports False.
Solution approach: floating-point values are approximate, so compare
using np.isclose() (a tolerance) instead of exact equality.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    0.1, 0.2, 0.3          values for the equality test
    true_value = sqrt(2)   reference value
    approx_value = 1.414   rounded approximation of sqrt(2)
    x = 1e-4, 1e-8, 1e-12  small inputs for the cancellation example
Operations:
    addition and == / np.isclose comparison
    absolute and relative error formulas
    direct formula  (sqrt(1+x) - 1)/x   vs   stable 1/(sqrt(1+x) + 1)

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from decimal import Decimal
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

print("=" * 65)
print("PART 1 - Original solution")
print("=" * 65)

# --- Floating-point comparison -------------------------------------------
calculated = 0.1 + 0.2
expected = 0.3

print("0.1 + 0.2 =", calculated)
# Show the exact binary values that are actually stored (20 decimals)
print(f"Stored 0.1 + 0.2 : {calculated:.20f}")
print(f"Stored 0.3       : {expected:.20f}")
print("Exact comparison:", calculated == expected)
# np.isclose uses |a - b| <= atol + rtol*|b| (defaults rtol=1e-5, atol=1e-8)
print("Tolerance-based comparison:", np.isclose(calculated, expected))


# --- Absolute and relative error -----------------------------------------
def errors(true_value, approx_value):
    """Return (absolute error, relative error) of an approximation."""
    absolute = abs(true_value - approx_value)
    return absolute, absolute / abs(true_value)


true_value = np.sqrt(2)
approx_value = 1.414
absolute_error, relative_error = errors(true_value, approx_value)

print()
print(f"sqrt(2) = {true_value:.10f}, approximation = {approx_value}")
print(f"Absolute error: {absolute_error:.8f}")
print(f"Relative error: {relative_error:.8%}")


# --- Numerically unstable vs stable formulation --------------------------
def direct_formula(x):
    """(sqrt(1+x) - 1)/x : subtracts two nearly equal numbers (unstable)."""
    return (np.sqrt(1 + x) - 1) / x


def stable_formula(x):
    """1/(sqrt(1+x) + 1) : algebraically identical, no subtraction."""
    return 1 / (np.sqrt(1 + x) + 1)


print()
print("Direct vs stable formula (exact limit as x -> 0 is 0.5):")
for x in [1e-4, 1e-8, 1e-12]:
    print(f"x={x:.0e}: direct={direct_formula(x):.12f}, "
          f"stable={stable_formula(x):.12f}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the sqrt(2) approximation is improved from 1.414 to 1.41421
#           (two more decimal places).
#   Expected: absolute and relative errors fall by about 100x, because
#   each extra correct decimal digit divides the error by about 10.
#
# Change 2: x is made even smaller: 1e-14, 1e-15 and 1e-16.
#   Expected: the direct formula becomes badly wrong and finally returns
#   0, because 1 + 1e-16 rounds to exactly 1.0 in double precision, so
#   sqrt(1+x) - 1 = 0. The stable formula stays at 0.5.

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
for approx in (1.414, 1.41421):
    a_err, r_err = errors(true_value, approx)
    print(f"approx={approx:<8} absolute error={a_err:.2e}  "
          f"relative error={r_err:.2e}")

print()
for x in [1e-14, 1e-15, 1e-16]:
    print(f"x={x:.0e}: direct={direct_formula(x):.12f}, "
          f"stable={stable_formula(x):.12f}")
print(f"Is 1 + 1e-16 == 1 in double precision? {1 + 1e-16 == 1}")
print(f"Machine epsilon (gap after 1.0): {np.finfo(float).eps:.3e}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Graph: relative error of each formula against x on log-log axes. The
# stable formula is used as the reference ("true") value because it is
# accurate to full precision for all x in this range.

x_values = np.logspace(-16, -1, 300)
reference = stable_formula(x_values)
direct_rel_error = np.abs(direct_formula(x_values) - reference) / reference
# Values that are exactly 0 cannot be drawn on a log axis, so floor them
# at 1e-17 (below double-precision resolution) just for plotting.
direct_rel_error = np.maximum(direct_rel_error, 1e-17)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(x_values, direct_rel_error, label="Direct (sqrt(1+x)-1)/x")
ax.axhline(np.finfo(float).eps, color="gray", linestyle="--",
           label="Machine epsilon")
ax.set_xlabel("x")
ax.set_ylabel("Relative error vs stable formula")
ax.set_title("Round-off error from catastrophic cancellation")
ax.legend()
ax.grid(True, which="both", alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_04_numerical_errors.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print("* 0.1 + 0.2 is stored as 0.30000000000000004, not 0.3, so '=='")
print("  fails. The difference (~5.6e-17) is pure round-off error; using a")
print("  tolerance (np.isclose) gives the correct answer True.")
print(f"* 1.414 is off from sqrt(2) by {absolute_error:.2e} "
      f"({relative_error:.4%}). Using 1.41421 cuts")
print("  the error about 100x, so precision of inputs limits accuracy.")
print("* The direct formula loses accuracy as x shrinks and returns 0 at")
print("  x = 1e-16, while the stable formula always gives about 0.5.")
print("  Same mathematics, different arithmetic -> very different results.")
print("* Graph: the direct formula's error rises roughly as eps/x as x gets")
print("  smaller (a straight line sloping up to the left), reaching 100 %")
print("  error near x = 1e-16. The stable formula has no such growth.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: accumulated round-off in a mobile-money ledger (banking).
#
# Problem:
#   A mobile-money platform charges a fee of KES 0.10 on each of
#   1,000,000 micro-transactions in a day. The daily fee total must be
#   exactly KES 100,000.00. Adding 0.10 one million times with floats
#   introduces round-off at every addition, and these errors accumulate.
#
# Assumptions:
#   * Every transaction carries exactly the same fee of KES 0.10.
#   * Money must be correct to the cent; any non-zero error is a
#     reconciliation problem.
#
# Input data:
#   fee = 0.10 KES, n_transactions = 1,000,000
#
# Methods compared:
#   1. Naive float loop  : total += 0.10           (round-off each step)
#   2. math.fsum         : compensated summation that tracks lost digits
#   3. Integer cents     : 10 cents * n, then / 100 (exact integers)
#   4. decimal.Decimal   : base-10 arithmetic, 0.10 stored exactly
#   5. float32 loop      : single precision (~7 digits), e.g. an
#                          embedded device, to show the precision effect

fee = 0.10
n_transactions = 1_000_000
exact_total = 100_000.00

total_float = 0.0
total_float32 = np.float32(0.0)
fee32 = np.float32(fee)
for _ in range(n_transactions):
    total_float += fee            # 64-bit double precision
    total_float32 += fee32        # 32-bit single precision

total_fsum = math.fsum([fee] * n_transactions)
total_cents = (10 * n_transactions) / 100
total_decimal = Decimal("0.10") * n_transactions

print()
print("=" * 65)
print("PART 4 - Modified case: summing 1,000,000 fees of KES 0.10")
print("=" * 65)
print(f"{'Method':<22}{'Total (KES)':>24}{'Abs. error':>16}")
for name, value in (("Naive float64 loop", total_float),
                    ("Naive float32 loop", float(total_float32)),
                    ("math.fsum", total_fsum),
                    ("Integer cents", total_cents),
                    ("decimal.Decimal", float(total_decimal))):
    print(f"{name:<22}{value:>24.10f}{abs(value - exact_total):>16.3e}")

print()
print("Interpretation:")
print(f"* The float64 loop is off by about {abs(total_float - exact_total):.1e}"
      " KES. That is tiny, but it")
print("  means the ledger does not exactly equal 100,000.00 and an equality")
print("  check in a reconciliation report would fail.")
print(f"* In float32 the error grows to about "
      f"{abs(float(total_float32) - exact_total):,.0f} KES: once the total is")
print("  large, float32 can only store it to the nearest 0.0078 or so, so")
print("  every addition of 0.10 is rounded by a large fraction of itself,")
print("  and these rounding errors pile up. Precision matters.")
print("* math.fsum, integer cents and Decimal give exactly 100,000.00.")
print("  Banking systems therefore store money as integer cents or")
print("  decimals, never as binary floating point.")

plt.show()
