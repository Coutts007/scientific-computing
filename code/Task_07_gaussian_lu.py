"""
BIT 2118 Scientific Computing - Task F (Task 07)
Gaussian Elimination and LU Decomposition
=========================================================================

Program file : code/Task_07_gaussian_lu.py
Run with     : python code/Task_07_gaussian_lu.py
Output graph : figures/Task_07_gaussian_lu.png

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Gaussian elimination turns A x = b into an equivalent upper-triangular
system U x = c by subtracting multiples of one row from the rows below it
(eliminating one variable per column). The last equation then contains
only one unknown; back-substitution solves upwards for the rest.

    multiplier  m_ik = a_ik / a_kk
    row_i      <- row_i - m_ik * row_k

LU decomposition stores that work: A = L U, or P A = L U with row
pivoting, where
    L : lower triangular, 1s on the diagonal, holds the multipliers m_ik
    U : upper triangular, the result of elimination
    P : permutation matrix recording the row swaps (partial pivoting
        swaps in the largest available pivot for numerical stability)
Solving A x = b then becomes two cheap triangular solves:
    L y = P b   (forward substitution),   U x = y   (back substitution)
Cost: factorising is about (2/3) n^3 operations; each triangular solve
pair is only about 2 n^2. So when A stays the same and only b changes,
factorise once and reuse L and U.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A simulation uses the same coefficient matrix A every second but receives
a new measurement vector b. Factor A once with LU decomposition and reuse
the factors for each new b instead of repeating full elimination.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    A : fixed 3x3 coefficient matrix [[4,-1,0], [-1,4,-1], [0,-1,3]]
    measurement_sets : three right-hand-side vectors b
Operations:
    gaussian_elimination() - written out by hand to show each step
    scipy.linalg.lu        - returns P, L, U explicitly (for display)
    lu_factor(A)           - factorises A once (compact LU + pivots)
    lu_solve((lu, piv), b) - reuses the factors for each b
    b - A @ x              - residual check;  np.linalg.solve for comparison

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

import os
from pathlib import Path
import timeit

# Use one CPU thread for the linear-algebra library so the timing
# comparison in Part 2 is fair and repeatable (multi-threading start-up
# costs otherwise add random noise to very short runs). Must be set
# before NumPy is imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

np.set_printoptions(precision=4, suppress=True)


def gaussian_elimination(A, b, verbose=False):
    """Solve A x = b by Gaussian elimination with partial pivoting.

    Written out step by step to show what lu_factor does internally.
    Works on copies so the caller's A and b are not changed.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # Forward elimination: create zeros below the diagonal, column by column
    for k in range(n - 1):
        # Partial pivoting: bring the row with the largest |a_ik| to row k
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
            if verbose:
                print(f"  swap rows {k} and {pivot_row}")
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]          # multiplier
            A[i, k:] -= m * A[k, k:]       # row_i <- row_i - m * row_k
            b[i] -= m * b[k]
            if verbose:
                print(f"  row{i} <- row{i} - ({m:.4f}) x row{k}")

    if verbose:
        print("  Upper-triangular system [U | c]:")
        print(np.column_stack([A, b]))

    # Back-substitution: solve from the last equation upwards
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# The same coefficient matrix is reused for several measurement vectors.
A = np.array([
    [4.0, -1.0, 0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 3.0]
])

measurement_sets = [
    np.array([15.0, 10.0, 10.0]),
    np.array([16.0, 11.0, 9.0]),
    np.array([14.0, 12.0, 11.0])
]

print("=" * 65)
print("PART 1 - Original solution")
print("=" * 65)

# Step-by-step Gaussian elimination for the first measurement set
print("Gaussian elimination steps for measurement set 1:")
x_manual = gaussian_elimination(A, measurement_sets[0], verbose=True)
print("  Back-substitution solution:", x_manual)

# Show the factors explicitly: A = P L U
P, L, U = lu(A)
print("\nL (multipliers):\n", L)
print("U (eliminated matrix):\n", U)
print("P @ L @ U equals A:", np.allclose(P @ L @ U, A))
print()

# Factor A only once
lu_matrix, piv = lu_factor(A)

for i, b in enumerate(measurement_sets, start=1):
    x = lu_solve((lu_matrix, piv), b)      # reuse factors: cheap solve
    residual = b - A @ x
    print(f"Measurement set {i}")
    print("Solution:", x)
    print("Residual norm:", np.linalg.norm(residual))
    print()

# Direct solution for comparison
print("Direct solution for first set:",
      np.linalg.solve(A, measurement_sets[0]))


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the first measurement b1 = [15, 10, 10] has its first entry
#   raised by 1 to [16, 10, 10].
#   Expected: because the system is linear, the solution changes by
#   A^-1 @ [1, 0, 0], i.e. by the first column of A^-1. The same L and U
#   are reused - no new factorisation is needed for a new b.
#
# Change 2: the problem size is increased from n = 3 to n = 300 and the
#   number of b vectors from 3 to as many as 1000.
#   Expected: re-solving from scratch each time costs ~(2/3)n^3 per b,
#   while LU reuse costs (2/3)n^3 once plus ~2n^2 per b. The time saved
#   grows with the number of b vectors.

b_changed = np.array([16.0, 10.0, 10.0])
x_original = lu_solve((lu_matrix, piv), measurement_sets[0])
x_changed = lu_solve((lu_matrix, piv), b_changed)

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print("Change 1 - b1 [15,10,10] -> [16,10,10]")
print("  Original solution :", x_original)
print("  New solution      :", x_changed)
print("  Change in x       :", x_changed - x_original)
print("  First column A^-1 :", np.linalg.inv(A)[:, 0])

# Change 2: timing on a larger, well-conditioned system
rng = np.random.default_rng(0)
n = 300
A_big = rng.random((n, n)) + n * np.eye(n)   # diagonally dominant
b_counts = [1, 10, 50, 100, 250, 500, 1000]
all_b = rng.random((max(b_counts), n))

time_resolve, time_reuse = [], []
for count in b_counts:
    bs = all_b[:count]

    def resolve_each_time():
        """Full Gaussian elimination (np.linalg.solve) for every b."""
        for vec in bs:
            np.linalg.solve(A_big, vec)

    def factor_once_reuse():
        """One LU factorisation, then cheap triangular solves."""
        factors = lu_factor(A_big)
        for vec in bs:
            lu_solve(factors, vec)

    # Run each once untimed (warm-up), then time several runs. Small
    # cases are repeated more often so each measurement is long enough to
    # be reliable; the best of 3 averages is kept to ignore background
    # activity on the computer.
    resolve_each_time()
    factor_once_reuse()
    runs = max(1, 200 // count)
    time_resolve.append(min(timeit.repeat(resolve_each_time,
                                          number=runs, repeat=3)) / runs)
    time_reuse.append(min(timeit.repeat(factor_once_reuse,
                                        number=runs, repeat=3)) / runs)

print()
print(f"Change 2 - n = {n}, many measurement vectors")
print(f"{'# of b':>8}{'Re-solve (s)':>15}{'LU reuse (s)':>15}{'Ratio':>8}")
for count, tr, tu in zip(b_counts, time_resolve, time_reuse):
    print(f"{count:>8}{tr:>15.4f}{tu:>15.4f}{tr / tu:>7.1f}x")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Graph: total time to process k measurement vectors with each strategy.

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(b_counts, time_resolve, marker="o", label="Re-solve every time")
ax.plot(b_counts, time_reuse, marker="s", label="LU factor once, reuse")
ax.set_xlabel("Number of measurement vectors b")
ax.set_ylabel("Total time (s)")
ax.set_title(f"Repeated solutions of a {n}x{n} system")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_07_gaussian_lu.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print("* Each solution x gives the three unknown state values that explain")
print("  one set of measurements. For set 1 they are about "
      f"{x_original.round(3)}.")
print("* Residual norms of ~1e-15 show every solution satisfies A x = b to")
print("  machine precision; LU, hand elimination and np.linalg.solve agree.")
print("* L stores the elimination multipliers and U the triangular result,")
print("  so P L U rebuilds A exactly.")
print("* Raising b1[0] by 1 changed x by the first column of A^-1 - the")
print("  response of a linear system to a unit change in one input.")
print("* Graph: both lines grow with the number of b vectors, but the LU")
print("  line grows much more slowly because the expensive n^3 step is done")
print(f"  only once. At 1000 vectors LU reuse was "
      f"{time_resolve[-1] / time_reuse[-1]:.1f}x faster in this run.")
print("  With a single b both strategies do one factorisation, so there is")
print("  no saving; the benefit appears as soon as b is repeated.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: Leontief input-output model of an economy.
#
# Problem:
#   A regional economy has three sectors: Agriculture, Manufacturing and
#   Services. To produce output, each sector uses some output of the
#   others. Planners want the total production x each sector must
#   achieve to meet the final (consumer) demand d in each of four
#   quarters of the year.
#
# Model:  x = C x + d   ->   (I - C) x = d
#   C_ij = value of sector i's output used to make 1 unit of sector j.
#   The matrix (I - C) is the same every quarter; only d changes, which
#   is exactly the "same A, new b" situation LU decomposition is for.
#
# Assumptions:
#   * Technical coefficients C are constant over the year.
#   * Values are in millions of KES; demand figures are illustrative.
#   * Each column of C sums to less than 1, so the economy is productive
#     and (I - C) is invertible with a non-negative solution.
#
# Input data:
#   C  (columns: input needed per unit of Agri, Manuf, Serv)
#   quarterly final demand d for Q1-Q4
#
# Mathematical operations: build I - C, factorise once with lu_factor,
# lu_solve for each quarter, check residuals.

sectors = ["Agriculture", "Manufacturing", "Services"]
C = np.array([
    [0.15, 0.25, 0.05],   # Agriculture output used by Agri, Manuf, Serv
    [0.20, 0.10, 0.10],   # Manufacturing output used by each sector
    [0.10, 0.20, 0.15],   # Services output used by each sector
])

quarterly_demand = {
    "Q1": np.array([120.0, 200.0, 310.0]),
    "Q2": np.array([150.0, 210.0, 320.0]),   # harvest season
    "Q3": np.array([110.0, 230.0, 330.0]),
    "Q4": np.array([125.0, 260.0, 380.0]),   # festive season
}

leontief = np.eye(3) - C
factors = lu_factor(leontief)          # factorise (I - C) only once

print()
print("=" * 65)
print("PART 4 - Modified case: Leontief input-output economic model")
print("=" * 65)
print("Technical coefficient matrix C:\n", C)
print("Leontief matrix I - C:\n", leontief)
print()
header = "".join(f"{s:>15}" for s in sectors)
print(f"{'Quarter':<8}{'':<11}{header}{'Residual':>11}")
outputs = {}
for quarter, d in quarterly_demand.items():
    x = lu_solve(factors, d)
    outputs[quarter] = x
    res = np.linalg.norm(d - leontief @ x)
    print(f"{quarter:<8}{'demand':<11}" + "".join(f"{v:>15.1f}" for v in d))
    print(f"{'':<8}{'production':<11}"
          + "".join(f"{v:>15.1f}" for v in x) + f"{res:>11.1e}")

q1_x, q1_d = outputs["Q1"], quarterly_demand["Q1"]
print()
print("Interpretation:")
print("* Every sector must produce more than consumers demand, because part")
print("  of its output is used up by other sectors. In Q1 manufacturing")
print(f"  must produce {q1_x[1]:.1f}M to deliver {q1_d[1]:.0f}M to consumers.")
print("* The extra amount (production - demand) is the inter-industry")
print("  'intermediate' use, e.g. "
      f"{q1_x[0] - q1_d[0]:.1f}M of agricultural output in Q1.")
print("* Q4 needs the highest manufacturing and services production because")
print("  of festive-season demand, while agriculture peaks in Q2 (harvest).")
print("  Note Q3 agriculture output is about equal to Q1 even though its")
print("  own demand is lower: higher manufacturing and services activity")
print("  uses more agricultural inputs. This knock-on effect between")
print("  sectors is exactly what the linear system captures.")
print("* (I - C) was factorised once and reused for all four quarters, just")
print("  like the measurement sets in Part 1; a planner could test hundreds")
print("  of demand scenarios at almost no extra cost.")

plt.show()
