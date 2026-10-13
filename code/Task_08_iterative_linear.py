"""
BIT 2118 Scientific Computing - Task G (Task 08)
Iterative Methods for Linear Systems
=========================================================================

Program file : code/Task_08_iterative_linear.py
Run with     : python code/Task_08_iterative_linear.py
Output graphs: figures/Task_08_iterative_linear.png  (convergence)
               figures/Task_08_heat_plate.png        (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Iterative methods solve A x = b by starting from a guess x^(0) and
improving it repeatedly, instead of factorising A.

Jacobi: every unknown is updated using only values from the PREVIOUS
iteration:
    x_i^(k+1) = ( b_i - sum_{j != i} a_ij x_j^(k) ) / a_ii
In matrix form with D = diagonal of A and R = A - D:
    x^(k+1) = D^-1 (b - R x^(k))

Gauss-Seidel: same formula, but each newly computed x_j is used
immediately within the same sweep, so information spreads faster and
it usually needs about half as many iterations as Jacobi.

Stopping rule: stop when the largest change between successive
estimates, ||x^(k+1) - x^(k)||_inf, is below a tolerance.
Convergence is guaranteed when A is (strictly or irreducibly) diagonally
dominant, as in heat-conduction models.

Why bother? For large SPARSE systems (most a_ij = 0) each iteration
costs only about (number of non-zeros) operations, and the matrix never
has to be stored or factorised.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A grid model of temperatures produces many linear equations, most of
whose coefficients are zero. Use Jacobi or Gauss-Seidel, track the error
each iteration and stop once it falls below the tolerance.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    A   : 5x5 tridiagonal matrix (2 on the diagonal, -1 beside it). Row i
          says  2 T_i - T_{i-1} - T_{i+1} = 0, i.e. each interior point's
          temperature is the average of its two neighbours.
    b   : [100, 0, 0, 0, 20] - the boundary temperatures (100 C at the
          left end, 20 C at the right end) move to the right-hand side.
    tol = 1e-8, max_iter = 500, initial guess x = 0
Operations:
    Jacobi and Gauss-Seidel iterations, infinity-norm of the change,
    np.linalg.solve for an exact reference.

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

np.set_printoptions(precision=4, suppress=True)


def jacobi(A, b, tol=1e-8, max_iter=500):
    """Solve A x = b with the Jacobi method.

    Returns the solution and the list of max-changes (one per iteration).
    """
    D = np.diag(A)                 # diagonal entries a_ii
    R = A - np.diagflat(D)         # off-diagonal part
    x = np.zeros_like(b)           # initial guess
    errors = []

    for _ in range(max_iter):
        x_new = (b - R @ x) / D    # all unknowns from the OLD x
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x_new, errors
        x = x_new

    return x, errors


def gauss_seidel(A, b, tol=1e-8, max_iter=500):
    """Solve A x = b with the Gauss-Seidel method.

    x[:i] already holds this sweep's new values (used immediately);
    old[i+1:] holds the previous sweep's values.
    """
    x = np.zeros_like(b)
    errors = []

    for _ in range(max_iter):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (
                b[i]
                - A[i, :i] @ x[:i]          # newest values
                - A[i, i+1:] @ old[i+1:]    # previous-sweep values
            ) / A[i, i]

        error = np.linalg.norm(x - old, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x, errors

    return x, errors


def rod_system(n, left=100.0, right=20.0):
    """Build the n-point steady-state rod model A T = b."""
    A = (2 * np.eye(n)
         - np.eye(n, k=1)
         - np.eye(n, k=-1))
    b = np.zeros(n)
    b[0], b[-1] = left, right
    return A, b


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Simplified steady-state temperature model for five interior grid points.
# Diagonal dominance makes Jacobi and Gauss-Seidel converge.
A = np.array([
    [ 2., -1.,  0.,  0.,  0.],
    [-1.,  2., -1.,  0.,  0.],
    [ 0., -1.,  2., -1.,  0.],
    [ 0.,  0., -1.,  2., -1.],
    [ 0.,  0.,  0., -1.,  2.]
])

# Boundary influence: left boundary = 100 C, right boundary = 20 C
b = np.array([100., 0., 0., 0., 20.])

x_j, err_j = jacobi(A, b)
x_gs, err_gs = gauss_seidel(A, b)
x_exact = np.linalg.solve(A, b)

print("=" * 65)
print("PART 1 - Original solution: 5-point rod, tol = 1e-8")
print("=" * 65)
print("Jacobi temperatures      :", x_j, f"({len(err_j)} iterations)")
print("Gauss-Seidel temperatures:", x_gs, f"({len(err_gs)} iterations)")
print("Direct solution          :", x_exact)
print(f"Max error Jacobi      : {np.max(np.abs(x_j - x_exact)):.2e}")
print(f"Max error Gauss-Seidel: {np.max(np.abs(x_gs - x_exact)):.2e}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: tolerance loosened from 1e-8 to 1e-3.
#   Expected: far fewer iterations, but a less accurate answer. The
#   error shrinks by a roughly constant factor each iteration, so the
#   iteration count grows with log(1/tol).
#
# Change 2: the rod is divided more finely, from 5 to 20 interior points.
#   Expected: many more iterations. Heat information moves only one grid
#   point per Jacobi sweep, and the convergence factor gets closer to 1
#   as n grows (for this matrix it is cos(pi/(n+1)) for Jacobi), so the
#   iteration count rises roughly with n^2.
#   Also note: the stopping test checks the change per iteration, so a
#   slowly converging run can stop while its true error is still larger
#   than the tolerance.

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print(f"{'Case':<26}{'Jacobi it.':>11}{'G-S it.':>9}{'Max error (G-S)':>18}")
for label, n, tol in (("n = 5,  tol = 1e-8", 5, 1e-8),
                      ("n = 5,  tol = 1e-3", 5, 1e-3),
                      ("n = 20, tol = 1e-8", 20, 1e-8)):
    A_n, b_n = rod_system(n)
    xj, ej = jacobi(A_n, b_n, tol=tol, max_iter=5000)
    xg, eg = gauss_seidel(A_n, b_n, tol=tol, max_iter=5000)
    err = np.max(np.abs(xg - np.linalg.solve(A_n, b_n)))
    print(f"{label:<26}{len(ej):>11}{len(eg):>9}{err:>18.2e}")

# Keep the n = 20 histories for the graph
A_20, b_20 = rod_system(20)
_, err_j20 = jacobi(A_20, b_20, max_iter=5000)
_, err_gs20 = gauss_seidel(A_20, b_20, max_iter=5000)

print()
print("Jacobi convergence factor cos(pi/(n+1)):",
      f"n=5 -> {np.cos(np.pi / 6):.4f}, n=20 -> {np.cos(np.pi / 21):.4f}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Graph: maximum change per iteration on a log scale. A straight falling
# line means the error is reduced by the same factor every iteration
# (linear convergence); a steeper line means faster convergence.

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(err_j, label="Jacobi, n = 5")
ax.semilogy(err_gs, label="Gauss-Seidel, n = 5")
ax.semilogy(err_j20, "--", label="Jacobi, n = 20")
ax.semilogy(err_gs20, "--", label="Gauss-Seidel, n = 20")
ax.axhline(1e-8, color="gray", linewidth=0.8, label="Tolerance 1e-8")
ax.set_xlabel("Iteration")
ax.set_ylabel("Maximum change")
ax.set_title("Convergence of iterative temperature solution")
ax.legend()
ax.grid(True, which="both", alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_08_iterative_linear.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print("* The five interior temperatures fall in equal steps from 100 C to")
print(f"  20 C: {x_exact.round(1)}. A straight-line profile is the correct")
print("  steady state for a uniform rod with fixed end temperatures.")
print(f"* Both methods reach the direct answer; Gauss-Seidel needed "
      f"{len(err_gs)} iterations,")
print(f"  about half of Jacobi's {len(err_j)}, because it uses new values "
      "immediately.")
print("* A looser tolerance stops earlier with a less accurate answer;")
print("  a finer grid needs many more iterations.")
print("* Graph: after a quick initial drop, every curve is a straight falling")
print("  line on the log scale (constant reduction per iteration). G-S lines are")
print("  twice as steep as Jacobi's. The n = 20 lines are much flatter, so")
print("  they need roughly 840-1700 iterations to cross the tolerance line.")
print("* Caution: the stopping test measures the CHANGE per iteration, not")
print("  the true error. For n = 20 Gauss-Seidel stopped with changes below")
print("  1e-8 but a true error of about 4e-7, because when convergence is")
print("  slow each step is small even though x is still some way off.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: steady-state temperature of a 2-D metal plate (heat sink).
#
# Problem:
#   A square aluminium plate has a heating element along its top edge
#   at 100 C, while the other three edges are clamped to a 20 C cooling
#   frame. Find the steady-state temperature at every interior point.
#
# Model (2-D Laplace equation, finite differences):
#   On a uniform grid each interior temperature equals the average of its
#   four neighbours:
#       4 T[i,j] - T[i-1,j] - T[i+1,j] - T[i,j-1] - T[i,j+1] = 0
#   With a 30 x 30 interior grid this is 900 equations in 900 unknowns,
#   but each row of A has at most 5 non-zeros out of 900 (over 99 % of
#   A is zero) - exactly the sparse case where iteration is useful.
#
# Assumptions:
#   * No internal heat sources, constant thermal conductivity.
#   * Edge temperatures are held fixed (Dirichlet boundary conditions).
#   * Corners belong to the cold frame.
#
# Methods:
#   * Gauss-Seidel applied directly on the grid (the matrix is never
#     built - only neighbour averages are needed).
#   * Successive Over-Relaxation (SOR): Gauss-Seidel whose update is
#     pushed further by a factor omega (1 < omega < 2):
#         T_new = T_old + omega * (GS value - T_old)
#     With a well chosen omega this converges much faster.
#   * Reference: sparse direct solve (scipy.sparse.linalg.spsolve).

N = 30                 # interior points per side
T_HOT, T_COLD = 100.0, 20.0
TOL = 1e-6


def plate_with_boundaries():
    """(N+2) x (N+2) grid with boundary values set and interior at 0."""
    T = np.zeros((N + 2, N + 2))
    T[0, :] = T_HOT                 # top edge: heater
    T[-1, :] = T_COLD               # bottom edge
    T[:, 0] = T_COLD                # left edge
    T[:, -1] = T_COLD               # right edge
    T[0, 0] = T[0, -1] = T_COLD     # corners belong to the cold frame
    return T


def solve_plate(omega=1.0, tol=TOL, max_iter=10_000):
    """Gauss-Seidel (omega = 1) or SOR (omega > 1) sweep over the plate."""
    T = plate_with_boundaries()
    history = []
    for _ in range(max_iter):
        max_change = 0.0
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                gs_value = 0.25 * (T[i - 1, j] + T[i + 1, j]
                                   + T[i, j - 1] + T[i, j + 1])
                change = omega * (gs_value - T[i, j])
                T[i, j] += change
                max_change = max(max_change, abs(change))
        history.append(max_change)
        if max_change < tol:
            break
    return T, history


# Optimal SOR factor for this square-grid problem (known formula)
omega_opt = 2 / (1 + np.sin(np.pi / (N + 1)))

T_gs, hist_gs = solve_plate(omega=1.0)
T_sor, hist_sor = solve_plate(omega=omega_opt)

# Sparse direct reference: build the 900 x 900 matrix with Kronecker
# products, and move the known boundary values into b.
one_d = diags([-1, 2, -1], [-1, 0, 1], shape=(N, N))
A_plate = (kron(identity(N), one_d) + kron(one_d, identity(N))).tocsr()
boundary = plate_with_boundaries()
b_plate = np.zeros((N, N))
b_plate[0, :] += boundary[0, 1:-1]       # heat from the top edge
b_plate[-1, :] += boundary[-1, 1:-1]
b_plate[:, 0] += boundary[1:-1, 0]
b_plate[:, -1] += boundary[1:-1, -1]
T_direct = spsolve(A_plate, b_plate.ravel()).reshape(N, N)

nonzero_fraction = A_plate.nnz / (N * N) ** 2
centre = N // 2

print()
print("=" * 65)
print("PART 4 - Modified case: 2-D heated plate (900 unknowns)")
print("=" * 65)
print(f"Unknowns: {N * N}, non-zeros in A: {A_plate.nnz} "
      f"({nonzero_fraction:.2%} of the matrix)")
print(f"Gauss-Seidel iterations     : {len(hist_gs)}")
print(f"SOR iterations (omega={omega_opt:.3f}): {len(hist_sor)}")
print(f"Max difference GS  vs direct: "
      f"{np.max(np.abs(T_gs[1:-1, 1:-1] - T_direct)):.2e} C")
print(f"Max difference SOR vs direct: "
      f"{np.max(np.abs(T_sor[1:-1, 1:-1] - T_direct)):.2e} C")
print(f"Temperature at plate centre : {T_direct[centre, centre]:.2f} C")
print(f"Just below the heater (row 1, middle): "
      f"{T_direct[0, centre]:.2f} C")

# Graph for Part 4: temperature map and convergence of GS vs SOR
fig2, (ax_map, ax_conv) = plt.subplots(1, 2, figsize=(11, 4.5))
image = ax_map.imshow(T_sor, cmap="inferno", origin="upper",
                      extent=[0, 1, 0, 1])
fig2.colorbar(image, ax=ax_map, label="Temperature (C)")
contours = ax_map.contour(np.linspace(0, 1, N + 2), np.linspace(1, 0, N + 2),
                          T_sor, levels=[25, 30, 40, 50, 60, 80],
                          colors="white", linewidths=0.7)
ax_map.clabel(contours, fontsize=7)
ax_map.set_title("Steady-state plate temperature")
ax_map.set_xlabel("x (plate width)")
ax_map.set_ylabel("y (plate height)")

ax_conv.semilogy(hist_gs, label="Gauss-Seidel")
ax_conv.semilogy(hist_sor, label=f"SOR, omega = {omega_opt:.3f}")
ax_conv.axhline(TOL, color="gray", linewidth=0.8, label="Tolerance")
ax_conv.set_xlabel("Iteration")
ax_conv.set_ylabel("Maximum change (C)")
ax_conv.set_title("Convergence on the 30 x 30 plate")
ax_conv.legend()
ax_conv.grid(True, which="both", alpha=0.4)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_08_heat_plate.png", dpi=120)

print()
print("Interpretation:")
print("* Heat spreads down from the hot top edge: temperatures are highest")
print("  just under the heater and fall towards 20 C at the other edges.")
print(f"  The centre settles at {T_direct[centre, centre]:.1f} C, i.e. about "
      "a quarter of the way")
print("  from 20 C to 100 C, which matches the symmetry of the problem")
print("  (one hot edge out of four).")
print("* Both methods agree with the direct sparse solver to a small")
print("  fraction of a degree, without ever storing the 900 x 900 matrix.")
print("  Gauss-Seidel's true error (~1e-4 C) is larger than its 1e-6 stopping")
print("  change, because it converges slowly; SOR, converging fast, ends")
print("  much closer (~3e-6 C). A slow method needs a tighter tolerance.")
print(f"* SOR needed {len(hist_sor)} iterations instead of {len(hist_gs)}: "
      f"about {len(hist_gs) / len(hist_sor):.0f}x fewer.")
print("  For large grids this choice of method matters far more than")
print("  computer speed.")
print("* Graph (left): colour shows temperature, white lines join points of")
print("  equal temperature. Graph (right): SOR's error line falls far more")
print("  steeply than Gauss-Seidel's.")

plt.show()
