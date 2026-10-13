"""
BIT 2118 Scientific Computing - Task E (Task 06)
Solution of Systems of Linear Equations
=========================================================================

Program file : code/Task_06_linear_systems.py
Run with     : python code/Task_06_linear_systems.py
Output graph : figures/Task_06_linear_systems.png

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Many problems can be written as  A x = b, where
    A : matrix of known coefficients
    x : vector of unknowns
    b : vector of known results (right-hand side)
Solving the system means finding x so that A @ x reproduces b.
    * A unique solution exists when det(A) != 0 (A is non-singular).
    * If det(A) = 0 the equations are dependent: there is either no
      solution or infinitely many.
    * The residual r = b - A x measures how well a computed x satisfies
      the equations; its norm should be close to 0.
    * For two unknowns each equation is a straight line, and the
      solution is the point where the lines intersect.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
Two server types consume different amounts of CPU and memory. Given the
total CPU and memory usage, determine how many servers of each type are
active. Form two simultaneous equations, solve with np.linalg.solve()
and verify with A @ x.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Unknowns:
    x = number of Type A servers,  y = number of Type B servers
Inputs:
    Type A uses 4 CPU units and 8 GB memory
    Type B uses 2 CPU units and 12 GB memory
    Observed totals: 20 CPU units and 72 GB memory
Equations:
    4x +  2y = 20     (CPU)
    8x + 12y = 72     (memory)
Operations:
    np.linalg.solve(A, b)  -> solves A x = b (LU factorisation inside)
    A @ solution           -> verification (should equal b)
    b - A @ solution       -> residual;  np.linalg.norm -> its size

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


def solve_and_check(A, b):
    """Solve A x = b and return (solution, residual, residual norm)."""
    solution = np.linalg.solve(A, b)
    residual = b - A @ solution
    return solution, residual, np.linalg.norm(residual)


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Row 1 = CPU equation, row 2 = memory equation.
# Column 1 = Type A server, column 2 = Type B server.
A = np.array([
    [4.0, 2.0],
    [8.0, 12.0]
])

b = np.array([20.0, 72.0])     # total CPU units, total memory (GB)

solution, residual, residual_norm = solve_and_check(A, b)

print("=" * 65)
print("PART 1 - Original solution: active servers from resource totals")
print("=" * 65)
print(f"det(A) = {np.linalg.det(A):.2f}  (non-zero -> unique solution)")
print(f"Type A servers: {solution[0]:.0f}")
print(f"Type B servers: {solution[1]:.0f}")
print("Verification A @ solution:", A @ solution)
print("Residual:", residual)
print("Residual norm:", residual_norm)


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: the observed memory total is changed from 72 GB to 80 GB.
#   Expected: the memory line moves up, so the intersection moves. Solving
#   by hand: y = 10 - 2x from the CPU equation, then
#   8x + 12(10 - 2x) = 80  ->  x = 2.5, y = 5.
#   Half a server is impossible, which tells the operator that the
#   measurements are inconsistent (e.g. another process is using memory).
#
# Change 2: Type B is changed to use 2 CPU and 4 GB (instead of 12 GB).
#   Now the memory row [8, 4] is exactly 2 x the CPU row [4, 2]. Both
#   server types use memory in the same 2 GB-per-CPU ratio, so the two
#   equations carry the same information, det(A) = 0 and no unique
#   answer exists; np.linalg.solve raises an error.

b_changed = np.array([20.0, 80.0])
solution_changed, _, norm_changed = solve_and_check(A, b_changed)

A_singular = np.array([
    [4.0, 2.0],
    [8.0, 4.0]
])

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print("Change 1 - memory total 72 GB -> 80 GB")
print(f"  Type A servers: {solution_changed[0]:.2f}")
print(f"  Type B servers: {solution_changed[1]:.2f}")
print(f"  Residual norm : {norm_changed:.2e}")
print()
print("Change 2 - Type B uses 4 GB, so memory row = 2 x CPU row")
print(f"  det(A) = {np.linalg.det(A_singular):.2f}")
try:
    np.linalg.solve(A_singular, b)
except np.linalg.LinAlgError as error:
    print(f"  np.linalg.solve failed: {error}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Graph: each equation rearranged for y and drawn as a line.
#   CPU    : y = (20 - 4x) / 2
#   Memory : y = (72 - 8x) / 12    (and the changed 80 GB version)
# The solution is where the lines cross.

x_line = np.linspace(0, 6, 100)
cpu_line = (b[0] - A[0, 0] * x_line) / A[0, 1]
mem_line = (b[1] - A[1, 0] * x_line) / A[1, 1]
mem_line_changed = (b_changed[1] - A[1, 0] * x_line) / A[1, 1]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x_line, cpu_line, label="CPU: 4x + 2y = 20")
ax.plot(x_line, mem_line, label="Memory: 8x + 12y = 72")
ax.plot(x_line, mem_line_changed, linestyle="--",
        label="Memory changed: 8x + 12y = 80")
ax.plot(*solution, "ko", label=f"Solution ({solution[0]:.0f}, "
        f"{solution[1]:.0f})")
ax.plot(*solution_changed, "rs", label=f"Changed ({solution_changed[0]:.1f}"
        f", {solution_changed[1]:.1f})")
ax.set_xlim(0, 6)
ax.set_ylim(0, 10)
ax.set_xlabel("Type A servers (x)")
ax.set_ylabel("Type B servers (y)")
ax.set_title("Server counts as the intersection of two equations")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_06_linear_systems.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* {solution[0]:.0f} Type A and {solution[1]:.0f} Type B servers are "
      "running. Check: 4(3) + 2(4) = 20 CPU")
print("  and 8(3) + 12(4) = 72 GB, matching the observed totals exactly.")
print("* A residual norm of 0 (or ~1e-15) means the answer satisfies both")
print("  equations up to round-off error.")
print("* With 80 GB the answer is 2.5 and 5 servers. The maths is correct,")
print("  but a fractional server signals bad or incomplete measurements.")
print("* When both server types use memory in the same ratio to CPU, the two")
print("  equations are really one, det(A) = 0 and the counts cannot be found.")
print("* Graph: each line is every (x, y) that matches one resource total.")
print("  Only the crossing point matches both. Raising the memory total")
print("  shifts the memory line up and moves the crossing to (2.5, 5).")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: mesh-current analysis of an electrical circuit.
#
# Problem:
#   A DC circuit has three loops (meshes) side by side. Find the current
#   circulating in each mesh, the current in each resistor and the total
#   power the resistors dissipate.
#
# Circuit and assumptions:
#   * Mesh 1 (left)  : 12 V source V1, resistor R1 = 4 ohm, and R2 = 6 ohm
#                      which it shares with mesh 2.
#   * Mesh 2 (middle): R2 (shared with mesh 1), R3 = 2 ohm, and
#                      R4 = 8 ohm shared with mesh 3. No source.
#   * Mesh 3 (right) : R4 (shared), R5 = 3 ohm and a 6 V source V2 that
#                      opposes the clockwise direction.
#   * Mesh currents I1, I2, I3 are all taken clockwise. Ideal sources and
#     wires; resistances are constant.
#
# Kirchhoff's Voltage Law (sum of voltage drops = sum of sources) gives:
#   Mesh 1: (R1 + R2) I1 - R2 I2                 =  V1
#   Mesh 2: -R2 I1 + (R2 + R3 + R4) I2 - R4 I3   =  0
#   Mesh 3: -R4 I2 + (R4 + R5) I3                = -V2
#
# Input data: R1..R5 = 4, 6, 2, 8, 3 ohm; V1 = 12 V; V2 = 6 V
# Mathematical operations: build A (resistance matrix) and b (sources),
# solve A I = b, verify the residual, then
#   shared-resistor current = difference of the two mesh currents
#   power P = I^2 R for each resistor.

R1, R2, R3, R4, R5 = 4.0, 6.0, 2.0, 8.0, 3.0   # ohms
V1, V2 = 12.0, 6.0                              # volts

A_circuit = np.array([
    [R1 + R2, -R2,           0.0],
    [-R2,      R2 + R3 + R4, -R4],
    [0.0,     -R4,           R4 + R5]
])
b_circuit = np.array([V1, 0.0, -V2])

currents, _, circuit_residual = solve_and_check(A_circuit, b_circuit)
I1, I2, I3 = currents

# Current through each resistor (shared resistors carry the difference)
resistor_currents = {
    "R1 (mesh 1)": (R1, I1),
    "R2 (mesh 1-2)": (R2, I1 - I2),
    "R3 (mesh 2)": (R3, I2),
    "R4 (mesh 2-3)": (R4, I2 - I3),
    "R5 (mesh 3)": (R5, I3),
}

print()
print("=" * 65)
print("PART 4 - Modified case: mesh currents in a 3-loop circuit")
print("=" * 65)
print("Resistance matrix A (ohm):\n", A_circuit)
print("Source vector b (V):", b_circuit)
print(f"Mesh currents: I1 = {I1:.4f} A, I2 = {I2:.4f} A, I3 = {I3:.4f} A")
print(f"Residual norm: {circuit_residual:.2e}")
print()
print(f"{'Resistor':<16}{'R (ohm)':>9}{'I (A)':>10}{'P (W)':>10}")
total_power = 0.0
for name, (r, i) in resistor_currents.items():
    power = i ** 2 * r
    total_power += power
    print(f"{name:<16}{r:>9.1f}{i:>10.4f}{power:>10.4f}")
print(f"Total power dissipated: {total_power:.4f} W")

# Energy check: power delivered by the sources must equal power used.
# V1 drives I1 clockwise; V2 opposes clockwise, so it delivers -V2*I3.
source_power = V1 * I1 - V2 * I3
print(f"Power delivered by sources: {source_power:.4f} W")

print()
print("Interpretation:")
print(f"* A current of {I1:.3f} A flows clockwise in mesh 1, driven by V1.")
print(f"* I3 = {I3:.3f} A is negative, so mesh 3 current actually flows")
print("  anticlockwise: V2 pushes it that way more strongly than the")
print("  current coming through R4 from the middle mesh.")
print(f"* R4 carries I2 - I3 = {I2 - I3:.3f} A, more than I2 or |I3| alone,")
print("  because the two mesh currents flow the same way through it and")
print("  add. R1 carries the largest current and dissipates the most power.")
print("* Power delivered by the sources equals the power dissipated in the")
print("  resistors, confirming energy is conserved and the solution is right.")

plt.show()
