"""
BIT 2118 Scientific Computing - Task D (Task 05)
Matrix Algebra and Matrix Operations
=========================================================================

Program file : code/Task_05_matrix_algebra.py
Run with     : python code/Task_05_matrix_algebra.py
Output graph : figures/Task_05_matrix_algebra.png

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
A matrix organises values in rows and columns.
    Addition       : (A + B)_ij = a_ij + b_ij
    Multiplication : c_ij = sum_k a_ik * b_kj  (row i of A times column
                     j of B). In general AB != BA (order matters).
    Determinant    : det(A). For a 2x2 matrix [[a, b], [c, d]] it is
                     ad - bc. Geometrically |det| is the factor by which
                     areas are scaled; det = 0 means A is not invertible.
    Inverse        : A^-1 satisfies A A^-1 = I (identity) and undoes A.

2-D transformations used here:
    Rotation by angle t : R = [[cos t, -sin t], [sin t, cos t]], det = 1
    Scaling             : S = [[sx, 0], [0, sy]],                det = sx*sy

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A graphics program needs to rotate and scale a set of 2-D coordinates
representing the points of an object. Build scaling and rotation
matrices, combine them, and use matrix multiplication to transform every
point consistently.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    points : 4 corners of the unit square, one [x, y] per row
    angle  : 45 degrees (rotation)
    scale  : sx = 2.0 (x direction), sy = 1.5 (y direction)
Operations:
    transformation = rotation @ scale   (matrix product: scale first,
                                         then rotate)
    points @ transformation.T           (apply to every point at once;
                                         the transpose is needed because
                                         points are stored as rows)
    np.linalg.det, np.linalg.inv        (area factor and undo matrix)

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

# Print small floating-point values such as 1e-17 as 0 for readability
np.set_printoptions(precision=4, suppress=True)


def rotation_matrix(degrees):
    """Return the 2x2 matrix that rotates points anticlockwise."""
    t = np.deg2rad(degrees)
    return np.array([
        [np.cos(t), -np.sin(t)],
        [np.sin(t), np.cos(t)]
    ])


def scaling_matrix(sx, sy):
    """Return the 2x2 matrix that stretches x by sx and y by sy."""
    return np.array([
        [sx, 0.0],
        [0.0, sy]
    ])


def polygon_area(pts):
    """Area of a polygon from its corner points (shoelace formula).

    Used to check that |det| really is the area scaling factor.
    """
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Four corners of a square: each row is [x, y]
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

rotation = rotation_matrix(45)
scale = scaling_matrix(2.0, 1.5)

# Combined transformation: scale first (rightmost), then rotate
transformation = rotation @ scale

# Row-vector form requires multiplication by the transpose:
# for a single column vector p we would write T @ p; for rows we use
# points @ T.T, which transforms every row in one matrix product.
transformed_points = points @ transformation.T

# Inverse matrix: applying it returns the original points
inverse = np.linalg.inv(transformation)
recovered = transformed_points @ inverse.T

print("=" * 65)
print("PART 1 - Original solution: rotate 45 deg and scale (2, 1.5)")
print("=" * 65)
print("Original points:\n", points)
print("Transformation matrix:\n", transformation)
print("Transformed points:\n", transformed_points)
print("Determinant:", np.linalg.det(transformation))
print(f"Area before: {polygon_area(points):.4f}, "
      f"area after: {polygon_area(transformed_points):.4f}")
print("Inverse matrix:\n", inverse)
print("T @ T^-1 (should be identity):\n", transformation @ inverse)
print("Points recovered with inverse match original:",
      np.allclose(recovered, points))


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change 1: rotation angle changed from 45 to 90 degrees.
#   Expected: the square is turned a further 45 degrees, but the
#   determinant stays 3.0, because rotation does not change area
#   (det of any rotation = 1) and det(RS) = det(R) * det(S) = 1 * 3.
#
# Change 2: order of multiplication swapped to scale @ rotation
#   (rotate first, then scale).
#   Expected: a different matrix and a different shape (a rhombus
#   instead of a rotated rectangle), because matrix multiplication is
#   not commutative. The determinant is still 3.0, since
#   det(SR) = det(S) det(R) as well.

transform_90 = rotation_matrix(90) @ scale
points_90 = points @ transform_90.T

swapped = scale @ rotation           # rotate first, then scale
points_swapped = points @ swapped.T

print()
print("=" * 65)
print("PART 2 - Effect of changing the inputs")
print("=" * 65)
print("Change 1 - angle 90 deg, transformation:\n", transform_90)
print("Transformed points:\n", points_90)
print(f"Determinant: {np.linalg.det(transform_90):.4f}")
print()
print("Change 2 - order swapped (scale @ rotation):\n", swapped)
print("Transformed points:\n", points_swapped)
print(f"Determinant: {np.linalg.det(swapped):.4f}")
print("Does rotation @ scale equal scale @ rotation?",
      np.allclose(transformation, swapped))


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Graph: the original square and the three transformed shapes. Each
# shape is closed by repeating its first point at the end.

def closed(pts):
    """Append the first point so the plotted outline is closed."""
    return np.vstack([pts, pts[:1]])


fig, ax = plt.subplots(figsize=(6.5, 7))
for pts, label, style in (
        (points, "Original unit square", "k-"),
        (transformed_points, "Rotate 45 after scale (original)", "C0-"),
        (points_90, "Rotate 90 after scale (change 1)", "C1--"),
        (points_swapped, "Scale after rotate 45 (change 2)", "C2-.")):
    ax.plot(*closed(pts).T, style, marker="o", label=label)
ax.set_aspect("equal")
ax.axhline(0, color="gray", linewidth=0.6)
ax.axvline(0, color="gray", linewidth=0.6)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("2-D transformations of a unit square")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_05_matrix_algebra.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print("* Each row of 'Transformed points' is where a corner of the square")
print("  moves to. The square became a 2 x 1.5 rectangle (scaling) that is")
print("  then turned 45 degrees anticlockwise about the origin (rotation).")
print("* The determinant 3.0 = 2 x 1.5 means areas grow 3x: the unit square")
print("  (area 1) became a shape of area 3. Since det != 0 the inverse exists")
print("  and recovers the original points exactly (T @ T^-1 = I).")
print("* Changing the angle only turned the shape; changing the order of")
print("  multiplication changed the shape itself (rectangle -> rhombus),")
print("  showing that AB != BA. Area (det = 3) is the same in all cases.")
print("* Graph: the black square is the input; the coloured outlines show")
print("  the three results. Corner (0,0) never moves because rotation and")
print("  scaling are both about the origin.")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: Hill cipher encryption (cryptography).
#
# Problem:
#   Send a short text message so that it cannot be read without a key.
#   The Hill cipher turns letters into numbers (A=0, ..., Z=25), groups
#   them into vectors of length 2, and multiplies each vector by a 2x2
#   key matrix modulo 26. Decryption multiplies by the inverse key
#   matrix modulo 26.
#
# Assumptions:
#   * Message contains only the letters A-Z (spaces removed); if it has
#     an odd number of letters it is padded with "X".
#   * Key K = [[3, 3], [2, 5]]. det(K) = 3*5 - 3*2 = 9. The key is only
#     usable if det(K) has an inverse modulo 26, i.e. gcd(det, 26) = 1.
#     gcd(9, 26) = 1, and 9 * 3 = 27 = 1 (mod 26), so 9^-1 = 3 (mod 26).
#   * This is a teaching cipher, not secure for real communication.
#
# Input data:
#   message = "MEET AT NOON", key = [[3, 3], [2, 5]]
#
# Mathematical operations:
#   matrix-vector multiplication (all blocks at once as one matrix
#   product), determinant, adjugate and modular inverse:
#       K^-1 (mod 26) = det(K)^-1 * adj(K)  (mod 26)
#   where adj([[a, b], [c, d]]) = [[d, -b], [-c, a]].

MODULUS = 26
key = np.array([[3, 3],
                [2, 5]])


def text_to_blocks(text):
    """Convert letters to numbers (A=0) and reshape into 2-letter columns."""
    letters = [c for c in text.upper() if c.isalpha()]
    if len(letters) % 2:
        letters.append("X")                      # pad odd-length message
    numbers = np.array([ord(c) - ord("A") for c in letters])
    return numbers.reshape(-1, 2).T              # shape (2, n_blocks)


def blocks_to_text(blocks):
    """Convert a (2, n_blocks) matrix of numbers back to letters."""
    return "".join(chr(int(n) + ord("A")) for n in blocks.T.flatten())


def modular_inverse_2x2(matrix, modulus):
    """Inverse of a 2x2 integer matrix modulo `modulus`, via adjugate."""
    a, b = matrix[0]
    c, d = matrix[1]
    det = int(round(a * d - b * c)) % modulus
    det_inv = pow(det, -1, modulus)   # raises ValueError if not invertible
    adjugate = np.array([[d, -b],
                         [-c, a]])
    return (det_inv * adjugate) % modulus


message = "MEET AT NOON"
plain_blocks = text_to_blocks(message)

# Encrypt every block at once: one matrix product, then reduce mod 26
cipher_blocks = (key @ plain_blocks) % MODULUS
ciphertext = blocks_to_text(cipher_blocks)

key_inverse = modular_inverse_2x2(key, MODULUS)
decrypted_blocks = (key_inverse @ cipher_blocks) % MODULUS
decrypted = blocks_to_text(decrypted_blocks)

print()
print("=" * 65)
print("PART 4 - Modified case: Hill cipher encryption with a 2x2 matrix")
print("=" * 65)
print("Message           :", message)
print("Plaintext blocks  :\n", plain_blocks)
print("Key matrix K      :\n", key)
print(f"det(K) = {np.linalg.det(key):.0f}, "
      f"det(K)^-1 mod 26 = {pow(round(np.linalg.det(key)), -1, MODULUS)}")
print("Ciphertext blocks :\n", cipher_blocks)
print("Ciphertext        :", ciphertext)
print("Inverse key K^-1 (mod 26):\n", key_inverse)
print("K @ K^-1 mod 26 (should be identity):\n",
      (key @ key_inverse) % MODULUS)
print("Decrypted text    :", decrypted)

# Show why the determinant matters: a key with det = 2 cannot be inverted
bad_key = np.array([[2, 4], [1, 3]])     # det = 2*3 - 4*1 = 2
try:
    modular_inverse_2x2(bad_key, MODULUS)
    bad_key_ok = True
except ValueError:
    bad_key_ok = False
print(f"Key [[2,4],[1,3]] (det = 2) usable for decryption? {bad_key_ok}")

print()
print("Interpretation:")
print(f"* 'MEETATNOON' was encrypted to '{ciphertext}'. Repeated letters")
print("  (EE, OO) do not show up as repeated cipher letters, because each")
print("  cipher letter depends on a pair of plaintext letters.")
print("* Multiplying by K^-1 (mod 26) undid the encryption exactly, the same")
print("  way the inverse undid the graphics transformation in Part 1.")
print("* The determinant decides whether a key works: det = 9 shares no")
print("  factor with 26, so K is invertible mod 26; det = 2 shares the factor")
print("  2 with 26, so that key could never be decrypted.")

plt.show()
