"""
BIT 2118 Scientific Computing - Task B (Task 03)
Mathematical Functions and Vectorization
=========================================================================

Program file : code/Task_03_vectorization.py
Run with     : python code/Task_03_vectorization.py
Output graph : figures/Task_03_vectorization.png

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
Vectorization means applying a mathematical operation to many values at
the same time. If y_i = sin^2(x_i) + cos^2(x_i), the identity gives
y_i = 1 for every x_i. Instead of looping through x_1, x_2, ..., x_n in
Python, NumPy evaluates the whole vector in one expression. The work is
still done element by element, but inside compiled C code instead of the
Python interpreter, which removes the per-element interpreter overhead.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A network-monitoring system has one million latency measurements and
needs to convert each from milliseconds to seconds and square the values
for an error calculation.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    latency_ms : 1,000,000 simulated latencies, uniformly distributed
                 between 5 ms and 250 ms (random seed 42 so the run is
                 repeatable)
Operations (applied to every element):
    seconds = latency_ms / 1000       (unit conversion ms -> s)
    squared = seconds ** 2            (square, used in error measures
                                       such as mean squared error)
The same calculation is done twice, with a Python for-loop and with a
vectorized NumPy expression, and timeit measures how long each takes.

The script is organised in four parts that map to the instructions:
    Part 1  Original solution                        (Instruction B)
    Part 2  Change an input and compare results      (Instruction C)
    Part 3  Meaning of the output and the graph      (Instruction D)
    Part 4  Modified, related real-world case        (Instruction E)
"""

from pathlib import Path
import timeit

import matplotlib.pyplot as plt
import numpy as np

# Folder where graphs are saved (created next to the code/ folder)
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Simulated network latency readings in milliseconds.
# default_rng(42) fixes the random seed so every run gives the same data.
rng = np.random.default_rng(42)
latency_ms = rng.uniform(5, 250, 1_000_000)


def loop_version(data):
    """Convert ms -> s and square, one value at a time (pure Python)."""
    squared = []
    for value in data:
        seconds = value / 1000.0
        squared.append(seconds ** 2)
    return squared


def vectorized_version(data):
    """Convert ms -> s and square the whole array in one NumPy expression."""
    seconds = data / 1000.0
    return seconds ** 2


def compare(data, vector_repeats=3):
    """Time both versions on `data` and return (loop_s, vector_s).

    The loop runs once because it is slow. The vectorized version is
    averaged over several repeats because a single run is too short to
    time accurately.
    """
    loop_time = timeit.timeit(lambda: loop_version(data), number=1)
    vector_time = timeit.timeit(lambda: vectorized_version(data),
                                number=vector_repeats) / vector_repeats
    return loop_time, vector_time


loop_time, vector_time = compare(latency_ms)
result = vectorized_version(latency_ms)

# Both methods must give the same numbers; only the speed should differ.
same_answer = np.allclose(result, loop_version(latency_ms))

print("=" * 65)
print("PART 1 - Original solution: 1,000,000 latency readings")
print("=" * 65)
print("First five latencies (ms)     :", np.round(latency_ms[:5], 2))
print("First five squared values (s^2):", result[:5])
print(f"Loop and vectorized results identical: {same_answer}")
print(f"Loop time       : {loop_time:.4f} s")
print(f"Vectorized time : {vector_time:.4f} s")
print(f"Approximate speed-up: {loop_time / vector_time:.1f}x")

# Demonstration of the identity from the concept section:
# sin^2(x) + cos^2(x) evaluated for every element of x in one line.
x = np.linspace(0, 2 * np.pi, 1_000_000)
identity = np.sin(x) ** 2 + np.cos(x) ** 2
print(f"sin^2+cos^2 over 1e6 values: min={identity.min():.15f}, "
      f"max={identity.max():.15f}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change made: the number of measurements N is varied from 1,000 to
# 1,000,000 instead of being fixed at 1,000,000.
#
# Expected effect:
#   * Loop time grows roughly in proportion to N, because the Python
#     interpreter does the same overhead work for every element.
#   * Vectorized time also grows with N, but from a much lower base, so
#     NumPy is tens to hundreds of times faster at every size.
#   * The exact speed-up varies with N and with the computer. For very
#     small N NumPy's fixed set-up cost is a larger share of its time; for
#     very large N the arrays no longer fit in the CPU cache, so the
#     vectorized code is limited by memory speed. Both effects reduce the
#     ratio, so the largest speed-up is usually seen at medium N.
#   * The computed values do not change, only the time needed.

sizes = [1_000, 10_000, 100_000, 1_000_000]
loop_times, vector_times = [], []

print()
print("=" * 65)
print("PART 2 - Effect of changing the number of measurements N")
print("=" * 65)
print(f"{'N':>10}{'Loop (s)':>14}{'Vector (s)':>14}{'Speed-up':>12}")
for n in sizes:
    lt, vt = compare(latency_ms[:n], vector_repeats=20)
    loop_times.append(lt)
    vector_times.append(vt)
    print(f"{n:>10,}{lt:>14.6f}{vt:>14.6f}{lt / vt:>11.1f}x")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Graph: execution time against N on log-log axes. On log-log axes a
# straight line means time grows in proportion to N. The gap between the
# two lines is the speed-up factor.

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(sizes, loop_times, marker="o", label="Python loop")
ax.loglog(sizes, vector_times, marker="s", label="NumPy vectorized")
ax.set_xlabel("Number of measurements N")
ax.set_ylabel("Execution time (s)")
ax.set_title("Loop vs vectorized execution time")
ax.legend()
ax.grid(True, which="both", alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_03_vectorization.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print("* Each output value is the squared latency in seconds^2, for")
print(f"  example {latency_ms[0]:.2f} ms -> {latency_ms[0] / 1000:.5f} s "
      f"-> {result[0]:.6f} s^2.")
print("* The loop and vectorized versions give identical numbers; the only")
print(f"  difference is speed: about {loop_time / vector_time:.0f}x faster "
      "with NumPy for 1e6 values.")
print("* sin^2+cos^2 = 1 holds for all 1e6 values up to round-off (~1e-16),")
print("  showing whole-array evaluation of mathematical functions.")
speedups = np.array(loop_times) / np.array(vector_times)
print("* Graph: both lines rise almost linearly with N (time is proportional")
print("  to the number of values), but the loop line sits one to two orders")
print("  of magnitude higher. The vertical gap is the speed-up, which ranged")
print(f"  from {speedups.min():.0f}x to {speedups.max():.0f}x in this run "
      f"(largest at N = {sizes[int(np.argmax(speedups))]:,}).")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: gamma correction of a grayscale image (image processing).
#
# Problem:
#   A security camera produces dark 8-bit grayscale frames. Before motion
#   detection, every pixel must be brightened with gamma correction:
#       out = 255 * (pixel / 255) ** gamma,   with gamma < 1 to brighten.
#   A 1080p frame has 1920 x 1080 = 2,073,600 pixels and the camera sends
#   many frames per second, so the per-pixel work must be fast.
#
# Assumptions:
#   * The frame is simulated: random pixel values 0-80, i.e. a dark image
#     (real frames would be loaded from the camera).
#   * gamma = 0.5 is used; this is a common brightening value.
#   * Output pixels are rounded and stored as 8-bit integers (0-255).
#
# Input data:
#   frame : 2-D array of shape (1080, 1920), dtype uint8
#   gamma : 0.5
#
# Mathematical operations:
#   division, power and multiplication applied element-wise to the whole
#   2-D matrix, then rounding and type conversion, all vectorized.

frame = rng.integers(0, 81, size=(1080, 1920), dtype=np.uint8)
GAMMA = 0.5


def gamma_loop(img, gamma):
    """Gamma-correct one pixel at a time using nested Python loops."""
    rows, cols = img.shape
    out = np.empty_like(img)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = round(255 * (img[r, c] / 255) ** gamma)
    return out


def gamma_vectorized(img, gamma):
    """Gamma-correct every pixel of the image in one NumPy expression."""
    return np.round(255 * (img / 255.0) ** gamma).astype(np.uint8)


# The loop is timed on a 200 x 200 crop only, because the full frame
# would take many seconds; its time is then scaled up to the full frame.
crop = frame[:200, :200]
loop_crop_time = timeit.timeit(lambda: gamma_loop(crop, GAMMA), number=1)
loop_full_estimate = loop_crop_time * frame.size / crop.size
vector_full_time = timeit.timeit(lambda: gamma_vectorized(frame, GAMMA),
                                 number=5) / 5

corrected = gamma_vectorized(frame, GAMMA)
crop_match = np.array_equal(gamma_loop(crop, GAMMA), corrected[:200, :200])

print()
print("=" * 65)
print("PART 4 - Modified case: gamma correction of a 1080p camera frame")
print("=" * 65)
print(f"Frame shape: {frame.shape}, pixels: {frame.size:,}, gamma = {GAMMA}")
print(f"Mean brightness before: {frame.mean():6.2f} / 255")
print(f"Mean brightness after : {corrected.mean():6.2f} / 255")
print(f"Pixel 40 -> {gamma_vectorized(np.array([40], np.uint8), GAMMA)[0]}, "
      f"pixel 80 -> {gamma_vectorized(np.array([80], np.uint8), GAMMA)[0]}")
print(f"Loop and vectorized agree on test crop: {crop_match}")
print(f"Loop time (estimated, full frame): {loop_full_estimate:.3f} s")
print(f"Vectorized time (full frame)     : {vector_full_time:.4f} s")
print(f"Speed-up: {loop_full_estimate / vector_full_time:.0f}x")
print(f"Frames per second possible: loop ~{1 / loop_full_estimate:.1f}, "
      f"vectorized ~{1 / vector_full_time:.0f}")

print()
print("Interpretation:")
print("* Gamma correction raised the average brightness of the frame by")
print(f"  about {corrected.mean() / frame.mean():.1f}x. Dark pixels gain the")
print("  most (40 -> 101 is 2.5x, while 80 -> 143 is 1.8x), which reveals")
print("  detail in shadows without saturating the brighter areas.")
print("* With the loop the camera could process only a fraction of a frame")
print("  per second, which is too slow for live video. The vectorized")
print("  version handles many frames per second on the same computer.")

plt.show()
