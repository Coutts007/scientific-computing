"""
BIT 2118 Scientific Computing - Task H (Task 09)
Eigenvalues and Eigenvectors
=========================================================================

Program file : code/Task_09_eigenvalues.py
Run with     : python code/Task_09_eigenvalues.py
Output graphs: figures/Task_09_eigenvalues.png   (network centrality)
               figures/Task_09_pagerank.png      (Part 4 case)

-------------------------------------------------------------------------
1. MATHEMATICAL CONCEPT
-------------------------------------------------------------------------
For a square matrix A, an eigenvector v is a non-zero vector whose
direction is unchanged by A:
        A v = lambda v
The eigenvalue lambda tells how much v is stretched (|lambda| > 1),
shrunk (|lambda| < 1) or reversed (lambda < 0). Eigenvalues are the roots
of the characteristic equation det(A - lambda I) = 0.

Dominant eigenvector and the power method:
    Repeatedly multiplying any starting vector by A and rescaling,
        x_{k+1} = A x_k / ||A x_k||,
    turns it towards the eigenvector of the largest |lambda| (the
    dominant one), because that component grows fastest.

Eigenvector centrality in a network:
    With adjacency matrix A (a_ij = 1 if nodes i and j are linked), the
    score of a node should be proportional to the sum of its neighbours'
    scores:  x_i = (1/lambda) sum_j a_ij x_j,  i.e.  A x = lambda x.
    For a connected network the dominant eigenvector has all entries of
    the same sign (Perron-Frobenius theorem), so it gives a valid
    importance ranking: being linked to important nodes makes you
    important.

-------------------------------------------------------------------------
2. REAL-WORLD PROBLEM (from the assignment)
-------------------------------------------------------------------------
A small computer network needs to identify which nodes are structurally
most influential from their pattern of connections. Represent the
network as an adjacency matrix, compute its dominant eigenvalue and
eigenvector, and normalise the eigenvector to obtain node importance.

-------------------------------------------------------------------------
3. INPUT VALUES AND MATHEMATICAL OPERATIONS  (Instruction B)
-------------------------------------------------------------------------
Inputs:
    A : 4x4 adjacency matrix of an undirected network with nodes A-D and
        links A-B, A-C, B-C, B-D, C-D
Operations:
    np.linalg.eig(A)        -> all eigenvalues and eigenvectors
    argmax of eigenvalues   -> dominant eigenvalue/eigenvector
    abs, divide by sum      -> sign fix and normalisation (scores sum
                               to 1)
    ||A v - lambda v||      -> verification of the eigen-equation
    power method            -> the same vector, found by iteration

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

np.set_printoptions(precision=4, suppress=True)


def eigen_centrality(adjacency):
    """Return (dominant eigenvalue, normalised centrality, raw vector)."""
    eigenvalues, eigenvectors = np.linalg.eig(adjacency)
    index = np.argmax(np.real(eigenvalues))
    dominant_value = np.real(eigenvalues[index])
    vector = np.real(eigenvectors[:, index])
    # eig may return the vector multiplied by -1; abs() fixes the sign.
    # Dividing by the sum turns the scores into shares that add up to 1.
    centrality = np.abs(vector) / np.abs(vector).sum()
    return dominant_value, centrality, vector


def power_method(matrix, tol=1e-12, max_iter=1000):
    """Find the dominant eigenvector by repeated multiplication.

    Returns (eigenvalue estimate, vector normalised to sum 1, list of
    changes per iteration).
    """
    x = np.ones(matrix.shape[0]) / matrix.shape[0]   # equal start scores
    changes = []
    for _ in range(max_iter):
        x_new = matrix @ x
        x_new /= x_new.sum()                         # rescale
        change = np.linalg.norm(x_new - x, ord=1)
        changes.append(change)
        x = x_new
        if change < tol:
            break
    # Rayleigh quotient gives the eigenvalue for the final vector
    eigenvalue = (x @ matrix @ x) / (x @ x)
    return eigenvalue, x, changes


# =========================================================================
# PART 1 - ORIGINAL SOLUTION  (Instruction B)
# =========================================================================

# Adjacency matrix for a small undirected network.
# Node order: A, B, C, D.  a_ij = 1 means nodes i and j are connected.
A = np.array([
    [0., 1., 1., 0.],
    [1., 0., 1., 1.],
    [1., 1., 0., 1.],
    [0., 1., 1., 0.]
])
nodes = ["A", "B", "C", "D"]

eigenvalues, _ = np.linalg.eig(A)
dominant_value, centrality, v = eigen_centrality(A)

print("=" * 65)
print("PART 1 - Original solution: 4-node network")
print("=" * 65)
print("All eigenvalues:", np.sort(np.real(eigenvalues))[::-1])
print("Dominant eigenvalue:", dominant_value)
print("Eigenvector-based importance:")
for node, score, degree in zip(nodes, centrality, A.sum(axis=1)):
    print(f"{node}: {score:.4f}   (links: {degree:.0f})")

# Verify A v = lambda v
print("Verification error:", np.linalg.norm(A @ v - dominant_value * v))

# Same answer by the power method
pm_value, pm_vector, pm_changes = power_method(A)
print(f"Power method: eigenvalue {pm_value:.6f} after {len(pm_changes)} "
      f"iterations, scores {pm_vector}")


# =========================================================================
# PART 2 - CHANGE AN INPUT AND OBSERVE THE EFFECT  (Instruction C)
# =========================================================================
# Change: a new node E is connected to node A only (e.g. a new laptop
# plugged into router A). The adjacency matrix grows to 5x5.
#
# Expected effect:
#   * The dominant eigenvalue increases, because the network now has more
#     links (it is at least as "connected" as before).
#   * A's score rises: it gains a neighbour, so its degree goes from 2
#     to 3 (equal to B and C).
#   * B and C still score higher than A, even though all three now have
#     3 links, because B and C are linked to well-connected nodes while
#     A's new neighbour E is linked to nobody else. Eigenvector
#     centrality rewards WHO you are connected to, not just how many.
#   * E gets the lowest score: one link, to a node of middling rank.

A_new = np.zeros((5, 5))
A_new[:4, :4] = A
A_new[0, 4] = A_new[4, 0] = 1.0          # new link A-E
nodes_new = nodes + ["E"]

new_value, new_centrality, _ = eigen_centrality(A_new)

print()
print("=" * 65)
print("PART 2 - Effect of adding node E linked to A")
print("=" * 65)
print(f"Dominant eigenvalue: {dominant_value:.4f} -> {new_value:.4f}")
print(f"{'Node':<6}{'Links':>6}{'Before':>10}{'After':>10}")
for i, node in enumerate(nodes_new):
    before = f"{centrality[i]:.4f}" if i < 4 else "-"
    print(f"{node:<6}{A_new[i].sum():>6.0f}{before:>10}"
          f"{new_centrality[i]:>10.4f}")


# =========================================================================
# PART 3 - MEANING OF THE OUTPUT AND THE GRAPH  (Instruction D)
# =========================================================================
# Graph: both networks drawn with node size and colour proportional to
# centrality, plus the power-method convergence.

positions = {"A": (0, 1), "B": (1, 2), "C": (1, 0), "D": (2, 1),
             "E": (-1, 1)}


def draw_network(ax, adjacency, names, scores, title):
    """Draw nodes sized by centrality and the links between them."""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if adjacency[i, j]:
                (x1, y1), (x2, y2) = positions[names[i]], positions[names[j]]
                ax.plot([x1, x2], [y1, y2], color="gray", zorder=1)
    xs = [positions[n][0] for n in names]
    ys = [positions[n][1] for n in names]
    ax.scatter(xs, ys, s=6000 * scores, c=scores, cmap="viridis",
               vmin=0.1, vmax=0.3, zorder=2, edgecolors="black")
    for name, x, y, score in zip(names, xs, ys, scores):
        ax.annotate(f"{name}\n{score:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, -48), ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_xlim(-1.6, 2.6)
    ax.set_ylim(-0.8, 2.6)
    ax.axis("off")


fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
draw_network(axes[0], A, nodes, centrality,
             f"Original network (lambda = {dominant_value:.3f})")
draw_network(axes[1], A_new, nodes_new, new_centrality,
             f"Node E added (lambda = {new_value:.3f})")
axes[2].semilogy(range(1, len(pm_changes) + 1), pm_changes, marker=".")
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Change in scores")
axes[2].set_title("Power method convergence (original)")
axes[2].grid(True, which="both", alpha=0.4)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "Task_09_eigenvalues.png", dpi=120)

print()
print("=" * 65)
print("PART 3 - Interpretation")
print("=" * 65)
print(f"* B and C each hold {centrality[1]:.1%} of the importance: they")
print("  have the most links (3) and are linked to each other. A and D")
print(f"  ({centrality[0]:.1%} each) depend on B and C to reach the rest.")
print("  A failure of B or C would disrupt the network the most, so they")
print("  should be monitored and backed up first.")
print(f"* The dominant eigenvalue {dominant_value:.3f} lies between the "
      "average degree (2.5)")
print("  and the maximum degree (3); it measures overall connectivity.")
print("* Verification error ~1e-15 confirms A v = lambda v holds.")
print("* After adding E, A's share rose but stayed below B and C even")
print("  though all three now have 3 links: connections to well-connected")
print("  nodes count for more than connections to isolated ones.")
print("* Graph: larger, brighter nodes are more central. The right panel")
print("  shows the power method's change falling in a straight line on a")
print("  log scale, i.e. it converges by a constant factor each step")
print("  (the ratio of the 2nd largest to the largest |eigenvalue|).")


# =========================================================================
# PART 4 - MODIFIED REAL-WORLD CASE  (Instruction E)
# =========================================================================
# New case: ranking web pages with Google's PageRank.
#
# Problem:
#   A small university website has 6 pages. Links between pages are
#   one-directional (directed). Rank the pages by importance so the site
#   search shows the most important pages first.
#
# Model:
#   * Link matrix M: M[i, j] = 1 / (number of links on page j) if page j
#     links to page i. Each column sums to 1, so M describes a "random
#     surfer" who clicks a random link on the current page.
#   * With probability d = 0.85 the surfer follows a link; with
#     probability 1 - d they jump to a random page. The Google matrix
#         G = d M + (1 - d)/n * (matrix of ones)
#     is column-stochastic, so its dominant eigenvalue is exactly 1.
#   * The PageRank vector r satisfies G r = 1 * r with sum(r) = 1: the
#     long-run share of time the surfer spends on each page.
#
# Assumptions:
#   * Damping factor d = 0.85 (the value used in the original PageRank
#     paper).
#   * Every page has at least one outgoing link (no dead ends).
#   * Link structure below is illustrative.
#
# Input data (page -> pages it links to):
#   Home      -> Courses, Admissions, News
#   Courses   -> Home, Library
#   Admissions-> Home, Fees
#   News      -> Home, Courses
#   Library   -> Courses
#   Fees      -> Admissions

pages = ["Home", "Courses", "Admissions", "News", "Library", "Fees"]
links = {
    "Home": ["Courses", "Admissions", "News"],
    "Courses": ["Home", "Library"],
    "Admissions": ["Home", "Fees"],
    "News": ["Home", "Courses"],
    "Library": ["Courses"],
    "Fees": ["Admissions"],
}
DAMPING = 0.85
n = len(pages)
index = {page: k for k, page in enumerate(pages)}

M = np.zeros((n, n))
for source, targets in links.items():
    for target in targets:
        M[index[target], index[source]] = 1 / len(targets)

G = DAMPING * M + (1 - DAMPING) / n * np.ones((n, n))

# Method 1: eigen-decomposition, eigenvalue closest to 1
g_values, g_vectors = np.linalg.eig(G)
k = np.argmin(np.abs(g_values - 1))
pagerank_eig = np.abs(np.real(g_vectors[:, k]))
pagerank_eig /= pagerank_eig.sum()

# Method 2: power method (how search engines compute it at web scale)
_, pagerank_pm, pr_changes = power_method(G)

# Effect of the damping factor on the ranking
damping_values = np.linspace(0.05, 0.95, 19)
ranks_by_d = []
for d in damping_values:
    G_d = d * M + (1 - d) / n * np.ones((n, n))
    _, r, _ = power_method(G_d)
    ranks_by_d.append(r)
ranks_by_d = np.array(ranks_by_d)

order = np.argsort(pagerank_eig)[::-1]
in_links = (M > 0).sum(axis=1)

print()
print("=" * 65)
print("PART 4 - Modified case: PageRank of a 6-page website")
print("=" * 65)
print("Column sums of M (should all be 1):", M.sum(axis=0))
print(f"Dominant eigenvalue of G: {np.real(g_values[k]):.6f}")
print(f"Power method converged in {len(pr_changes)} iterations; max "
      f"difference from eig: {np.max(np.abs(pagerank_pm - pagerank_eig)):.1e}")
print()
print(f"{'Rank':<6}{'Page':<12}{'PageRank':>10}{'In-links':>10}")
for rank, p in enumerate(order, start=1):
    print(f"{rank:<6}{pages[p]:<12}{pagerank_eig[p]:>10.4f}"
          f"{in_links[p]:>10}")

fig2, (ax_bar, ax_d) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_bar.bar([pages[p] for p in order], pagerank_eig[order], color="C0")
ax_bar.axhline(1 / n, color="gray", linestyle="--", label="Equal share 1/6")
ax_bar.set_ylabel("PageRank")
ax_bar.set_title(f"PageRank of site pages (d = {DAMPING})")
ax_bar.legend()
ax_bar.grid(True, axis="y", alpha=0.4)
for p in range(n):
    ax_d.plot(damping_values, ranks_by_d[:, p], marker=".", label=pages[p])
ax_d.axvline(DAMPING, color="gray", linestyle="--")
ax_d.set_xlabel("Damping factor d")
ax_d.set_ylabel("PageRank")
ax_d.set_title("Effect of the damping factor")
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig(FIGURE_DIR / "Task_09_pagerank.png", dpi=120)

top, bottom = order[0], order[-1]
print()
print("Interpretation:")
print(f"* {pages[top]} ranks first ({pagerank_eig[top]:.1%}): a random "
      "visitor spends that share")
print("  of their time on it in the long run, because most pages link")
print("  back to it.")
print("* Rank depends on who links to you, not only how many: Library,")
print("  Fees and News each have exactly 1 in-link, yet score differently.")
print("  Library's link comes from high-ranked Courses, which shares its")
print("  vote over only 2 links. News is linked only from Home, whose vote")
print("  is split 3 ways, so despite Home's high rank,")
print(f"  {pages[bottom]} ranks last ({pagerank_eig[bottom]:.1%}).")
print("* Graph (right): with small d the surfer mostly jumps at random, so")
print("  every page tends towards 1/6. As d grows, the link structure")
print("  dominates and the gaps between pages widen. The order can even")
print("  change: Courses is ahead of Home for d below about 0.7, and Home")
print("  only takes first place at higher d. The chosen d is part of the model.")
print("* The dominant eigenvalue is exactly 1, as theory predicts for a")
print("  column-stochastic matrix; the eigenvector is the ranking.")

plt.show()
