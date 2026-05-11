"""centrality.py — Four network centrality measures on weighted adjacency matrix.

All measures are computed from scratch using numpy/scipy — no networkx
dependency required, which avoids install issues on GitHub Actions.
"""
from __future__ import annotations

import numpy as np
import config

# ── 1. Degree Centrality ──────────────────────────────────────────────────────

def degree_centrality(adj: np.ndarray) -> np.ndarray:
    """Normalised weighted degree: sum of edge weights / (n-1)."""
    n = adj.shape[0]
    if n <= 1:
        return np.zeros(n)
    deg = adj.sum(axis=1)
    return deg / (n - 1)


# ── 2. Eigenvector Centrality ─────────────────────────────────────────────────

def eigenvector_centrality(
    adj: np.ndarray, max_iter: int = 200, tol: float = 1e-6
) -> np.ndarray:
    """Power iteration eigenvector centrality on weighted adjacency matrix.

    Converges to the dominant eigenvector — nodes connected to high-centrality
    nodes get higher scores. Reflects systemic importance recursively.
    """
    n = adj.shape[0]
    if n <= 1:
        return np.ones(n)
    x = np.ones(n) / n
    for _ in range(max_iter):
        x_new = adj @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    # Ensure non-negative and normalise to [0, 1]
    x = np.abs(x)
    x_max = x.max()
    return x / x_max if x_max > 1e-10 else x


# ── 3. PageRank ───────────────────────────────────────────────────────────────

def pagerank_centrality(
    adj: np.ndarray, damping: float = 0.85, max_iter: int = 200, tol: float = 1e-6
) -> np.ndarray:
    """Weighted PageRank via power iteration.

    damping=0.85 means 85% of the time the random walker follows an edge,
    15% teleports uniformly — prevents sink nodes from dominating.
    """
    n = adj.shape[0]
    if n <= 1:
        return np.ones(n) / max(n, 1)
    # Row-normalise to get transition matrix
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M = adj / row_sums  # row-stochastic
    # PageRank: r = d * M^T @ r + (1-d)/n
    r = np.ones(n) / n
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        r_new = damping * (M.T @ r) + teleport
        if np.linalg.norm(r_new - r, ord=1) < tol:
            r = r_new
            break
        r = r_new
    r_max = r.max()
    return r / r_max if r_max > 1e-10 else r


# ── 4. Betweenness Centrality ─────────────────────────────────────────────────

def betweenness_centrality(adj: np.ndarray) -> np.ndarray:
    """Approximate betweenness centrality via shortest paths (Floyd-Warshall).

    Betweenness(v) = sum over s,t of (shortest_paths_through_v / total_shortest_paths_s_t)

    High betweenness = contagion bridge — removing this node disconnects the graph.
    Uses inverse weights (1/w) so strong correlations = short distances.
    """
    n = adj.shape[0]
    if n <= 2:
        return np.zeros(n)

    # Convert to distance matrix: strong correlation = short distance
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(adj > 1e-8, 1.0 / adj, np.inf)
    np.fill_diagonal(dist, 0.0)

    # Floyd-Warshall: all-pairs shortest paths + path counts
    D = dist.copy()

    # sigma[i, j] = number of shortest paths from i to j
    sigma = np.zeros((n, n))
    # Initialise: self-paths = 1; direct edges get count 1
    sigma[D == 0] = 1.0           # diagonal (self)
    sigma[D < np.inf] = 1.0       # any finite direct edge

    for k in range(n):
        # Distance through intermediate node k
        through_k = D[:, k : k + 1] + D[k : k + 1, :]  # shape (n, n)

        # Strictly shorter path found through k — reset sigma
        improved = through_k < D - 1e-10
        np.fill_diagonal(improved, False)

        # Equally short path through k — accumulate sigma
        tied = np.abs(through_k - D) < 1e-10
        np.fill_diagonal(tied, False)
        tied &= (D < np.inf)       # only where a path already exists

        # Outer-product style: sigma_through_k[i,j] = sigma[i,k] * sigma[k,j]
        sigma_through_k = sigma[:, k : k + 1] * sigma[k : k + 1, :]

        sigma[improved] = sigma_through_k[improved]
        sigma[tied] += sigma_through_k[tied]

        D = np.minimum(D, through_k)

    # Betweenness: for each node v, count how often it lies on shortest paths
    btw = np.zeros(n)
    for v in range(n):
        for s in range(n):
            for t in range(n):
                if s == t or s == v or t == v:
                    continue
                if D[s, t] >= np.inf:
                    continue
                path_s_v = D[s, v]
                path_v_t = D[v, t]
                if path_s_v + path_v_t - D[s, t] < 1e-8:
                    # v lies on a shortest path from s to t
                    btw[v] += sigma[s, v] * sigma[v, t] / max(sigma[s, t], 1.0)

    # Normalise
    norm = (n - 1) * (n - 2)
    btw = btw / norm if norm > 0 else btw
    btw_max = btw.max()
    return btw / btw_max if btw_max > 1e-10 else btw


# ── Composite systemic risk score ─────────────────────────────────────────────

def systemic_risk_score(
    adj: np.ndarray,
    nodes: list[str],
    weights: dict[str, float] = config.CENTRALITY_WEIGHTS,
) -> dict[str, float]:
    """Compute composite systemic risk score for each node.

    Returns dict: node_name → composite_centrality_score in [0, 1].
    Higher score = more systemically important = higher contagion risk.
    """
    c_deg = degree_centrality(adj)
    c_eig = eigenvector_centrality(adj)
    c_pr  = pagerank_centrality(adj)
    c_btw = betweenness_centrality(adj)

    composite = (
        weights["degree"]      * c_deg
        + weights["eigenvector"] * c_eig
        + weights["pagerank"]    * c_pr
        + weights["betweenness"] * c_btw
    )

    # Normalise composite to [0, 1]
    c_max = composite.max()
    if c_max > 1e-10:
        composite /= c_max

    return {
        node: {
            "composite":    float(composite[i]),
            "degree":       float(c_deg[i]),
            "eigenvector":  float(c_eig[i]),
            "pagerank":     float(c_pr[i]),
            "betweenness":  float(c_btw[i]),
        }
        for i, node in enumerate(nodes)
    }
