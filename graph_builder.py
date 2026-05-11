"""graph_builder.py — Build partial correlation graph from returns + macro.

Partial correlation (rather than Pearson) is used because it controls for
the common macro factors — removing the confounding effect of VIX/DXY on
all ETF correlations simultaneously, leaving only the direct ETF-ETF links.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import config


def partial_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute partial correlation matrix via precision matrix (inverse covariance).

    Partial corr(i,j) = -precision(i,j) / sqrt(precision(i,i) * precision(j,j))

    Uses Ledoit-Wolf shrinkage for stability with small T/N ratios.
    Falls back to standard correlation if precision computation fails.
    """
    n, p = data.shape
    if n < p + 5:
        # Not enough observations — fall back to Pearson correlation
        return np.corrcoef(data.T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lw = LedoitWolf()
            lw.fit(data)
            precision = lw.precision_
        except Exception:
            cov = np.cov(data.T) + np.eye(p) * 1e-6
            precision = np.linalg.inv(cov)

    # Normalise to partial correlations
    diag = np.diag(precision)
    with np.errstate(divide="ignore", invalid="ignore"):
        pcorr = -precision / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(pcorr, 1.0)
    pcorr = np.clip(pcorr, -1.0, 1.0)
    return pcorr


def build_adjacency(
    log_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    end_idx: int,
    window: int = config.COV_WINDOW,
    threshold: float = config.EDGE_THRESHOLD,
) -> tuple[np.ndarray, list[str]]:
    """Build weighted adjacency matrix for one time window.

    Nodes = ETF tickers (returns) + macro variables (levels, standardised).
    Edges = |partial_corr| > threshold, weighted by partial correlation magnitude.

    Returns
    -------
    adj   : (N, N) weighted adjacency matrix (absolute partial correlations)
    nodes : list of node names (ETFs first, then macro)
    """
    start_idx = max(0, end_idx - window)

    etf_window = log_returns.iloc[start_idx:end_idx]
    macro_window = macro_df.iloc[start_idx:end_idx]

    # Standardise macro to same scale as log returns
    macro_std = (macro_window - macro_window.mean()) / (macro_window.std() + 1e-8)
    macro_std = macro_std * etf_window.std().mean()  # rescale to return magnitude

    # Combined data matrix: ETFs + macro
    combined = pd.concat([etf_window, macro_std], axis=1).dropna()
    nodes = list(combined.columns)

    if len(combined) < 30:
        n = len(nodes)
        return np.zeros((n, n)), nodes

    pcorr = partial_correlation_matrix(combined.values)

    # Threshold: keep only edges above threshold
    adj = np.abs(pcorr)
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 0.0)  # no self-loops

    return adj, nodes


def adjacency_etf_submatrix(
    adj: np.ndarray,
    nodes: list[str],
    etf_tickers: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Extract the ETF-only submatrix from the full adjacency matrix."""
    etf_idx = [i for i, n in enumerate(nodes) if n in etf_tickers]
    etf_nodes = [nodes[i] for i in etf_idx]
    etf_adj = adj[np.ix_(etf_idx, etf_idx)]
    return etf_adj, etf_nodes
