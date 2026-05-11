"""config.py — Network Centrality / Systemic Risk engine configuration."""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-network-centrality-results"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
EQUITY_SECTORS_TICKERS = [
    "SPY",
    "QQQ",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "GDX",
    "XME",
    "IWF",
    "XSD",
    "XBI",
    "IWM",
]
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
COMBINED_TICKERS = sorted(set(EQUITY_SECTORS_TICKERS + FI_COMMODITIES_TICKERS))

UNIVERSES = {
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": COMBINED_TICKERS,
}

# ── Macro nodes added to graph ────────────────────────────────────────────────
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# ── Graph construction ────────────────────────────────────────────────────────
COV_WINDOW = 63  # rolling window for correlation/partial correlation
EDGE_THRESHOLD = 0.20  # |partial_corr| threshold for edge inclusion
GRAPH_UPDATE_FREQ = 1  # update graph every N days (1 = daily)

# ── Centrality weights (sum to 1) ─────────────────────────────────────────────
CENTRALITY_WEIGHTS = {
    "eigenvector": 0.30,  # importance by neighbour importance
    "pagerank": 0.30,  # random-walk systemic importance
    "betweenness": 0.25,  # contagion bridge score
    "degree": 0.15,  # raw connectivity count
}

# ── Scoring ───────────────────────────────────────────────────────────────────
EWM_SPAN_RETURN = 21  # EWM span for expected return estimation
TOP_N = 6  # top N ETFs in output recommendation
RETURN_RISK_ALPHA = 1.0  # denominator scaling: score = ret / (1 + alpha * risk)

# ── Output ────────────────────────────────────────────────────────────────────
TODAY = datetime.now().strftime("%Y-%m-%d")
