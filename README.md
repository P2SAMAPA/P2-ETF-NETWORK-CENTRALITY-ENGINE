# 🕸️ P2-ETF-NETWORK-CENTRALITY

**P2Quant Engine** · Network Centrality / Systemic Risk Ranking

[![Daily Network Centrality Engine](https://github.com/P2SAMAPA/P2-ETF-NETWORK-CENTRALITY/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-NETWORK-CENTRALITY/actions/workflows/daily_run.yml)

---

## What Is This?

This engine builds a **daily partial correlation graph** of ETFs and macro
variables, computes four network centrality measures, and ranks ETFs by their
**return-to-systemic-risk ratio**. High centrality = more connected = higher
contagion risk = underweight.

The key insight: in a financial crisis, the most central nodes in the
correlation network are the first to transmit shocks. By explicitly measuring
and penalising centrality, the engine avoids concentration in systemically
dangerous assets.

---

## Scoring Formula

```
Final Score = Expected Return / (1 + α × Systemic Risk Score)
```

Where:
- **Expected Return** = EWM(log_returns, span=21) × 252 (annualised)
- **Systemic Risk Score** = weighted composite of 4 centrality measures ∈ [0, 1]
- **α = 1.0** (risk penalty scaling)

---

## Four Centrality Measures

| Measure | Weight | What It Captures |
|---|---|---|
| **Eigenvector** | 30% | Importance weighted by neighbours' importance — recursive systemic risk |
| **PageRank** | 30% | Random-walk importance — contagion propagation probability |
| **Betweenness** | 25% | How often ETF sits on shortest path between others — contagion bridge |
| **Degree** | 15% | Raw weighted connectivity — number and strength of direct links |

---

## Graph Construction

- **Nodes:** ETF tickers + macro variables (VIX, DXY, T10Y2Y, TBILL_3M)
- **Edges:** |partial correlation| > 0.20 threshold
- **Edge weights:** partial correlation magnitude
- **Partial correlation:** computed via Ledoit-Wolf precision matrix (controls for common macro factors — removes spurious ETF-ETF links driven by shared macro exposure)
- **Rolling window:** 63 trading days (~1 quarter)
- **Update frequency:** daily

---

## Universes

| Universe | Tickers |
|---|---|
| EQUITY_SECTORS | SPY QQQ XLK XLF XLE XLV XLI XLY XLP XLU GDX XME IWF XSD XBI IWM |
| COMBINED | All above + TLT VCIT LQD HYG VNQ GLD SLV |

---

## Output

Results pushed daily to HuggingFace:
- `netcent_YYYY-MM-DD_{universe}.json` — latest scores, rankings, adjacency matrix
- `daily_{universe}.csv` — graph diagnostics (edges, density, most central node)
- `centrality_{universe}.csv` — full daily centrality history per ETF
- `rankings_{universe}.csv` — full daily final score history per ETF

**Results repo:** [P2SAMAPA/p2-etf-network-centrality-results](https://huggingface.co/datasets/P2SAMAPA/p2-etf-network-centrality-results)

---

## Streamlit Dashboard — 5 Tabs

1. **Rankings & Risk** — return-to-risk bar chart, systemic risk bar chart, centrality component breakdown, top-N recommendation cards
2. **Network Graph** — interactive circular network visualisation (node size = risk, colour = risk level, edge thickness = correlation strength)
3. **Centrality History** — time series of systemic risk per ETF, risk heatmap (last 252 days)
4. **Graph Diagnostics** — edge count over time, graph density, most-central-node frequency
5. **Full Scores Table** — all four centrality measures + final score per ETF

---

## References

- Mantegna, R.N. (1999). *Hierarchical Structure in Financial Markets*. EPJB.
- Billio, M. et al. (2012). *Econometric Measures of Connectedness and Systemic Risk*. JFE.
- Brin, S. & Page, L. (1998). *The Anatomy of a Large-Scale Hypertextual Web Search Engine*. WWW.
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. JMVA.

---

*P2Quant Engine Suite · Built by P2SAMAPA*
