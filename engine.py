"""engine.py — Network Centrality / Systemic Risk engine.

Daily loop:
  1. Build partial correlation graph (ETFs + macro nodes)
  2. Compute four centrality measures → composite systemic risk score
  3. Compute EWM expected return
  4. Final score = expected_return / (1 + alpha * systemic_risk)
  5. Rank ETFs, output top-N
"""

from __future__ import annotations

import pandas as pd

import config
from centrality import systemic_risk_score
from graph_builder import adjacency_etf_submatrix, build_adjacency


def run_engine(
    log_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    universe_tickers: list[str],
    universe_name: str,
) -> dict:
    """Run the Network Centrality engine for one universe."""
    avail = [t for t in universe_tickers if t in log_returns.columns]
    dates = log_returns.index
    T = len(dates)

    print(
        f"\n{'='*60}\n"
        f"Universe: {universe_name}  ({len(avail)} ETFs)\n"
        f"Period: {dates[0].date()} → {dates[-1].date()}  ({T} days)\n"
        f"{'='*60}"
    )

    # Output containers
    daily_records: list[dict] = []
    centrality_records: list[dict] = []
    ranking_records: list[dict] = []

    for i, date in enumerate(dates):
        if i < config.COV_WINDOW:
            continue

        # ── Build graph ───────────────────────────────────────────────────────
        adj_full, all_nodes = build_adjacency(
            log_returns[avail],
            macro_df,
            end_idx=i,
            window=config.COV_WINDOW,
            threshold=config.EDGE_THRESHOLD,
        )

        # ETF-only submatrix for centrality
        adj_etf, etf_nodes = adjacency_etf_submatrix(adj_full, all_nodes, avail)

        # ── Centrality scores ─────────────────────────────────────────────────
        risk_scores = systemic_risk_score(adj_etf, etf_nodes)

        # ── EWM expected returns ──────────────────────────────────────────────
        window_rets = log_returns[avail].iloc[
            max(0, i - config.EWM_SPAN_RETURN * 3) : i
        ]
        ewm_ret = (
            window_rets.ewm(span=config.EWM_SPAN_RETURN, min_periods=10).mean().iloc[-1]
            * 252  # annualise
        )

        # ── Final score = return / (1 + alpha * systemic_risk) ───────────────
        scores = {}
        for ticker in avail:
            if ticker not in risk_scores:
                continue
            exp_ret = float(ewm_ret.get(ticker, 0.0))
            sys_risk = risk_scores[ticker]["composite"]
            final = exp_ret / (1.0 + config.RETURN_RISK_ALPHA * sys_risk)
            scores[ticker] = {
                "expected_return": round(exp_ret, 6),
                "systemic_risk": round(sys_risk, 6),
                "degree": round(risk_scores[ticker]["degree"], 6),
                "eigenvector": round(risk_scores[ticker]["eigenvector"], 6),
                "pagerank": round(risk_scores[ticker]["pagerank"], 6),
                "betweenness": round(risk_scores[ticker]["betweenness"], 6),
                "final_score": round(final, 6),
            }

        # ── Rank and select top N ─────────────────────────────────────────────
        ranked = sorted(scores.items(), key=lambda x: x[1]["final_score"], reverse=True)
        top_n = ranked[: config.TOP_N]

        # ── Count graph edges (density) ───────────────────────────────────────
        n_edges = int((adj_etf > 0).sum() / 2)
        n_possible = len(etf_nodes) * (len(etf_nodes) - 1) / 2
        density = n_edges / n_possible if n_possible > 0 else 0.0

        # ── Record ────────────────────────────────────────────────────────────
        daily_records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "n_edges": n_edges,
                "graph_density": round(density, 4),
                "top_ticker": top_n[0][0] if top_n else "",
                "top_score": top_n[0][1]["final_score"] if top_n else 0.0,
                "most_central": (
                    max(scores.items(), key=lambda x: x[1]["systemic_risk"])[0]
                    if scores
                    else ""
                ),
            }
        )

        centrality_records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                **{t: v["systemic_risk"] for t, v in scores.items()},
            }
        )

        ranking_records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                **{t: v["final_score"] for t, v in scores.items()},
            }
        )

        if i % 252 == 0 or i == T - 1:
            print(
                f"  {date.date()} | "
                f"edges={n_edges} density={density:.2f} | "
                f"top={top_n[0][0] if top_n else 'N/A'} "
                f"(score={top_n[0][1]['final_score']:.4f}) | "
                f"most_central={daily_records[-1]['most_central']}"
            )

    # ── Latest snapshot ───────────────────────────────────────────────────────
    latest_date = dates[-1]
    latest_i = T - 1

    adj_final, nodes_final = build_adjacency(
        log_returns[avail],
        macro_df,
        end_idx=latest_i,
        window=config.COV_WINDOW,
        threshold=config.EDGE_THRESHOLD,
    )
    adj_etf_final, etf_nodes_final = adjacency_etf_submatrix(
        adj_final, nodes_final, avail
    )
    risk_final = systemic_risk_score(adj_etf_final, etf_nodes_final)

    window_final = log_returns[avail].iloc[
        max(0, latest_i - config.EWM_SPAN_RETURN * 3) : latest_i
    ]
    ewm_final = (
        window_final.ewm(span=config.EWM_SPAN_RETURN, min_periods=10).mean().iloc[-1]
        * 252
    )

    latest_scores = {}
    for ticker in avail:
        if ticker not in risk_final:
            continue
        exp_ret = float(ewm_final.get(ticker, 0.0))
        sys_risk = risk_final[ticker]["composite"]
        final = exp_ret / (1.0 + config.RETURN_RISK_ALPHA * sys_risk)
        latest_scores[ticker] = {
            "expected_return": round(exp_ret, 6),
            "systemic_risk": round(sys_risk, 6),
            "degree": round(risk_final[ticker]["degree"], 6),
            "eigenvector": round(risk_final[ticker]["eigenvector"], 6),
            "pagerank": round(risk_final[ticker]["pagerank"], 6),
            "betweenness": round(risk_final[ticker]["betweenness"], 6),
            "final_score": round(final, 6),
        }

    latest_ranked = sorted(
        latest_scores.items(), key=lambda x: x[1]["final_score"], reverse=True
    )

    # ── Adjacency matrix for latest snapshot (ETF submatrix) ─────────────────
    adj_dict = {
        etf_nodes_final[i]: {
            etf_nodes_final[j]: round(float(adj_etf_final[i, j]), 4)
            for j in range(len(etf_nodes_final))
            if adj_etf_final[i, j] > 0
        }
        for i in range(len(etf_nodes_final))
    }

    print(
        f"\n  Latest ({latest_date.date()}) top-{config.TOP_N}: "
        + " | ".join(
            f"{t}(score={v['final_score']:.3f}, risk={v['systemic_risk']:.3f})"
            for t, v in latest_ranked[: config.TOP_N]
        )
    )

    return {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "latest_scores": latest_scores,
        "latest_ranked": [(t, v) for t, v in latest_ranked],
        "latest_adjacency": adj_dict,
        "daily_df": pd.DataFrame(daily_records).set_index("date"),
        "centrality_df": pd.DataFrame(centrality_records).set_index("date"),
        "ranking_df": pd.DataFrame(ranking_records).set_index("date"),
        "universe": universe_name,
    }
