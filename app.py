"""app.py — Network Centrality / Systemic Risk Dashboard."""

from __future__ import annotations

import os
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from us_calendar import next_trading_day

st.set_page_config(
    page_title="Network Centrality · P2Quant",
    layout="wide",
    page_icon="🕸️",
)

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

RISK_COLOURS = {
    "low": "#27AE60",
    "medium": "#F39C12",
    "high": "#E74C3C",
}
CENTRALITY_COLOURS = {
    "eigenvector": "#1B4F8A",
    "pagerank": "#27AE60",
    "betweenness": "#E74C3C",
    "degree": "#F39C12",
}


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading centrality results…")
def load_json(universe: str) -> dict | None:
    slug = universe.lower().replace("_", "-")
    try:
        r = requests.get(BASE_API, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        files = sorted(f["path"] for f in r.json() if f["path"].endswith(".json"))
        matches = [f for f in files if f"_{slug}.json" in f]
        if not matches:
            return None
        resp = requests.get(f"{BASE_RAW}/{matches[-1]}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading history…")
def load_csv(filename: str) -> pd.DataFrame | None:
    try:
        r = requests.get(f"{BASE_RAW}/{filename}", headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def risk_colour(score: float) -> str:
    if score >= 0.66:
        return RISK_COLOURS["high"]
    elif score >= 0.33:
        return RISK_COLOURS["medium"]
    return RISK_COLOURS["low"]


def risk_label(score: float) -> str:
    if score >= 0.66:
        return "HIGH RISK"
    elif score >= 0.33:
        return "MEDIUM RISK"
    return "LOW RISK"


def fmt_pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🕸️ Network Centrality · Systemic Risk Engine")
st.caption(
    "Partial correlation graph · Eigenvector + PageRank + Betweenness + Degree centrality · "
    "High centrality = systemic risk = underweight · Score = Return / (1 + Risk)"
)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    universe = st.selectbox("Universe", list(config.UNIVERSES.keys()))
    st.divider()
    st.markdown(f"**ETFs:** {len(config.UNIVERSES[universe])}")
    st.markdown(f"**Cov window:** {config.COV_WINDOW} days")
    st.markdown(f"**Edge threshold:** |pcorr| > {config.EDGE_THRESHOLD}")
    st.markdown(f"**Top N:** {config.TOP_N}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    st.divider()
    st.markdown("**Centrality weights:**")
    for k, v in config.CENTRALITY_WEIGHTS.items():
        st.markdown(f"- {k}: {v:.0%}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

slug = universe.lower().replace("_", "-")
data = load_json(universe)
daily_df = load_csv(f"daily_{slug}.csv")
centrality_df = load_csv(f"centrality_{slug}.csv")
ranking_df = load_csv(f"rankings_{slug}.csv")

if data is None:
    st.warning("⚠️ No results found. Run `python trainer.py` first.")
    st.stop()

latest_scores = data.get("latest_scores", {})
latest_ranked = data.get("latest_ranked", [])
latest_adj = data.get("latest_adjacency", {})
latest_date = data.get("latest_date", "?")

# ── KPI row ───────────────────────────────────────────────────────────────────
h1, h2, h3 = st.columns(3)
h1.metric("Run Date", data.get("run_date", "?"))
h2.metric("Latest Date", latest_date)
h3.metric("Universe", universe)

if latest_ranked:
    top = latest_ranked[0]
    most_risky = max(latest_ranked, key=lambda x: x["systemic_risk"])
    least_risky = min(latest_ranked, key=lambda x: x["systemic_risk"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top Pick", top["ticker"], help="Highest return-to-risk score")
    k2.metric("Top Score", f"{top['final_score']:.4f}")
    k3.metric(
        "Most Central (avoid)",
        most_risky["ticker"],
        delta=f"risk={most_risky['systemic_risk']:.3f}",
        delta_color="inverse",
    )
    k4.metric(
        "Safest Node",
        least_risky["ticker"],
        delta=f"risk={least_risky['systemic_risk']:.3f}",
        delta_color="normal",
    )

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Rankings & Risk",
        "🕸️ Network Graph",
        "📈 Centrality History",
        "📊 Graph Diagnostics",
        "📋 Full Scores Table",
    ]
)

# ── Tab 1: Rankings & Risk ────────────────────────────────────────────────────
with tab1:
    st.subheader(f"ETF Rankings as of {latest_date}")

    if latest_ranked:
        left, right = st.columns(2)

        with left:
            st.markdown("**Return-to-Risk Score (Final Ranking)**")
            tickers_r = [r["ticker"] for r in latest_ranked]
            scores_r = [r["final_score"] for r in latest_ranked]
            colours_r = [risk_colour(r["systemic_risk"]) for r in latest_ranked]

            fig_rank = go.Figure(
                go.Bar(
                    y=tickers_r,
                    x=scores_r,
                    orientation="h",
                    marker_color=colours_r,
                    text=[f"{s:.4f}" for s in scores_r],
                    textposition="outside",
                )
            )
            fig_rank.add_vline(x=0, line_dash="dot", line_color="gray")
            fig_rank.update_layout(
                title="Final Score = Expected Return / (1 + Systemic Risk)",
                xaxis_title="Score",
                yaxis=dict(autorange="reversed"),
                height=max(300, len(tickers_r) * 30),
                margin=dict(t=50, b=20, l=60, r=80),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_rank, use_container_width=True, key="rank_bar")

        with right:
            st.markdown("**Systemic Risk Score (composite centrality)**")
            risk_vals = [r["systemic_risk"] for r in latest_ranked]

            fig_risk = go.Figure(
                go.Bar(
                    y=tickers_r,
                    x=risk_vals,
                    orientation="h",
                    marker_color=[risk_colour(v) for v in risk_vals],
                    text=[f"{v:.3f}" for v in risk_vals],
                    textposition="outside",
                )
            )
            fig_risk.update_layout(
                title="Composite Systemic Risk (0=safe, 1=most central)",
                xaxis_title="Systemic Risk Score",
                yaxis=dict(autorange="reversed"),
                height=max(300, len(tickers_r) * 30),
                margin=dict(t=50, b=20, l=60, r=80),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_risk, use_container_width=True, key="risk_bar")

        # Per-centrality breakdown for latest date
        st.markdown("**Centrality Component Breakdown**")
        comp_tickers = [r["ticker"] for r in latest_ranked]
        fig_comp = go.Figure()
        for measure, colour in CENTRALITY_COLOURS.items():
            vals = [latest_scores.get(t, {}).get(measure, 0) for t in comp_tickers]
            fig_comp.add_trace(
                go.Bar(
                    name=measure.capitalize(),
                    x=comp_tickers,
                    y=vals,
                    marker_color=colour,
                )
            )
        fig_comp.update_layout(
            barmode="group",
            title="Individual Centrality Measures by ETF",
            yaxis_title="Centrality Score (0–1)",
            height=380,
            margin=dict(t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_comp, use_container_width=True, key="comp_bar")

        # Top N recommendation box
        st.markdown(
            f"### 🎯 Top {config.TOP_N} Recommendation for {next_trading_day()}"
        )
        cols = st.columns(config.TOP_N)
        for i, row in enumerate(latest_ranked[: config.TOP_N]):
            with cols[i]:
                rc = risk_colour(row["systemic_risk"])
                st.markdown(
                    f"**#{i+1} {row['ticker']}**\n\n"
                    f"Score: `{row['final_score']:.4f}`\n\n"
                    f"Exp. Ret: `{fmt_pct(row['expected_return'])}`\n\n"
                    f'<span style="background:{rc};color:white;padding:2px 8px;'
                    f'border-radius:8px;font-size:11px">'
                    f"{risk_label(row['systemic_risk'])}</span>",
                    unsafe_allow_html=True,
                )

# ── Tab 2: Network Graph ──────────────────────────────────────────────────────
with tab2:
    st.subheader(f"ETF Correlation Network — {latest_date}")
    st.caption(
        "Nodes sized by systemic risk. Edges = |partial correlation| > threshold. "
        "Red nodes = high centrality (avoid). Green = low centrality (prefer)."
    )

    if latest_adj and latest_scores:
        etf_list = list(latest_scores.keys())
        n = len(etf_list)

        # Layout: circle
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {etf: (np.cos(a), np.sin(a)) for etf, a in zip(etf_list, angles)}

        fig_net = go.Figure()

        # Draw edges
        for i, ticker_i in enumerate(etf_list):
            for j, ticker_j in enumerate(etf_list):
                if j <= i:
                    continue
                w = latest_adj.get(ticker_i, {}).get(ticker_j, 0)
                if w <= 0:
                    continue
                x0, y0 = pos[ticker_i]
                x1, y1 = pos[ticker_j]
                fig_net.add_trace(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode="lines",
                        line=dict(
                            width=w * 4, color=f"rgba(100,100,200,{min(w, 0.6)})"
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

        # Draw nodes
        node_x = [pos[t][0] for t in etf_list]
        node_y = [pos[t][1] for t in etf_list]
        node_risk = [latest_scores[t]["systemic_risk"] for t in etf_list]
        node_score = [latest_scores[t]["final_score"] for t in etf_list]
        node_colours = [risk_colour(r) for r in node_risk]
        node_sizes = [12 + 28 * r for r in node_risk]

        fig_net.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=etf_list,
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=node_sizes,
                    color=node_colours,
                    line=dict(width=1.5, color="white"),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Systemic Risk: %{customdata[0]:.3f}<br>"
                    "Final Score: %{customdata[1]:.4f}<br>"
                    "<extra></extra>"
                ),
                customdata=list(zip(node_risk, node_score)),
                showlegend=False,
            )
        )

        fig_net.update_layout(
            title=f"ETF Partial Correlation Network — {universe}",
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20, l=20, r=20),
        )
        # Legend annotation
        fig_net.add_annotation(
            text="🟢 Low Risk  🟡 Medium Risk  🔴 High Risk  (node size = systemic risk)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.02,
            showarrow=False,
            font=dict(size=10),
        )
        st.plotly_chart(fig_net, use_container_width=True, key="network_graph")
    else:
        st.info("No adjacency data available.")

# ── Tab 3: Centrality History ─────────────────────────────────────────────────
with tab3:
    st.subheader("Centrality History Over Time")
    if centrality_df is not None:
        avail_tickers = [
            c for c in centrality_df.columns if c in config.UNIVERSES[universe]
        ]

        selected = st.multiselect(
            "Select ETFs",
            avail_tickers,
            default=avail_tickers[:6],
            key="cent_select",
        )

        if selected:
            fig_cent = go.Figure()
            colours = [
                "#1B4F8A",
                "#27AE60",
                "#E74C3C",
                "#F39C12",
                "#8E44AD",
                "#148F77",
                "#CA6F1E",
                "#2471A3",
            ]
            for i, t in enumerate(selected):
                if t in centrality_df.columns:
                    fig_cent.add_trace(
                        go.Scatter(
                            x=centrality_df.index,
                            y=centrality_df[t],
                            mode="lines",
                            name=t,
                            line=dict(width=1.5, color=colours[i % len(colours)]),
                        )
                    )
            fig_cent.add_hline(
                y=0.66,
                line_dash="dash",
                line_color="#E74C3C",
                annotation_text="High risk threshold",
            )
            fig_cent.add_hline(
                y=0.33,
                line_dash="dash",
                line_color="#F39C12",
                annotation_text="Medium risk threshold",
            )
            fig_cent.update_layout(
                title="Composite Systemic Risk Score Over Time",
                yaxis_title="Systemic Risk (0=safe, 1=most central)",
                height=420,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cent, use_container_width=True, key="cent_history")

        # Centrality heatmap — latest 252 days
        if len(centrality_df) > 0:
            recent = centrality_df[avail_tickers].tail(252)
            fig_heat = go.Figure(
                go.Heatmap(
                    z=recent.values.T,
                    x=recent.index.strftime("%Y-%m-%d"),
                    y=avail_tickers,
                    colorscale="RdYlGn_r",  # red=high risk
                    zmid=0.5,
                    colorbar=dict(title="Systemic Risk"),
                    hoverongaps=False,
                )
            )
            fig_heat.update_layout(
                title="Systemic Risk Heatmap (last 252 days)",
                height=max(300, len(avail_tickers) * 22 + 80),
                margin=dict(t=40, b=60, l=60, r=20),
                xaxis=dict(tickangle=-45, nticks=12),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heat, use_container_width=True, key="cent_heatmap")
    else:
        st.info("No centrality history found.")

# ── Tab 4: Graph Diagnostics ──────────────────────────────────────────────────
with tab4:
    st.subheader("Graph Structure Diagnostics")
    if daily_df is not None:
        c1, c2 = st.columns(2)

        with c1:
            fig_edges = go.Figure(
                go.Scatter(
                    x=daily_df.index,
                    y=daily_df["n_edges"],
                    mode="lines",
                    line=dict(color="#1B4F8A", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(27,79,138,0.1)",
                )
            )
            fig_edges.update_layout(
                title="Number of Graph Edges Over Time",
                yaxis_title="Edges",
                height=280,
                margin=dict(t=40, b=30),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_edges, use_container_width=True, key="edges_chart")

        with c2:
            fig_dens = go.Figure(
                go.Scatter(
                    x=daily_df.index,
                    y=daily_df["graph_density"],
                    mode="lines",
                    line=dict(color="#8E44AD", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(142,68,173,0.1)",
                )
            )
            fig_dens.update_layout(
                title="Graph Density Over Time",
                yaxis_title="Density (0=sparse, 1=full)",
                height=280,
                margin=dict(t=40, b=30),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dens, use_container_width=True, key="density_chart")

        # Most central ETF over time
        if "most_central" in daily_df.columns:
            central_counts = daily_df["most_central"].value_counts()
            fig_central = go.Figure(
                go.Bar(
                    x=central_counts.index,
                    y=central_counts.values,
                    marker_color="#E74C3C",
                    text=central_counts.values,
                    textposition="outside",
                )
            )
            fig_central.update_layout(
                title="Most Central ETF (days as highest-risk node)",
                yaxis_title="Days",
                height=300,
                margin=dict(t=40, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_central, use_container_width=True, key="central_bar")
    else:
        st.info("No daily diagnostics found.")

# ── Tab 5: Full Scores Table ──────────────────────────────────────────────────
with tab5:
    st.subheader(f"Full Scores — {latest_date}")
    if latest_ranked:
        rows = []
        for row in latest_ranked:
            rows.append(
                {
                    "Ticker": row["ticker"],
                    "Final Score": f"{row['final_score']:.4f}",
                    "Exp. Return": fmt_pct(row["expected_return"]),
                    "Systemic Risk": f"{row['systemic_risk']:.4f}",
                    "Eigenvector": f"{row.get('eigenvector', 0):.4f}",
                    "PageRank": f"{row.get('pagerank', 0):.4f}",
                    "Betweenness": f"{row.get('betweenness', 0):.4f}",
                    "Degree": f"{row.get('degree', 0):.4f}",
                    "Risk Label": risk_label(row["systemic_risk"]),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=600,
        )

st.divider()
st.caption(
    f"P2Quant Network Centrality Engine · Run: {data.get('run_date', '?')} · "
    f"Partial correlation graph · Ledoit-Wolf precision matrix · "
    f"Data: {config.HF_DATA_REPO}"
)
