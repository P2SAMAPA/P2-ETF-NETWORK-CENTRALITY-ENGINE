"""trainer.py — Network Centrality engine orchestrator."""

from __future__ import annotations

import io
import json
import os

from huggingface_hub import HfApi

import config
import data_manager
from engine import run_engine


def push_results(result: dict, universe: str, token: str) -> None:
    slug = universe.lower().replace("_", "-")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    # Summary JSON
    output = {
        "run_date": config.TODAY,
        "universe": universe,
        "latest_date": result["latest_date"],
        "latest_scores": result["latest_scores"],
        "latest_ranked": [{"ticker": t, **v} for t, v in result["latest_ranked"]],
        "latest_adjacency": result["latest_adjacency"],
        "config": {
            "cov_window": config.COV_WINDOW,
            "edge_threshold": config.EDGE_THRESHOLD,
            "centrality_weights": config.CENTRALITY_WEIGHTS,
            "top_n": config.TOP_N,
        },
    }
    api.upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(output, indent=2, default=str).encode()),
        path_in_repo=f"netcent_{config.TODAY}_{slug}.json",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Network Centrality results {config.TODAY} — {slug}",
    )

    # Daily diagnostics CSV
    api.upload_file(
        path_or_fileobj=io.BytesIO(result["daily_df"].to_csv().encode()),
        path_in_repo=f"daily_{slug}.csv",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Daily diagnostics {config.TODAY} — {slug}",
    )

    # Centrality history CSV
    api.upload_file(
        path_or_fileobj=io.BytesIO(result["centrality_df"].to_csv().encode()),
        path_in_repo=f"centrality_{slug}.csv",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Centrality history {config.TODAY} — {slug}",
    )

    # Ranking history CSV
    api.upload_file(
        path_or_fileobj=io.BytesIO(result["ranking_df"].to_csv().encode()),
        path_in_repo=f"rankings_{slug}.csv",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Rankings history {config.TODAY} — {slug}",
    )

    print(f"Pushed → {config.HF_OUTPUT_REPO}/netcent_{config.TODAY}_{slug}.json")


def main() -> None:
    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting.")
        return

    target = os.environ.get("NC_UNIVERSE", "ALL").upper()
    log_returns, macro_df = data_manager.load_data(token=token)

    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue
        result = run_engine(
            log_returns=log_returns,
            macro_df=macro_df,
            universe_tickers=tickers,
            universe_name=universe_name,
        )
        push_results(result, universe_name, token)

    print("\n✅ Network Centrality engine complete.")


if __name__ == "__main__":
    main()
