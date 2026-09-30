"""Run the training pipeline across a diversified ticker basket."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

TICKERS_TO_SCREEN = {
    "Technology": ["CRM","INTC","ORCL"],
    "Financials": ["C", "BRK-B", "WFC"],
    "Healthcare": ["LLY", "VRTX", "DHR"],
    "Consumer Discretionary": ["AMZN","WMT", "HD"],
    "Consumer Staples": ["PM","MNST","MCD"],
    "Energy": ["COP", "CVX"],
    "Industrials": ["GE","CAT"],
    "Communication": ["SPOT", "GOOGL"],
    "Utilities": ["NEE","CEG"],
    "Real Estate": ["WELL","VMRK"],
    "Commodities/Niche": ["NEM","UEC","KOD"] #[ "CCJ","TSLA","MDB","BA","SMMT","NU","QS","XPEV","RDDT","RBLX","AMC","AMZN","COP","LMT","ABCL"],
#    "New": ["AAL","ORCL","COP"] #["META","KOD", "MDB", "LMT","RDDT","TSLA","NVDA"]
}

#SECTOR_OVERRIDES = {"UEC": "URA", "CCJ": "URA", "AAL": "JETS"}
SECTOR_OVERRIDES = {}
PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
MODEL_LABELS = {
    "ridge": "Ridge",
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
    "lstm": "LSTM",
}


def run_pipeline(ticker: str) -> dict | None:
    """Run train.py for one ticker and return its generated metrics."""
    print(f"Processing {ticker}...")
    command = [sys.executable, "-m", "src.train", "--ticker", ticker]
    sector_override = SECTOR_OVERRIDES.get(ticker)
    if sector_override:
        command.extend(["--sector", sector_override])

    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        with METRICS_PATH.open(encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except subprocess.CalledProcessError as error:
        print(f"Error training model for {ticker}: {error.stderr.strip()}")
    except (OSError, json.JSONDecodeError) as error:
        print(f"Error reading metrics for {ticker}: {error}")
    return None


def build_screening_row(sector: str, ticker: str, metrics: dict) -> dict:
    walk_forward = metrics.get("walk_forward", {})
    holdout_comparison = metrics.get("holdout_model_comparison", {})
    row = {
        "Sector": sector,
        "Ticker": ticker,
        "Mapped ETF": metrics.get("sector", "SPY"),
        "Trade Accuracy": walk_forward.get("trade_directional_accuracy", 0.0),
        "Strategy Return": walk_forward.get("strategy_total_return", 0.0),
        "Buy & Hold Return": walk_forward.get("buy_and_hold_total_return", 0.0),
        "Max Drawdown": walk_forward.get("strategy_max_drawdown", 0.0),
        "Total Trades": walk_forward.get("total_turnover", 0.0),
    }
    for model_name, label in MODEL_LABELS.items():
        model_metrics = holdout_comparison.get(model_name, {})
        row[f"{label} Trade Accuracy"] = model_metrics.get(
            "trade_directional_accuracy"
        )
    return row


def main() -> None:
    summary_data = []
    tickers = [ticker for sector_tickers in TICKERS_TO_SCREEN.values() for ticker in sector_tickers]
    completed = 0

    for sector, sector_tickers in TICKERS_TO_SCREEN.items():
        print(f"\n================ Sector: {sector} ================")
        for ticker in sector_tickers:
            metrics = run_pipeline(ticker)
            if metrics and "walk_forward" in metrics:
                summary_data.append(build_screening_row(sector, ticker, metrics))

            completed += 1
            if completed < len(tickers):
                time.sleep(2)

    if not summary_data:
        print("Screening finished without any parsed metrics.")
        return

    results = pd.DataFrame(summary_data)
    accuracy_columns = [
        "Trade Accuracy",
        "Max Drawdown",
        *(f"{label} Trade Accuracy" for label in MODEL_LABELS.values()),
    ]
    for column in accuracy_columns:
        results[column] = results[column].map(
            lambda value: "N/A"
            if value is None or pd.isna(value)
            else f"{value * 100:.2f}%"
        )
    for column in ("Strategy Return", "Buy & Hold Return"):
        results[column] = results[column].map(lambda value: f"{value * 100:+.2f}%")

    print("\nFINAL MULTI-SECTOR PERFORMANCE MATRIX:")
    print(results.to_string(index=False))

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    results.to_csv(artifacts_dir / "multi_sector_screening_results.csv", index=False)


if __name__ == "__main__":
    main()