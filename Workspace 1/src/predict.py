"""Predict the next-session return from the latest market data."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .data import aligned_context_features, download_prices
from .features import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the next-session return")
    parser.add_argument("--ticker", default=None, help="Ticker; defaults to the training ticker")
    parser.add_argument("--model", default="artifacts/model.joblib")
    args = parser.parse_args()

    bundle = joblib.load(Path(args.model))
    ticker = (args.ticker or bundle["ticker"]).upper()
    prices = download_prices(ticker, period="1y")
    benchmark = download_prices(bundle.get("benchmark", "SPY"), period="1y")
    sector = download_prices(bundle.get("sector", "XLK"), period="1y")
    features = build_features(prices, aligned_context_features(benchmark, sector), include_target=False)
    latest = features.iloc[-1]
    predicted_return = float(bundle["model"].predict(latest[bundle["features"]].to_frame().T)[0])
    weekly_model = bundle.get("weekly_model")
    predicted_weekly_return = None
    if weekly_model is not None:
        predicted_weekly_return = float(
            weekly_model.predict(latest[bundle["features"]].to_frame().T)[0]
        )
    last_close = float(prices["Close"].iloc[-1])

    print(f"Ticker: {ticker}")
    print(f"Last close: {last_close:.2f}")
    print(f"Predicted next-session return: {predicted_return:+.2%}")
    print(f"Illustrative predicted close: {last_close * (1 + predicted_return):.2f}")
    if predicted_weekly_return is not None:
        print(f"Predicted 1-week return: {predicted_weekly_return:+.2%}")
        print(f"Illustrative 1-week predicted close: {last_close * (1 + predicted_weekly_return):.2f}")
    print("This is a statistical estimate, not financial advice.")


if __name__ == "__main__":
    main()
