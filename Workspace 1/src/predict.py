"""Predict the next-session return from the latest market data."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .data import aligned_context_features, download_prices
from .evaluation import predict_probability
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
    volatility_index = download_prices(bundle.get("volatility_index", "^VIX"), period="1y")
    features = build_features(
        prices,
        aligned_context_features(benchmark, sector, volatility_index),
        include_target=False,
    )
    latest = features.iloc[-1]
    model_name = bundle.get("model_name")
    if model_name is None:
        model_name = "lstm" if bundle["model"].__class__.__name__ == "LSTMRegressor" else "random_forest"
    feature_key = "sequence_features" if model_name == "lstm" else "features"
    latest_values = latest[bundle.get(feature_key, bundle["features"])].to_frame().T
    predicted_probability = float(predict_probability(bundle["model"], latest_values)[0])
    calibration = bundle.get("return_calibration", {})
    daily_calibration = calibration.get("1d", {"up_return": 0.0, "down_return": 0.0})
    predicted_return = (
        predicted_probability * daily_calibration["up_return"]
        + (1 - predicted_probability) * daily_calibration["down_return"]
    )
    weekly_model = bundle.get("weekly_model")
    predicted_weekly_return = None
    if weekly_model is not None:
        predicted_weekly_probability = float(
            predict_probability(weekly_model, latest_values)[0]
        )
        weekly_calibration = calibration.get("5d", {"up_return": 0.0, "down_return": 0.0})
        predicted_weekly_return = (
            predicted_weekly_probability * weekly_calibration["up_return"]
            + (1 - predicted_weekly_probability) * weekly_calibration["down_return"]
        )
    last_close = float(prices["Close"].iloc[-1])

    print(f"Ticker: {ticker}")
    print(f"Last close: {last_close:.2f}")
    print(f"Predicted next-session return: {predicted_return:+.2%}")
    print(f"Probability of next-session increase: {predicted_probability:.2%}")
    print(f"Illustrative predicted close: {last_close * (1 + predicted_return):.2f}")
    if predicted_weekly_return is not None:
        print(f"Predicted 1-week return: {predicted_weekly_return:+.2%}")
        print(f"Illustrative 1-week predicted close: {last_close * (1 + predicted_weekly_return):.2f}")
    print("This is a statistical estimate, not financial advice.")


if __name__ == "__main__":
    main()
