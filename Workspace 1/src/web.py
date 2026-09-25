"""Flask web service for market-price predictions."""

from __future__ import annotations

import os
import re
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

from .data import aligned_context_features, download_prices
from .evaluation import directional_return_calibration, model_factory, predict_probability
from .features import build_features, get_feature_columns, get_sequence_feature_columns

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.=-]{0,9}$")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
TRAIN_START = os.getenv("TRAIN_START", "2015-01-01")


def create_app(model_path: Path = MODEL_PATH) -> Flask:
    app = Flask(__name__)
    bundle = joblib.load(model_path)
    trained_models: dict[str, tuple[object, object, list[str], list[str], dict[str, dict[str, float]]]] = {}

    def train_for_ticker(ticker: str):
        if ticker in trained_models:
            return trained_models[ticker]

        model_name = bundle.get("model_name")
        if model_name is None:
            model_name = "lstm" if bundle["model"].__class__.__name__ == "LSTMRegressor" else "random_forest"

        prices = download_prices(ticker, start=TRAIN_START)
        benchmark = download_prices(bundle.get("benchmark", "SPY"), start=TRAIN_START)
        sector = download_prices(bundle.get("sector", "XLK"), start=TRAIN_START)
        volatility_index = download_prices(
            bundle.get("volatility_index", "^VIX"), start=TRAIN_START
        )
        context = aligned_context_features(benchmark, sector, volatility_index)
        dataset = build_features(prices, context, include_target=True)
        feature_columns = get_feature_columns(dataset)
        sequence_feature_columns = get_sequence_feature_columns(dataset)
        model_columns = sequence_feature_columns if model_name == "lstm" else feature_columns

        model = model_factory(model_name)
        model.fit(dataset[model_columns], dataset["target"])
        weekly_model = model_factory(model_name)
        weekly_model.fit(dataset[model_columns], dataset["target_5d"])
        calibration = {
            "1d": directional_return_calibration(
                dataset, "target", "target_return_1d"
            ),
            "5d": directional_return_calibration(
                dataset, "target_5d", "target_return_5d"
            ),
        }
        trained_models[ticker] = (
            model,
            weekly_model,
            feature_columns,
            sequence_feature_columns,
            calibration,
        )
        return trained_models[ticker]

    @app.get("/")
    def index():
        return render_template("index.html", default_ticker=bundle["ticker"])

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_ticker": bundle["ticker"]})

    @app.get("/api/predict")
    def predict():
        raw_ticker = request.args.get("ticker", bundle["ticker"])
        ticker = raw_ticker.strip().upper()
        if not TICKER_PATTERN.fullmatch(ticker):
            return jsonify({"error": "Enter a valid stock symbol, such as AAPL or MSFT."}), 400

        try:
            model, weekly_model, feature_columns, sequence_feature_columns, calibration = train_for_ticker(ticker)
            prices = download_prices(ticker, period="1y")
            benchmark = download_prices(bundle.get("benchmark", "SPY"), period="1y")
            sector = download_prices(bundle.get("sector", "XLK"), period="1y")
            volatility_index = download_prices(
                bundle.get("volatility_index", "^VIX"), period="1y"
            )
            context = aligned_context_features(benchmark, sector, volatility_index)
            features = build_features(prices, context, include_target=False)
            latest = features.iloc[-1]
            model_columns = sequence_feature_columns if model_name == "lstm" else feature_columns
            latest_values = latest[model_columns].to_frame().T
            predicted_probability = float(predict_probability(model, latest_values)[0])
            predicted_weekly_probability = float(
                predict_probability(weekly_model, latest_values)[0]
            )
            predicted_return = (
                predicted_probability * calibration["1d"]["up_return"]
                + (1 - predicted_probability) * calibration["1d"]["down_return"]
            )
            predicted_weekly_return = (
                predicted_weekly_probability * calibration["5d"]["up_return"]
                + (1 - predicted_weekly_probability) * calibration["5d"]["down_return"]
            )
            last_close = float(prices["Close"].iloc[-1])
            session_date = prices.index[-1].date().isoformat()
        except Exception as error:
            app.logger.exception("Prediction failed for %s", ticker)
            return jsonify({"error": f"Could not retrieve or analyze {ticker}: {error}"}), 502

        return jsonify(
            {
                "ticker": ticker,
                "as_of": session_date,
                "last_close": last_close,
                "predicted_return": predicted_return,
                "predicted_probability": predicted_probability,
                "predicted_close": last_close * (1 + predicted_return),
                "predicted_weekly_return": predicted_weekly_return,
                "predicted_weekly_probability": predicted_weekly_probability,
                "predicted_weekly_close": (
                    last_close * (1 + predicted_weekly_return)
                    if predicted_weekly_return is not None
                    else None
                ),
                "data_notice": "Yahoo Finance data may be delayed; this is not exchange-grade real-time data.",
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
