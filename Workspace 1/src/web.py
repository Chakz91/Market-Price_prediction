"""Flask web service for market-price predictions."""

from __future__ import annotations

import os
import re
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

from .data import aligned_context_features, download_prices
from .features import build_features

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.=-]{0,9}$")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))


def create_app(model_path: Path = MODEL_PATH) -> Flask:
    app = Flask(__name__)
    bundle = joblib.load(model_path)

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
            prices = download_prices(ticker, period="1y")
            benchmark = download_prices(bundle.get("benchmark", "SPY"), period="1y")
            sector = download_prices(bundle.get("sector", "XLK"), period="1y")
            context = aligned_context_features(benchmark, sector)
            features = build_features(prices, context, include_target=False)
            latest = features.iloc[-1]
            predicted_return = float(bundle["model"].predict(latest[bundle["features"]].to_frame().T)[0])
            weekly_model = bundle.get("weekly_model")
            predicted_weekly_return = None
            if weekly_model is not None:
                predicted_weekly_return = float(
                    weekly_model.predict(latest[bundle["features"]].to_frame().T)[0]
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
                "predicted_close": last_close * (1 + predicted_return),
                "predicted_weekly_return": predicted_weekly_return,
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
