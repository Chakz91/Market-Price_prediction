"""Flask web service for market-price predictions."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

from .data import (
    aligned_context_features,
    build_macro_sentiment_features,
    download_prices,
    download_ticker_news_sentiment,
    get_sector_benchmark,
)
from .evaluation import (
    compare_probability_models,
    directional_return_calibration,
    fit_classifier_or_constant,
    model_factory,
    predict_probability,
)
from .features import build_features, get_feature_columns, get_sequence_feature_columns
from .train import (
    _probability_model_candidates,
    get_regime_hyperparameters,
    ticker_model_artifact_path,
)

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.=-]{0,9}$")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
TRAIN_START = os.getenv("TRAIN_START", "2015-01-01")
MAX_TICKER_MODEL_AGE = timedelta(days=60)


def create_app(model_path: Path = MODEL_PATH) -> Flask:
    app = Flask(__name__)
    bundle = joblib.load(model_path)
    trained_models: dict[
        str,
        tuple[
            object,
            object,
            list[str],
            list[str],
            dict[str, dict[str, float]],
            str,
            str,
            str,
            bool,
        ],
    ] = {}
    trained_model_times: dict[str, datetime] = {}

    def train_for_ticker(ticker: str):
        now = datetime.now(timezone.utc)
        cached_model_time = trained_model_times.get(ticker)
        if ticker in trained_models and cached_model_time is not None:
            cached_age = now - cached_model_time
            if timedelta(0) <= cached_age <= MAX_TICKER_MODEL_AGE:
                return trained_models[ticker]
            trained_models.pop(ticker, None)
            trained_model_times.pop(ticker, None)

        ticker_artifact = ticker_model_artifact_path(ticker, model_path.parent)
        if ticker_artifact.is_file():
            try:
                saved_bundle = joblib.load(ticker_artifact)
            except Exception:
                app.logger.exception(
                    "Could not load ticker model artifact %s", ticker_artifact
                )
            else:
                if not isinstance(saved_bundle, dict):
                    app.logger.warning(
                        "Ignoring non-dictionary model artifact for %s", ticker
                    )
                    saved_bundle = {}
                required_keys = {
                    "ticker",
                    "model",
                    "weekly_model",
                    "model_name",
                    "features",
                    "sequence_features",
                    "return_calibration",
                    "trained_at",
                }
                saved_ticker = str(saved_bundle.get("ticker", "")).strip().upper()
                if saved_ticker != ticker:
                    app.logger.warning(
                        "Ignoring model artifact for %s while serving %s",
                        saved_ticker or "unknown ticker",
                        ticker,
                    )
                elif not required_keys.issubset(saved_bundle):
                    app.logger.warning(
                        "Ignoring incomplete model artifact for %s", ticker
                    )
                else:
                    try:
                        saved_model_time = saved_bundle["trained_at"]
                        if isinstance(saved_model_time, str):
                            saved_model_time = datetime.fromisoformat(
                                saved_model_time.replace("Z", "+00:00")
                            )
                        if not isinstance(saved_model_time, datetime):
                            raise ValueError("trained_at must be a datetime or ISO string")
                        if saved_model_time.tzinfo is None:
                            saved_model_time = saved_model_time.replace(
                                tzinfo=timezone.utc
                            )
                        saved_model_time = saved_model_time.astimezone(timezone.utc)
                    except (TypeError, ValueError):
                        saved_model_time = None
                    saved_age = (
                        now - saved_model_time if saved_model_time is not None else None
                    )
                    if saved_age is None or not (
                        timedelta(0) <= saved_age <= MAX_TICKER_MODEL_AGE
                    ):
                        app.logger.warning(
                            "Ignoring expired or invalid model artifact for %s",
                            ticker,
                        )
                    else:
                        saved_model_name = saved_bundle["model_name"]
                        saved_weekly_model_name = saved_bundle.get(
                            "weekly_model_name", saved_model_name
                        )
                        cached_bundle = (
                            saved_bundle["model"],
                            saved_bundle["weekly_model"],
                            list(saved_bundle.get("features", [])),
                            list(saved_bundle.get("sequence_features", [])),
                            saved_bundle["return_calibration"],
                            saved_bundle.get("sector") or get_sector_benchmark(ticker),
                            saved_model_name,
                            saved_weekly_model_name,
                            bool(saved_bundle.get("is_mean_reverter", False)),
                        )
                        trained_models[ticker] = cached_bundle
                        trained_model_times[ticker] = saved_model_time
                        return cached_bundle

        sector_symbol = bundle.get("sector") if ticker == bundle.get("ticker") else None
        sector_symbol = sector_symbol or get_sector_benchmark(ticker)
        prices = download_prices(ticker, start=TRAIN_START)
        news_df = download_ticker_news_sentiment(ticker)
        regime_params = get_regime_hyperparameters(prices["Close"], sector_symbol)
        is_mean_reverter = bool(regime_params["is_mean_reverter"])
        horizon = 5 if is_mean_reverter else 1
        target_column = "target_5d" if is_mean_reverter else "target"
        forced_model_name = (
            "xgboost"
            if regime_params.get("force_xgboost", False)
            else "ridge"
            if regime_params.get("force_linear", False)
            else None
        )
        regime_model_kwargs = {
            key: value
            for key, value in regime_params.items()
            if key not in {"is_mean_reverter", "force_linear", "force_xgboost"}
        }
        benchmark = download_prices(bundle.get("benchmark", "SPY"), start=TRAIN_START)
        sector = download_prices(sector_symbol, start=TRAIN_START)
        volatility_index = download_prices(
            bundle.get("volatility_index", "^VIX"), start=TRAIN_START
        )
        context = aligned_context_features(benchmark, sector, volatility_index)
        macro_sentiment_df = build_macro_sentiment_features(
            bundle.get("benchmark", "SPY"), sector_symbol
        )
        dataset = build_features(
            prices,
            context,
            include_target=True,
            news_df=news_df,
            macro_sentiment_df=macro_sentiment_df,
        )
        feature_columns = get_feature_columns(dataset, mean_reversion=is_mean_reverter)
        sequence_feature_columns = get_sequence_feature_columns(
            dataset, mean_reversion=is_mean_reverter
        )
        if forced_model_name is None:
            feature_sets = {
                "ridge": dataset[feature_columns],
                "gradient_boosting": dataset[feature_columns],
                "random_forest": dataset[feature_columns],
                "lstm": dataset[sequence_feature_columns],
            }
            candidates = _probability_model_candidates(regime_model_kwargs, horizon)
            ticker_model_name, model_kwargs, _primary_validation = (
                compare_probability_models(
                    feature_sets,
                    dataset[target_column],
                    candidates,
                    folds=3,
                    gap=horizon,
                )
            )
            ticker_weekly_model_name, weekly_model_kwargs, _weekly_validation = (
                compare_probability_models(
                    feature_sets,
                    dataset["target_5d"],
                    _probability_model_candidates(regime_model_kwargs, 5),
                    folds=3,
                    gap=5,
                )
            )
        else:
            ticker_model_name = forced_model_name
            ticker_weekly_model_name = forced_model_name
            model_kwargs = {}
            weekly_model_kwargs = {}
        if ticker_model_name == "lstm":
            model_kwargs.setdefault("validation_gap", horizon)
        if ticker_weekly_model_name == "lstm":
            weekly_model_kwargs.setdefault("validation_gap", 5)
        model_columns = (
            sequence_feature_columns
            if ticker_model_name == "lstm"
            else feature_columns
        )
        weekly_model_columns = (
            sequence_feature_columns
            if ticker_weekly_model_name == "lstm"
            else feature_columns
        )

        model = model_factory(ticker_model_name, **model_kwargs)
        model = fit_classifier_or_constant(
            model, dataset[model_columns], dataset[target_column]
        )
        weekly_model = model_factory(
            ticker_weekly_model_name, **weekly_model_kwargs
        )
        weekly_model = fit_classifier_or_constant(
            weekly_model, dataset[weekly_model_columns], dataset["target_5d"]
        )
        calibration = {
            "1d": directional_return_calibration(
                dataset, "target", "target_return_1d"
            ),
            "5d": directional_return_calibration(
                dataset, "target_5d", "target_return_5d"
            ),
        }
        saved_bundle = {
            "model": model,
            "weekly_model": weekly_model,
            "model_name": ticker_model_name,
            "weekly_model_name": ticker_weekly_model_name,
            "model_horizon_sessions": horizon,
            "weekly_horizon_sessions": 5,
            "features": feature_columns,
            "sequence_features": sequence_feature_columns,
            "weekly_features": feature_columns,
            "weekly_sequence_features": sequence_feature_columns,
            "ticker": ticker,
            "benchmark": bundle.get("benchmark", "SPY"),
            "sector": sector_symbol,
            "volatility_index": bundle.get("volatility_index", "^VIX"),
            "is_mean_reverter": is_mean_reverter,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "lstm_parameters": model_kwargs if ticker_model_name == "lstm" else {},
            "weekly_lstm_parameters": (
                weekly_model_kwargs if ticker_weekly_model_name == "lstm" else {}
            ),
            "return_calibration": calibration,
        }
        ticker_artifact.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(saved_bundle, ticker_artifact)
        except Exception:
            app.logger.exception(
                "Could not persist ticker model artifact %s", ticker_artifact
            )
        trained_models[ticker] = (
            model,
            weekly_model,
            feature_columns,
            sequence_feature_columns,
            calibration,
            sector_symbol,
            ticker_model_name,
            ticker_weekly_model_name,
            is_mean_reverter,
        )
        trained_model_times[ticker] = datetime.now(timezone.utc)
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
            (
                model,
                weekly_model,
                feature_columns,
                sequence_feature_columns,
                calibration,
                sector_symbol,
                ticker_model_name,
                ticker_weekly_model_name,
                is_mean_reverter,
            ) = train_for_ticker(ticker)
            prices = download_prices(ticker, period="1y")
            benchmark = download_prices(bundle.get("benchmark", "SPY"), period="1y")
            sector = download_prices(sector_symbol, period="1y")
            volatility_index = download_prices(
                bundle.get("volatility_index", "^VIX"), period="1y"
            )
            context = aligned_context_features(benchmark, sector, volatility_index)
            news_df = download_ticker_news_sentiment(ticker)
            macro_sentiment_df = build_macro_sentiment_features(
                bundle.get("benchmark", "SPY"), sector_symbol
            )
            features = build_features(
                prices,
                context,
                include_target=False,
                news_df=news_df,
                macro_sentiment_df=macro_sentiment_df,
            )
            latest = features.iloc[-1]
            model_columns = (
                sequence_feature_columns
                if ticker_model_name == "lstm"
                else feature_columns
            )
            latest_values = latest[model_columns].to_frame().T
            weekly_model_columns = (
                sequence_feature_columns
                if ticker_weekly_model_name == "lstm"
                else feature_columns
            )
            weekly_latest_values = latest[weekly_model_columns].to_frame().T
            predicted_probability = float(predict_probability(model, latest_values)[0])
            model_horizon_sessions = (
                5
                if is_mean_reverter
                else 1
            )
            primary_calibration = calibration[
                "5d" if model_horizon_sessions == 5 else "1d"
            ]
            predicted_weekly_probability = float(
                predict_probability(weekly_model, weekly_latest_values)[0]
            )
            predicted_return = (
                predicted_probability * primary_calibration["up_return"]
                + (1 - predicted_probability) * primary_calibration["down_return"]
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
                "sector": sector_symbol,
                "model": ticker_model_name,
                "weekly_model": ticker_weekly_model_name,
                "regime": (
                    "Mean-Reversion"
                    if is_mean_reverter
                    else "Momentum-Trend"
                ),
                "model": ticker_model_name,
                "weekly_model": ticker_weekly_model_name,
                "model_horizon_sessions": model_horizon_sessions,
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
                "data_notice": "This is not a financial advice. Please consult a financial advisor before making investment decisions.",
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
