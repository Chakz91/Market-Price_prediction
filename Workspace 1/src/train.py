"""Train, compare, and walk-forward-evaluate market return models."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import (
    aligned_context_features,
    build_macro_sentiment_features,
    download_prices,
    download_ticker_news_sentiment,
    get_sector_benchmark,
)
from .evaluation import (
    MODEL_NAMES,
    classification_metrics,
    directional_return_calibration,
    fit_classifier_or_constant,
    model_factory,
    predict_probability,
    dynamic_positions,
    strategy_metrics,
    compare_probability_models,
    walk_forward_predict,
)
from .features import build_features, get_feature_columns, get_sequence_feature_columns


def ticker_model_artifact_path(
    ticker: str, artifacts_dir: Path = Path("artifacts")
) -> Path:
    normalized_ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.=-]{0,9}", normalized_ticker):
        raise ValueError(f"Invalid ticker symbol {ticker!r}")
    return Path(artifacts_dir) / "models" / f"{normalized_ticker}.joblib"


def calculate_trend_strength(close_prices: pd.Series, window: int = 60) -> bool:
    """Return whether a narrow channel is not being driven by a strong trend."""
    if window < 2:
        raise ValueError("window must be at least 2")
    close = pd.to_numeric(pd.Series(close_prices), errors="coerce")
    close = close.replace([float("inf"), float("-inf")], float("nan")).dropna()
    close = close[close > 0]
    if len(close) < window:
        return False
    rolling_max = close.rolling(window=window).max()
    rolling_min = close.rolling(window=window).min()
    channel_width = (rolling_max - rolling_min) / close
    log_close = np.log(close.iloc[-window:].to_numpy(dtype=float))
    x = np.arange(len(log_close))
    slope = float(np.polyfit(x, log_close, 1)[0])
    is_strongly_trending = abs(slope) > 0.0015
    return bool(channel_width.median() < 0.15 and not is_strongly_trending)


def get_regime_hyperparameters(
    close_prices: pd.Series, sector_symbol: str
) -> dict[str, int | float | bool]:
    """Select model capacity from channel width and recent log-price velocity."""
    is_mean_reverter = calculate_trend_strength(close_prices)
    close = pd.to_numeric(pd.Series(close_prices), errors="coerce")
    close = close.replace([float("inf"), float("-inf")], float("nan")).dropna()
    close = close[close > 0]
    returns = close.pct_change(fill_method=None)
    returns = returns.replace([float("inf"), float("-inf")], float("nan")).dropna()
    historical_skew = returns.skew()
    if abs(historical_skew) > 3.0:
        print(
            f"Extreme Skew Detected ({historical_skew:.2f}). "
            "Routing to XGBoost."
        )
        return {
            "lookback": 5,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 10,
            "learning_rate": 0.0001,
            "is_mean_reverter": is_mean_reverter,
            "force_xgboost": True,
        }
    if len(close_prices) < 500:
        print(
            f"Short history detected ({len(close_prices)} rows). "
            "Forcing baseline linear mode."
        )
        return {
            "lookback": 5,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.3,
            "epochs": 10,
            "learning_rate": 0.0001,
            "is_mean_reverter": is_mean_reverter,
            "force_linear": True,
        }
    if is_mean_reverter:
        return {
            "lookback": 10,
            "hidden_size": 24,
            "num_layers": 1,
            "dropout": 0.4,
            "epochs": 25,
            "learning_rate": 0.0003,
            "is_mean_reverter": True,
        }
    return {
        "lookback": 20,
        "hidden_size": 32,
        "num_layers": 2,
        "dropout": 0.4,
        "epochs": 40,
        "learning_rate": 0.0001,
        "is_mean_reverter": False,
    }


def _lstm_parameter_candidates(base: dict[str, int | float]) -> list[dict[str, int | float]]:
    lookback = int(base.get("lookback", 20))
    hidden_size = int(base.get("hidden_size", 16))
    dropout = float(base.get("dropout", 0.0))
    candidates = [
        {**base, "batch_size": 128},
        {
            **base,
            "lookback": max(5, lookback // 2),
            "hidden_size": max(8, hidden_size // 2),
            "num_layers": 1,
            "dropout": min(0.2, dropout) if dropout else 0.2,
            "learning_rate": max(float(base.get("learning_rate", 0.0001)), 0.0003),
            "batch_size": 128,
        },
        {
            **base,
            "lookback": min(30, lookback + 10),
            "num_layers": 1,
            "dropout": min(0.2, dropout) if dropout else 0.2,
            "batch_size": 128,
        },
    ]
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        signature = tuple(sorted(candidate.items()))
        if signature not in seen:
            seen.add(signature)
            unique_candidates.append(candidate)
    return unique_candidates


def _probability_model_candidates(
    lstm_base: dict[str, int | float], horizon: int
) -> dict[str, list[dict[str, int | float]]]:
    lstm_parameters = {**lstm_base, "validation_gap": horizon}
    return {
        "ridge": [{}],
        "gradient_boosting": [{}],
        "random_forest": [{}],
        "lstm": _lstm_parameter_candidates(lstm_parameters),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a market-price analysis model")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL or MSFT")
    parser.add_argument("--start", default="2015-01-01", help="Historical data start date")
    parser.add_argument("--end", default=None, help="Historical data end date")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ETF for market-regime features")
    parser.add_argument("--sector", default=None, help="Sector ETF override for sector features")
    parser.add_argument("--volatility-index", default="^VIX", help="Volatility index for context features")
    parser.add_argument("--model", choices=MODEL_NAMES, default="lstm")
    parser.add_argument("--test-size", type=float, default=0.2, help="Chronological test fraction")
    parser.add_argument("--walk-forward-step", type=int, default=5, help="Rows predicted per expanding-window refit")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-position", type=float, default=1.0, help="Absolute position size from 0 to 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 0.5:
        raise ValueError("--test-size must be between 0 and 0.5")

    sector_symbol = args.sector or get_sector_benchmark(args.ticker)
    prices = download_prices(args.ticker, start=args.start, end=args.end)
    benchmark = download_prices(args.benchmark, start=args.start, end=args.end)
    sector = download_prices(sector_symbol, start=args.start, end=args.end)
    volatility_index = download_prices(args.volatility_index, start=args.start, end=args.end)
    context = aligned_context_features(benchmark, sector, volatility_index)
    news_df = download_ticker_news_sentiment(args.ticker)
    macro_sentiment_df = build_macro_sentiment_features(
        args.benchmark, sector_symbol
    )
    dataset = build_features(
        prices,
        context,
        news_df=news_df,
        macro_sentiment_df=macro_sentiment_df,
    )
    if len(dataset) < 2:
        raise ValueError(
            f"Not enough usable feature rows for {args.ticker}: "
            f"found {len(dataset)}, need at least 2 to split training and test data."
        )
    split_index = int(len(dataset) * (1 - args.test_size))
    train, test = dataset.iloc[:split_index], dataset.iloc[split_index:]
    holdout_regime_params = get_regime_hyperparameters(
        prices["Close"].loc[:dataset.index[split_index - 1]], sector_symbol
    )
    holdout_model_name = (
        "xgboost"
        if holdout_regime_params.get("force_xgboost", False)
        else "ridge"
        if holdout_regime_params.get("force_linear", False)
        else args.model
    )
    holdout_is_mean_reverter = bool(holdout_regime_params["is_mean_reverter"])
    holdout_target = "target_5d" if holdout_is_mean_reverter else "target"
    holdout_horizon = 5 if holdout_is_mean_reverter else 1
    holdout_feature_columns = get_feature_columns(
        dataset, mean_reversion=holdout_is_mean_reverter
    )
    holdout_sequence_columns = get_sequence_feature_columns(
        dataset, mean_reversion=holdout_is_mean_reverter
    )
    lstm_base_parameters = {
        key: value
        for key, value in holdout_regime_params.items()
        if key not in {"is_mean_reverter", "force_linear", "force_xgboost"}
    }
    holdout_train = train.iloc[: len(train) - holdout_horizon]
    if holdout_train.empty:
        raise ValueError("Not enough training rows for the selected target horizon")
    weekly_train = train.iloc[: len(train) - 5]
    if weekly_train.empty:
        raise ValueError("Not enough training rows for the one-week target horizon")

    forced_model_name = (
        "xgboost"
        if holdout_regime_params.get("force_xgboost", False)
        else "ridge"
        if holdout_regime_params.get("force_linear", False)
        else None
    )
    daily_model_selection = []
    weekly_model_selection = []
    if forced_model_name is None:
        daily_feature_sets = {
            "ridge": holdout_train[holdout_feature_columns],
            "gradient_boosting": holdout_train[holdout_feature_columns],
            "random_forest": holdout_train[holdout_feature_columns],
            "lstm": holdout_train[holdout_sequence_columns],
        }
        selected_probability_model_name, selected_probability_model_parameters, daily_model_selection = (
            compare_probability_models(
                daily_feature_sets,
                holdout_train[holdout_target],
                _probability_model_candidates(lstm_base_parameters, holdout_horizon),
                folds=3,
                gap=holdout_horizon,
            )
        )
        weekly_feature_sets = {
            "ridge": weekly_train[holdout_feature_columns],
            "gradient_boosting": weekly_train[holdout_feature_columns],
            "random_forest": weekly_train[holdout_feature_columns],
            "lstm": weekly_train[holdout_sequence_columns],
        }
        (
            selected_weekly_model_name,
            selected_weekly_model_parameters,
            weekly_model_selection,
        ) = compare_probability_models(
            weekly_feature_sets,
            weekly_train["target_5d"],
            _probability_model_candidates(lstm_base_parameters, 5),
            folds=3,
            gap=5,
        )
    else:
        selected_probability_model_name = forced_model_name
        selected_weekly_model_name = forced_model_name
        selected_probability_model_parameters = {}
        selected_weekly_model_parameters = {}

    holdout_model_name = selected_probability_model_name
    holdout_model_kwargs = (
        selected_probability_model_parameters
        if holdout_model_name == "lstm"
        else {}
    )
    holdout_results: dict[str, dict[str, float]] = {}
    comparison_model_names = ("ridge", "gradient_boosting", "random_forest", "lstm")
    if forced_model_name == "xgboost" or args.model == "xgboost":
        comparison_model_names += ("xgboost",)
    daily_lstm_result = min(
        (result for result in daily_model_selection if result["model"] == "lstm"),
        key=lambda result: (result["log_loss"], result["brier_score"]),
        default=None,
    )
    daily_lstm_holdout_parameters = (
        daily_lstm_result["parameters"]
        if daily_lstm_result
        else _probability_model_candidates(lstm_base_parameters, holdout_horizon)["lstm"][0]
    )
    holdout_sentiment_volatility = (
        dataset["macro_spy_sentiment_ema"]
        .rolling(window=20)
        .std()
        .fillna(0.0)
        .reindex(test.index)
    )
    holdout_long_allowed = (
        None if holdout_is_mean_reverter else test["long_allowed"]
    )
    for model_name in comparison_model_names:
        model = (
            model_factory(model_name, **daily_lstm_holdout_parameters)
            if model_name == "lstm"
            else model_factory(model_name)
        )
        columns = (
            holdout_sequence_columns
            if model_name == "lstm"
            else holdout_feature_columns
        )
        model = fit_classifier_or_constant(
            model, holdout_train[columns], holdout_train[holdout_target]
        )
        predictions = pd.Series(predict_probability(model, test[columns]), index=test.index)
        model_metrics = classification_metrics(
            test[holdout_target], predictions
        )
        positions = dynamic_positions(
            predictions,
            max_position=args.max_position,
            long_allowed=holdout_long_allowed,
            invert_positions=holdout_is_mean_reverter,
            sentiment_volatility=holdout_sentiment_volatility,
        )
        active_trades = positions != 0
        if active_trades.any():
            active_predictions = (positions.loc[active_trades] > 0).astype(int)
            trade_accuracy = classification_metrics(
                test.loc[active_trades, holdout_target], active_predictions
            )["directional_accuracy"]
        else:
            trade_accuracy = 0.0
        model_metrics["trade_directional_accuracy"] = trade_accuracy
        model_metrics["active_trades"] = int(active_trades.sum())
        holdout_results[model_name] = model_metrics

    model_kwargs = holdout_model_kwargs if holdout_model_name == "lstm" else None
    weekly_lstm_result = min(
        (result for result in weekly_model_selection if result["model"] == "lstm"),
        key=lambda result: (result["log_loss"], result["brier_score"]),
        default=None,
    )
    weekly_lstm_holdout_parameters = (
        weekly_lstm_result["parameters"]
        if weekly_lstm_result
        else _probability_model_candidates(lstm_base_parameters, 5)["lstm"][0]
    )
    weekly_holdout = {}
    for model_name in comparison_model_names:
        weekly_columns = (
            holdout_sequence_columns
            if model_name == "lstm"
            else holdout_feature_columns
        )
        weekly_model = model_factory(
            model_name,
            **(weekly_lstm_holdout_parameters if model_name == "lstm" else {}),
        )
        weekly_model = fit_classifier_or_constant(
            weekly_model, weekly_train[weekly_columns], weekly_train["target_5d"]
        )
        weekly_predictions = pd.Series(
            predict_probability(weekly_model, test[weekly_columns]), index=test.index
        )
        weekly_holdout[model_name] = classification_metrics(
            test["target_5d"], weekly_predictions
        )

    weekly_walk_forward_results = {}
    for model_name in comparison_model_names:
        weekly_columns = (
            holdout_sequence_columns
            if model_name == "lstm"
            else holdout_feature_columns
        )
        weekly_model_kwargs = (
            weekly_lstm_holdout_parameters if model_name == "lstm" else None
        )
        weekly_walk_forward_result = walk_forward_predict(
            dataset,
            model_name,
            initial_train_size=split_index,
            step=args.walk_forward_step,
            target_column="target_5d",
            return_column="target_return_5d",
            model_kwargs=weekly_model_kwargs,
            sequence_columns=holdout_sequence_columns,
            feature_columns=holdout_feature_columns,
            target_horizon=5,
        )
        weekly_walk_forward_results[model_name] = classification_metrics(
            weekly_walk_forward_result.actuals,
            weekly_walk_forward_result.predictions,
        )
        if model_name == comparison_model_names[0]:
            weekly_walk_forward_dates = weekly_walk_forward_result.predictions.index
        elif not weekly_walk_forward_result.predictions.index.equals(
            weekly_walk_forward_dates
        ):
            raise RuntimeError(
                "Weekly walk-forward models did not use identical prediction dates"
            )

    def select_walk_forward_regime(history: pd.Series) -> dict[str, int | float | bool]:
        params = get_regime_hyperparameters(history, sector_symbol)
        same_regime = bool(params["is_mean_reverter"]) == holdout_is_mean_reverter
        if (
            selected_probability_model_name == "lstm"
            and same_regime
            and not params.get("force_linear", False)
            and not params.get("force_xgboost", False)
        ):
            params.update(selected_probability_model_parameters)
        return params

    walk_forward = walk_forward_predict(
        dataset,
        holdout_model_name,
        initial_train_size=split_index,
        step=args.walk_forward_step,
        model_kwargs=model_kwargs,
        sequence_columns=get_sequence_feature_columns(dataset, mean_reversion=False),
        feature_columns=get_feature_columns(dataset, mean_reversion=False),
        close_prices=prices["Close"],
        regime_selector=select_walk_forward_regime,
    )

    long_allowed = dataset.loc[walk_forward.actuals.index, "long_allowed"]
    sentiment_volatility = (
        dataset["macro_spy_sentiment_ema"]
        .rolling(window=20)
        .std()
        .fillna(0.0)
        .reindex(walk_forward.predictions.index)
    )
    selected_metrics = classification_metrics(
        walk_forward.actuals, walk_forward.predictions
    )
    mean_reversion_regimes = walk_forward.is_mean_reverter.fillna(False)
    execution_long_allowed = long_allowed.where(~mean_reversion_regimes, True)
    trade_positions = dynamic_positions(
        walk_forward.predictions,
        max_position=1.0,
        long_allowed=execution_long_allowed,
        invert_positions=mean_reversion_regimes,
        sentiment_volatility=sentiment_volatility,
    )
    active_trades = trade_positions != 0
    if active_trades.any():
        active_actuals = walk_forward.actuals[active_trades]
        active_preds = (trade_positions[active_trades] > 0).astype(int)
        trade_specific_metrics = classification_metrics(active_actuals, active_preds)
        selected_metrics["trade_directional_accuracy"] = trade_specific_metrics[
            "directional_accuracy"
        ]
        print(
            f"Real Trading Directional Accuracy: "
            f"{trade_specific_metrics['directional_accuracy']:.4f}"
        )

    selected_metrics.setdefault("trade_directional_accuracy", 0.0)
    selected_metrics.update(
        strategy_metrics(
            walk_forward.actual_returns,
            walk_forward.predictions,
            args.transaction_cost_bps,
            args.slippage_bps,
            args.max_position,
            long_allowed=execution_long_allowed,
            invert_positions=mean_reversion_regimes,
            sentiment_volatility=sentiment_volatility,
        )
    )

    regime_params = get_regime_hyperparameters(prices["Close"], sector_symbol)
    is_mean_reverter = bool(regime_params["is_mean_reverter"])
    target_column = "target_5d" if is_mean_reverter else "target"
    model_horizon_sessions = 5 if is_mean_reverter else 1
    feature_columns = get_feature_columns(dataset, mean_reversion=is_mean_reverter)
    sequence_feature_columns = get_sequence_feature_columns(
        dataset, mean_reversion=is_mean_reverter
    )
    lstm_params = {
        key: value
        for key, value in regime_params.items()
        if key not in {"is_mean_reverter", "force_linear", "force_xgboost"}
    }
    final_model_name = (
        "xgboost"
        if regime_params.get("force_xgboost", False)
        else "ridge"
        if regime_params.get("force_linear", False)
        else selected_probability_model_name
        if (
            selected_probability_model_name is not None
            and is_mean_reverter == holdout_is_mean_reverter
        )
        else args.model
    )
    final_weekly_model_name = (
        "xgboost"
        if regime_params.get("force_xgboost", False)
        else "ridge"
        if regime_params.get("force_linear", False)
        else selected_weekly_model_name
        if (
            selected_weekly_model_name is not None
            and is_mean_reverter == holdout_is_mean_reverter
        )
        else final_model_name
    )
    mode = "Mean-Reversion" if is_mean_reverter else "Momentum-Trend"
    print(f"Sector resolved to: {sector_symbol} | Current mode: {mode}")
    final_model_kwargs = lstm_params if final_model_name == "lstm" else {}
    if (
        final_model_name == "lstm"
        and selected_probability_model_name == "lstm"
        and is_mean_reverter == holdout_is_mean_reverter
    ):
        final_model_kwargs = selected_probability_model_parameters
    final_model = model_factory(final_model_name, **final_model_kwargs)
    final_columns = (
        sequence_feature_columns if final_model_name == "lstm" else feature_columns
    )
    final_model = fit_classifier_or_constant(
        final_model, dataset[final_columns], dataset[target_column]
    )
    final_weekly_model_kwargs = lstm_params if final_weekly_model_name == "lstm" else {}
    if (
        final_weekly_model_name == "lstm"
        and selected_weekly_model_name == "lstm"
        and is_mean_reverter == holdout_is_mean_reverter
    ):
        final_weekly_model_kwargs = selected_weekly_model_parameters
    final_weekly_columns = (
        sequence_feature_columns
        if final_weekly_model_name == "lstm"
        else feature_columns
    )
    final_weekly_model = model_factory(
        final_weekly_model_name, **final_weekly_model_kwargs
    )
    final_weekly_model = fit_classifier_or_constant(
        final_weekly_model, dataset[final_weekly_columns], dataset["target_5d"]
    )
    return_calibration = {
        "1d": directional_return_calibration(
            dataset, "target", "target_return_1d"
        ),
        "5d": directional_return_calibration(
            dataset, "target_5d", "target_return_5d"
        ),
    }
    metrics = {
        "ticker": args.ticker.upper(),
        "benchmark": args.benchmark.upper(),
        "sector": sector_symbol.upper(),
        "regime": mode,
        "is_mean_reverter": is_mean_reverter,
        "model_horizon_sessions": model_horizon_sessions,
        "volatility_index": args.volatility_index.upper(),
        "train_rows": len(train),
        "test_rows": len(test),
        "holdout_model_comparison": holdout_results,
        "weekly_holdout": weekly_holdout,
        "weekly_holdout_selected_model": selected_weekly_model_name,
        "weekly_walk_forward": {
            "selected_model": selected_weekly_model_name,
            "step": args.walk_forward_step,
            "target_horizon_sessions": 5,
            "purge_sessions": 5,
            "prediction_rows": len(weekly_walk_forward_dates),
            "first_prediction_date": weekly_walk_forward_dates[0].date().isoformat(),
            "last_prediction_date": weekly_walk_forward_dates[-1].date().isoformat(),
            "models": weekly_walk_forward_results,
        },
        "walk_forward": {
            "model": final_model_name,
            "step": args.walk_forward_step,
            **selected_metrics,
        },
        "costs": {
            "transaction_cost_bps": args.transaction_cost_bps,
            "slippage_bps": args.slippage_bps,
            "max_position": args.max_position,
        },
    }
    if daily_model_selection:
        metrics["probability_model_selection"] = {
            "selection_metric": "pooled_log_loss",
            "folds": 3,
            "selected_model": selected_probability_model_name,
            "selected_parameters": selected_probability_model_parameters or {},
            "candidates": daily_model_selection,
        }
    if weekly_model_selection:
        metrics["weekly_model_selection"] = {
            "selection_metric": "pooled_log_loss",
            "folds": 3,
            "selected_model": selected_weekly_model_name,
            "selected_parameters": selected_weekly_model_parameters or {},
            "candidates": weekly_model_selection,
        }

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    model_bundle = {
        "model": final_model,
        "weekly_model": final_weekly_model,
        "weekly_model_name": final_weekly_model_name,
        "weekly_horizon_sessions": 5,
        "model_horizon_sessions": model_horizon_sessions,
        "model_name": final_model_name,
        "features": feature_columns,
        "sequence_features": sequence_feature_columns,
        "weekly_features": feature_columns,
        "weekly_sequence_features": sequence_feature_columns,
        "ticker": args.ticker.strip().upper(),
        "benchmark": args.benchmark.upper(),
        "sector": sector_symbol.upper(),
        "is_mean_reverter": is_mean_reverter,
        "lstm_parameters": final_model_kwargs if final_model_name == "lstm" else lstm_params,
        "weekly_lstm_parameters": (
            final_weekly_model_kwargs
            if final_weekly_model_name == "lstm"
            else lstm_params
        ),
        "volatility_index": args.volatility_index.upper(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "return_calibration": return_calibration,
    }
    ticker_artifact = ticker_model_artifact_path(args.ticker, artifacts)
    ticker_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, ticker_artifact)
    joblib.dump(model_bundle, artifacts / "model.joblib")
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
