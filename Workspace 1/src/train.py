"""Train, compare, and walk-forward-evaluate market return models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import aligned_context_features, download_prices
from .evaluation import (
    MODEL_NAMES,
    classification_metrics,
    directional_return_calibration,
    model_factory,
    predict_probability,
    strategy_metrics,
    walk_forward_predict,
)
from .features import build_features, get_feature_columns, get_sequence_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a market-price analysis model")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL or MSFT")
    parser.add_argument("--start", default="2015-01-01", help="Historical data start date")
    parser.add_argument("--end", default=None, help="Historical data end date")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ETF for market-regime features")
    parser.add_argument("--sector", default="XLK", help="Sector ETF for sector features")
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

    prices = download_prices(args.ticker, start=args.start, end=args.end)
    benchmark = download_prices(args.benchmark, start=args.start, end=args.end)
    sector = download_prices(args.sector, start=args.start, end=args.end)
    volatility_index = download_prices(args.volatility_index, start=args.start, end=args.end)
    context = aligned_context_features(benchmark, sector, volatility_index)
    dataset = build_features(prices, context)
    feature_columns = get_feature_columns(dataset)
    sequence_feature_columns = get_sequence_feature_columns(dataset)

    split_index = int(len(dataset) * (1 - args.test_size))
    train, test = dataset.iloc[:split_index], dataset.iloc[split_index:]
    holdout_results: dict[str, dict[str, float]] = {}
    for model_name in MODEL_NAMES:
        model = model_factory(model_name)
        columns = sequence_feature_columns if model_name == "lstm" else feature_columns
        model.fit(train[columns], train["target"])
        predictions = pd.Series(predict_probability(model, test[columns]), index=test.index)
        holdout_results[model_name] = classification_metrics(test["target"], predictions)

    weekly_model = model_factory(args.model)
    weekly_columns = sequence_feature_columns if args.model == "lstm" else feature_columns
    weekly_model.fit(train[weekly_columns], train["target_5d"])
    weekly_predictions = pd.Series(
        predict_probability(weekly_model, test[weekly_columns]), index=test.index
    )
    weekly_holdout = classification_metrics(test["target_5d"], weekly_predictions)

    walk_forward = walk_forward_predict(
        dataset,
        args.model,
        initial_train_size=split_index,
        step=args.walk_forward_step,
        target_column="target_5d",
        return_column="target_return_5d",
    )
    selected_metrics = classification_metrics(walk_forward.actuals, walk_forward.predictions)
    selected_metrics.update(
        strategy_metrics(
            walk_forward.actual_returns,
            walk_forward.predictions,
            args.transaction_cost_bps,
            args.slippage_bps,
            args.max_position,
            long_allowed=dataset.loc[walk_forward.actuals.index, "long_allowed"],
        )
    )

    final_model = model_factory(args.model)
    final_columns = sequence_feature_columns if args.model == "lstm" else feature_columns
    final_model.fit(dataset[final_columns], dataset["target"])
    final_weekly_model = model_factory(args.model)
    final_weekly_model.fit(dataset[weekly_columns], dataset["target_5d"])
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
        "sector": args.sector.upper(),
        "volatility_index": args.volatility_index.upper(),
        "train_rows": len(train),
        "test_rows": len(test),
        "holdout_model_comparison": holdout_results,
        "weekly_holdout": weekly_holdout,
        "walk_forward": {
            "model": args.model,
            "step": args.walk_forward_step,
            **selected_metrics,
        },
        "costs": {
            "transaction_cost_bps": args.transaction_cost_bps,
            "slippage_bps": args.slippage_bps,
            "max_position": args.max_position,
        },
    }

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "weekly_model": final_weekly_model,
            "weekly_horizon_sessions": 5,
            "model_name": args.model,
            "features": feature_columns,
            "sequence_features": sequence_feature_columns,
            "ticker": args.ticker.upper(),
            "benchmark": args.benchmark.upper(),
            "sector": args.sector.upper(),
            "volatility_index": args.volatility_index.upper(),
            "return_calibration": return_calibration,
        },
        artifacts / "model.joblib",
    )
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
