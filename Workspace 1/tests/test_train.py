import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.linear_model import LogisticRegression

from src import evaluation, train
from src.train import calculate_trend_strength, get_regime_hyperparameters


def test_regime_selector_uses_mean_reversion_settings_for_narrow_channel() -> None:
    close = pd.Series(
        np.tile(100 + np.sin(np.arange(120) / 3) * 2, 5)
    )
    params = get_regime_hyperparameters(close, "XLRE")

    assert calculate_trend_strength(close) is True
    assert params["is_mean_reverter"] is True
    assert params["lookback"] == 10
    assert params["hidden_size"] == 24
    assert params["num_layers"] == 1
    assert params["epochs"] == 25


def test_regime_selector_uses_momentum_settings_for_wide_channel() -> None:
    close = pd.Series(100 + np.sin(np.arange(600) / 3) * 20)
    params = get_regime_hyperparameters(close, "XLK")

    assert calculate_trend_strength(close) is False
    assert params["is_mean_reverter"] is False
    assert params["lookback"] == 20
    assert params["hidden_size"] == 32
    assert params["num_layers"] == 2
    assert params["epochs"] == 40


def test_short_history_uses_linear_baseline_without_changing_regime() -> None:
    close = pd.Series(100 + np.sin(np.arange(300) / 3) * 2)

    params = get_regime_hyperparameters(close, "XLRE")

    assert params["is_mean_reverter"] is calculate_trend_strength(close) is True
    assert params["force_linear"] is True
    assert params["lookback"] == 5
    assert params["hidden_size"] == 8
    assert params["num_layers"] == 1
    assert params["dropout"] == 0.3
    assert params["epochs"] == 10


@pytest.mark.parametrize("outlier_return", [1.76, -0.9])
def test_extreme_return_skew_routes_to_xgboost(
    outlier_return: float,
) -> None:
    daily_returns = pd.Series([0.001] * 600)
    daily_returns.iloc[300] = outlier_return
    close = 100 * (1 + daily_returns).cumprod()

    params = get_regime_hyperparameters(close, "XLK")

    assert abs(close.pct_change().skew()) > 3.0
    assert params["force_xgboost"] is True
    assert "force_linear" not in params
    assert params["is_mean_reverter"] is calculate_trend_strength(close)
    assert params["lookback"] == 5
    assert params["hidden_size"] == 8
    assert params["epochs"] == 10


def test_train_main_reports_when_feature_history_is_too_short(monkeypatch) -> None:
    index = pd.bdate_range("2025-01-01", periods=20)
    close = pd.Series(np.linspace(100, 101, len(index)), index=index)
    prices = pd.DataFrame(
        {
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )
    monkeypatch.setattr(
        train,
        "parse_args",
        lambda: argparse.Namespace(
            ticker="ANTH.PVT",
            start="2025-01-01",
            end=None,
            benchmark="SPY",
            sector=None,
            volatility_index="^VIX",
            model="lstm",
            test_size=0.2,
            walk_forward_step=5,
            transaction_cost_bps=5.0,
            slippage_bps=5.0,
            max_position=1.0,
        ),
    )
    monkeypatch.setattr(train, "get_sector_benchmark", lambda _ticker: "SPY")
    monkeypatch.setattr(train, "download_prices", lambda *_args, **_kwargs: prices)
    monkeypatch.setattr(
        train,
        "download_ticker_news_sentiment",
        lambda _ticker: pd.DataFrame(columns=["Text"]),
    )
    monkeypatch.setattr(
        train,
        "build_macro_sentiment_features",
        lambda *_args: pd.DataFrame(
            columns=["macro_spy_sentiment", "macro_sector_sentiment"]
        ),
    )

    with pytest.raises(ValueError, match="Not enough usable feature rows for ANTH.PVT"):
        train.main()


@pytest.mark.parametrize("direction", [1, -1])
def test_strong_trend_velocity_overrides_narrow_channel(direction: int) -> None:
    close = pd.Series(100 * np.exp(direction * 0.002 * np.arange(120)))
    rolling_width = ((close.rolling(60).max() - close.rolling(60).min()) / close)

    assert rolling_width.median() < 0.15
    assert calculate_trend_strength(close) is False
    params = get_regime_hyperparameters(close, "XLRE")
    assert params["is_mean_reverter"] is False
    assert params["force_linear"] is True


def test_train_main_routes_mean_reversion_parameters_and_writes_artifacts(
    monkeypatch, tmp_path
) -> None:
    index = pd.bdate_range("2020-01-01", periods=320)
    steps = np.arange(len(index))
    close = pd.Series(
        100 + steps * 0.05 + np.sin(steps / 3) * 2,
        index=index,
    )
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000 + steps,
        },
        index=index,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "parse_args", lambda: argparse.Namespace(
        ticker="TEST",
        start="2020-01-01",
        end=None,
        benchmark="SPY",
        sector=None,
        volatility_index="^VIX",
        model="lstm",
        test_size=0.3,
        walk_forward_step=30,
        transaction_cost_bps=5.0,
        slippage_bps=5.0,
        max_position=1.0,
    ))
    monkeypatch.setattr(train, "get_sector_benchmark", lambda _ticker: "XLRE")
    monkeypatch.setattr(train, "download_prices", lambda *_args, **_kwargs: prices)
    monkeypatch.setattr(
        train,
        "download_ticker_news_sentiment",
        lambda _ticker: pd.DataFrame(columns=["Text"]),
    )
    monkeypatch.setattr(
        train,
        "build_macro_sentiment_features",
        lambda *_args: pd.DataFrame(
            columns=["macro_spy_sentiment", "macro_sector_sentiment"]
        ),
    )

    factory_calls = []

    def configured_factory(name, **kwargs):
        factory_calls.append((name, kwargs))
        return LogisticRegression(max_iter=200)

    monkeypatch.setattr(train, "model_factory", configured_factory)
    monkeypatch.setattr(evaluation, "model_factory", configured_factory)

    train.main()

    metrics = json.loads((tmp_path / "artifacts" / "metrics.json").read_text())
    assert metrics["regime"] == "Mean-Reversion"
    assert metrics["is_mean_reverter"] is True
    assert metrics["model_horizon_sessions"] == 5
    assert "trade_directional_accuracy" in metrics["walk_forward"]
    expected_models = {"ridge", "gradient_boosting", "random_forest", "lstm"}
    assert set(metrics["weekly_holdout"]) == expected_models
    assert set(metrics["weekly_holdout"]["ridge"]) == {
        "log_loss",
        "brier_score",
        "expected_calibration_error",
        "directional_accuracy",
    }
    assert set(metrics["holdout_model_comparison"]) == expected_models
    assert set(metrics["holdout_model_comparison"]["ridge"]) == {
        "log_loss",
        "brier_score",
        "expected_calibration_error",
        "directional_accuracy",
        "trade_directional_accuracy",
        "active_trades",
    }
    bundle = joblib.load(tmp_path / "artifacts" / "model.joblib")
    assert bundle["model_name"] == "ridge"
    assert bundle["model_horizon_sessions"] == 5
    assert "news_sentiment_ema" in bundle["features"]
    assert "macro_spy_sentiment_ema" in bundle["features"]
    assert "macro_sector_sentiment_ema" in bundle["features"]
    np.testing.assert_allclose(bundle["model"].coef_, bundle["weekly_model"].coef_)
    assert factory_calls
    assert expected_models.issubset({name for name, _kwargs in factory_calls})


def test_train_selects_probability_models_for_both_horizons(
    monkeypatch, tmp_path
) -> None:
    index = pd.bdate_range("2020-01-01", periods=700)
    steps = np.arange(len(index))
    close = pd.Series(
        100 * np.exp(0.0005 * steps) * (1 + 0.06 * np.sin(steps / 4)),
        index=index,
    )
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000 + steps,
        },
        index=index,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        train,
        "parse_args",
        lambda: argparse.Namespace(
            ticker="TEST",
            start="2020-01-01",
            end=None,
            benchmark="SPY",
            sector=None,
            volatility_index="^VIX",
            model="lstm",
            test_size=0.25,
            walk_forward_step=80,
            transaction_cost_bps=5.0,
            slippage_bps=5.0,
            max_position=1.0,
        ),
    )
    monkeypatch.setattr(train, "get_sector_benchmark", lambda _ticker: "XLK")
    monkeypatch.setattr(
        train,
        "get_regime_hyperparameters",
        lambda *_args: {
            "lookback": 10,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 2,
            "learning_rate": 0.001,
            "is_mean_reverter": False,
        },
    )
    monkeypatch.setattr(train, "download_prices", lambda *_args, **_kwargs: prices)
    monkeypatch.setattr(
        train,
        "download_ticker_news_sentiment",
        lambda _ticker: pd.DataFrame(columns=["Text"]),
    )
    monkeypatch.setattr(
        train,
        "build_macro_sentiment_features",
        lambda *_args: pd.DataFrame(
            columns=["macro_spy_sentiment", "macro_sector_sentiment"]
        ),
    )

    def configured_factory(_name, **_kwargs):
        return LogisticRegression(max_iter=200)

    monkeypatch.setattr(train, "model_factory", configured_factory)
    monkeypatch.setattr(evaluation, "model_factory", configured_factory)

    train.main()

    metrics = json.loads((tmp_path / "artifacts" / "metrics.json").read_text())
    expected_models = {"ridge", "gradient_boosting", "random_forest", "lstm"}
    assert set(metrics["holdout_model_comparison"]) == expected_models
    assert set(metrics["weekly_holdout"]) == expected_models
    assert {
        result["model"]
        for result in metrics["probability_model_selection"]["candidates"]
    } == expected_models
    assert {
        result["model"]
        for result in metrics["weekly_model_selection"]["candidates"]
    } == expected_models
    assert {
        result["validation_rows"]
        for result in metrics["probability_model_selection"]["candidates"]
    } == {metrics["probability_model_selection"]["candidates"][0]["validation_rows"]}
    assert metrics["weekly_model_selection"]["selected_model"] == metrics[
        "weekly_holdout_selected_model"
    ]
    weekly_walk_forward = metrics["weekly_walk_forward"]
    assert weekly_walk_forward["selected_model"] == metrics[
        "weekly_model_selection"
    ]["selected_model"]
    assert weekly_walk_forward["purge_sessions"] == 5
    assert weekly_walk_forward["prediction_rows"] == metrics["test_rows"]
    assert set(weekly_walk_forward["models"]) == expected_models
    assert all(
        {"log_loss", "brier_score", "expected_calibration_error", "directional_accuracy"}
        == set(model_metrics)
        for model_metrics in weekly_walk_forward["models"].values()
    )

    bundle = joblib.load(tmp_path / "artifacts" / "model.joblib")
    assert bundle["model_name"] == metrics["probability_model_selection"]["selected_model"]
    assert bundle["weekly_model_name"] == metrics["weekly_model_selection"]["selected_model"]
    ticker_bundle = joblib.load(tmp_path / "artifacts" / "models" / "TEST.joblib")
    assert ticker_bundle["ticker"] == "TEST"
    trained_at = datetime.fromisoformat(ticker_bundle["trained_at"])
    assert trained_at.utcoffset() == timezone.utc.utcoffset(trained_at)
    assert ticker_bundle["model_name"] == bundle["model_name"]
    assert ticker_bundle["weekly_model_name"] == bundle["weekly_model_name"]