from pathlib import Path
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src import web
from src.train import ticker_model_artifact_path
from src.web import create_app


def test_health_endpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model = DummyRegressor(strategy="constant", constant=0.01)
    model.fit([[0] * 8], [0.01])
    joblib.dump({"model": model, "features": [], "ticker": "AAPL"}, model_path)

    client = create_app(model_path).test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_invalid_ticker_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model = DummyRegressor(strategy="constant", constant=0.01)
    model.fit([[0]], [0.01])
    joblib.dump({"model": model, "features": [], "ticker": "AAPL"}, model_path)

    response = create_app(model_path).test_client().get("/api/predict?ticker=not%20valid")

    assert response.status_code == 400


def test_health_does_not_require_ticker_training(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model = DummyRegressor(strategy="constant", constant=0.01)
    model.fit([[0]], [0.01])
    joblib.dump({"model": model, "features": [], "ticker": "AAPL"}, model_path)

    client = create_app(model_path).test_client()
    response = client.get("/health")

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("routing_flag", "expected_model"),
    [("force_linear", "ridge"), ("force_xgboost", "xgboost")],
)
def test_regime_override_selects_web_prediction_model(
    monkeypatch, tmp_path: Path, routing_flag: str, expected_model: str
) -> None:
    model_path = tmp_path / "model.joblib"
    model = DummyRegressor(strategy="constant", constant=0.01)
    model.fit([[0]], [0.01])
    joblib.dump(
        {
            "model": model,
            "model_name": "lstm",
            "ticker": "TEST",
            "sector": "XLRE",
        },
        model_path,
    )

    index = pd.bdate_range("2024-01-01", periods=12)
    prices = pd.DataFrame({"Close": np.linspace(100, 101, len(index))}, index=index)
    dataset = pd.DataFrame(
        {
            "feature": np.arange(len(index)),
            "sequence_feature": np.arange(len(index)),
            "target": [0, 1] * 6,
            "target_5d": [0, 1] * 6,
            "target_return_1d": [0.01] * len(index),
            "target_return_5d": [0.05] * len(index),
        },
        index=index,
    )
    model_names = []

    class PredictionModel:
        def fit(self, _features, _target):
            return self

        def predict_proba(self, features):
            return np.tile([0.4, 0.6], (len(features), 1))

    def recording_factory(name, **_kwargs):
        model_names.append(name)
        return PredictionModel()

    monkeypatch.setattr(web, "download_prices", lambda *_args, **_kwargs: prices)
    monkeypatch.setattr(
        web,
        "download_ticker_news_sentiment",
        lambda _ticker: pd.DataFrame(columns=["Text"]),
    )
    monkeypatch.setattr(
        web,
        "build_macro_sentiment_features",
        lambda *_args: pd.DataFrame(
            columns=["macro_spy_sentiment", "macro_sector_sentiment"]
        ),
    )
    monkeypatch.setattr(web, "aligned_context_features", lambda *_args: None)
    monkeypatch.setattr(
        web, "build_features", lambda *_args, **_kwargs: dataset.copy()
    )
    monkeypatch.setattr(web, "get_feature_columns", lambda *_args, **_kwargs: ["feature"])
    monkeypatch.setattr(
        web, "get_sequence_feature_columns", lambda *_args, **_kwargs: ["sequence_feature"]
    )
    monkeypatch.setattr(
        web,
        "get_regime_hyperparameters",
        lambda *_args: {
            "is_mean_reverter": True,
            routing_flag: True,
            "lookback": 5,
        },
    )
    monkeypatch.setattr(web, "model_factory", recording_factory)
    monkeypatch.setattr(
        web,
        "directional_return_calibration",
        lambda *_args: {"up_return": 0.01, "down_return": -0.01},
    )

    response = create_app(model_path).test_client().get("/api/predict?ticker=TEST")

    assert response.status_code == 200
    assert model_names == [expected_model, expected_model]
    assert response.json["regime"] == "Mean-Reversion"


@pytest.mark.parametrize(
    ("artifact_ticker", "artifact_age_days"),
    [("OTHER", 1), ("TEST", 61)],
)
def test_web_retrains_when_artifact_mismatches_or_expires(
    monkeypatch,
    tmp_path: Path,
    artifact_ticker: str,
    artifact_age_days: int,
) -> None:
    model_path = tmp_path / "model.joblib"
    model = DummyRegressor(strategy="constant", constant=0.01)
    model.fit([[0]], [0.01])
    joblib.dump(
        {
            "model": model,
            "model_name": "ridge",
            "weekly_model_name": "lstm",
            "weekly_lstm_parameters": {"lookback": 5, "validation_gap": 5},
            "ticker": "TEST",
            "sector": "XLRE",
        },
        model_path,
    )
    stale_artifact = ticker_model_artifact_path("TEST", model_path.parent)
    stale_artifact.parent.mkdir(parents=True)
    joblib.dump(
        {
            "ticker": artifact_ticker,
            "trained_at": (
                datetime.now(timezone.utc) - timedelta(days=artifact_age_days)
            ).isoformat(),
            "model": model,
            "weekly_model": model,
            "model_name": "ridge",
            "features": ["feature"],
            "sequence_features": ["sequence_feature"],
            "return_calibration": {},
        },
        stale_artifact,
    )

    index = pd.bdate_range("2024-01-01", periods=12)
    prices = pd.DataFrame({"Close": np.linspace(100, 101, len(index))}, index=index)
    dataset = pd.DataFrame(
        {
            "feature": np.arange(len(index)),
            "sequence_feature": np.arange(len(index)),
            "target": [0, 1] * 6,
            "target_5d": [0, 1] * 6,
            "target_return_1d": [0.01] * len(index),
            "target_return_5d": [0.05] * len(index),
        },
        index=index,
    )
    model_calls = []
    fitted_columns = []
    selection_calls = []

    class PredictionModel:
        def fit(self, features, _target):
            fitted_columns.append(features.columns.tolist())
            return self

        def predict_proba(self, features):
            return np.tile([0.4, 0.6], (len(features), 1))

    def recording_factory(name, **kwargs):
        model_calls.append((name, kwargs))
        return PredictionModel()

    def selecting_model(features_by_model, target, candidates, *, folds, gap):
        selection_calls.append((target.name, tuple(features_by_model), folds, gap))
        if target.name == "target":
            return "ridge", {}, []
        return "lstm", {"lookback": 5, "validation_gap": 5}, []

    monkeypatch.setattr(web, "download_prices", lambda *_args, **_kwargs: prices)
    monkeypatch.setattr(
        web,
        "download_ticker_news_sentiment",
        lambda _ticker: pd.DataFrame(columns=["Text"]),
    )
    monkeypatch.setattr(
        web,
        "build_macro_sentiment_features",
        lambda *_args: pd.DataFrame(
            columns=["macro_spy_sentiment", "macro_sector_sentiment"]
        ),
    )
    monkeypatch.setattr(web, "aligned_context_features", lambda *_args: None)
    monkeypatch.setattr(
        web, "build_features", lambda *_args, **_kwargs: dataset.copy()
    )
    monkeypatch.setattr(web, "get_feature_columns", lambda *_args, **_kwargs: ["feature"])
    monkeypatch.setattr(
        web,
        "get_sequence_feature_columns",
        lambda *_args, **_kwargs: ["sequence_feature"],
    )
    monkeypatch.setattr(
        web,
        "get_regime_hyperparameters",
        lambda *_args: {
            "is_mean_reverter": False,
            "lookback": 5,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 2,
            "learning_rate": 0.001,
        },
    )
    monkeypatch.setattr(web, "model_factory", recording_factory)
    monkeypatch.setattr(web, "compare_probability_models", selecting_model)
    monkeypatch.setattr(
        web,
        "directional_return_calibration",
        lambda *_args: {"up_return": 0.01, "down_return": -0.01},
    )

    response = create_app(model_path).test_client().get("/api/predict?ticker=TEST")

    assert response.status_code == 200
    assert response.json["model"] == "ridge"
    assert response.json["weekly_model"] == "lstm"
    assert [name for name, _kwargs in model_calls] == ["ridge", "lstm"]
    assert model_calls[1][1]["validation_gap"] == 5
    assert fitted_columns == [["feature"], ["sequence_feature"]]
    assert [call[0] for call in selection_calls] == ["target", "target_5d"]
    assert [call[3] for call in selection_calls] == [1, 5]


def test_web_reuses_only_matching_ticker_artifact(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    startup_model = DummyRegressor(strategy="constant", constant=0.01)
    startup_model.fit([[0]], [0.01])
    joblib.dump({"model": startup_model, "ticker": "SEED"}, model_path)

    daily_model = DummyRegressor(strategy="constant", constant=0.6)
    weekly_model = DummyRegressor(strategy="constant", constant=0.7)
    daily_model.fit([[0]], [0.6])
    weekly_model.fit([[0]], [0.7])
    artifact = ticker_model_artifact_path("TEST", model_path.parent)
    artifact.parent.mkdir(parents=True)
    joblib.dump(
        {
            "ticker": "TEST",
            "trained_at": (
                datetime.now(timezone.utc) - timedelta(days=59)
            ).isoformat(),
            "model": daily_model,
            "weekly_model": weekly_model,
            "model_name": "ridge",
            "weekly_model_name": "gradient_boosting",
            "features": ["daily_feature"],
            "sequence_features": ["daily_sequence_feature"],
            "weekly_features": ["weekly_feature"],
            "weekly_sequence_features": ["weekly_sequence_feature"],
            "sector": "XLK",
            "is_mean_reverter": False,
            "return_calibration": {
                "1d": {"up_return": 0.01, "down_return": -0.01},
                "5d": {"up_return": 0.05, "down_return": -0.05},
            },
        },
        artifact,
    )

    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=index)
    feature_data = pd.DataFrame(
        {
            "daily_feature": [0.1, 0.2, 0.3],
            "daily_sequence_feature": [0.2, 0.3, 0.4],
            "weekly_feature": [0.3, 0.4, 0.5],
            "weekly_sequence_feature": [0.4, 0.5, 0.6],
        },
        index=index,
    )

    def inference_prices(_ticker, **kwargs):
        assert "start" not in kwargs
        return prices

    monkeypatch.setattr(web, "download_prices", inference_prices)
    monkeypatch.setattr(web, "get_sector_benchmark", lambda _ticker: pytest.fail("unexpected sector lookup"))
    monkeypatch.setattr(web, "get_regime_hyperparameters", lambda *_args: pytest.fail("unexpected training"))
    monkeypatch.setattr(web, "compare_probability_models", lambda *_args, **_kwargs: pytest.fail("unexpected CV"))
    monkeypatch.setattr(web, "model_factory", lambda *_args, **_kwargs: pytest.fail("unexpected refit"))
    monkeypatch.setattr(web, "aligned_context_features", lambda *_args: None)
    monkeypatch.setattr(
        web,
        "download_ticker_news_sentiment",
        lambda _ticker: pd.DataFrame(columns=["Text"]),
    )
    monkeypatch.setattr(web, "build_macro_sentiment_features", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(web, "build_features", lambda *_args, **_kwargs: feature_data)
    monkeypatch.setattr(web, "get_feature_columns", lambda *_args, **_kwargs: ["daily_feature"])
    monkeypatch.setattr(
        web,
        "get_sequence_feature_columns",
        lambda *_args, **_kwargs: ["daily_sequence_feature"],
    )

    response = create_app(model_path).test_client().get("/api/predict?ticker=TEST")

    assert response.status_code == 200
    assert response.json["model"] == "ridge"
    assert response.json["weekly_model"] == "gradient_boosting"
