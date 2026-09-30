import sys

import numpy as np
import pandas as pd

from src import predict


def test_cli_prediction_uses_weekly_model_specific_features(monkeypatch, capsys) -> None:
    class FeatureCheckingModel:
        def __init__(self, expected_columns):
            self.expected_columns = expected_columns
            self.seen_columns = None

        def predict_proba(self, features):
            self.seen_columns = features.columns.tolist()
            return np.tile([0.4, 0.6], (len(features), 1))

    daily_model = FeatureCheckingModel(["daily_feature"])
    weekly_model = FeatureCheckingModel(["weekly_sequence_feature"])
    bundle = {
        "model": daily_model,
        "weekly_model": weekly_model,
        "model_name": "ridge",
        "weekly_model_name": "lstm",
        "features": ["daily_feature"],
        "sequence_features": ["daily_sequence_feature"],
        "weekly_features": ["weekly_feature"],
        "weekly_sequence_features": ["weekly_sequence_feature"],
        "ticker": "TEST",
        "return_calibration": {
            "1d": {"up_return": 0.01, "down_return": -0.01},
            "5d": {"up_return": 0.05, "down_return": -0.05},
        },
    }
    index = pd.date_range("2025-01-01", periods=2, freq="D")
    prices = pd.DataFrame({"Close": [100.0, 101.0]}, index=index)
    features = pd.DataFrame(
        {
            "daily_feature": [0.1, 0.2],
            "daily_sequence_feature": [0.3, 0.4],
            "weekly_feature": [0.5, 0.6],
            "weekly_sequence_feature": [0.7, 0.8],
        },
        index=index,
    )

    monkeypatch.setattr(sys, "argv", ["src.predict", "--ticker", "TEST"])
    monkeypatch.setattr(predict.joblib, "load", lambda _path: bundle)
    monkeypatch.setattr(
        predict, "download_prices", lambda *_args, **_kwargs: prices
    )
    monkeypatch.setattr(predict, "aligned_context_features", lambda *_args: None)
    monkeypatch.setattr(
        predict,
        "download_ticker_news_sentiment",
        lambda _ticker: pd.DataFrame(columns=["Text"]),
    )
    monkeypatch.setattr(
        predict,
        "build_macro_sentiment_features",
        lambda *_args: pd.DataFrame(),
    )
    monkeypatch.setattr(
        predict,
        "build_features",
        lambda *_args, **_kwargs: features,
    )

    predict.main()

    assert daily_model.seen_columns == ["daily_feature"]
    assert weekly_model.seen_columns == ["weekly_sequence_feature"]
    assert "Predicted 1-week return:" in capsys.readouterr().out