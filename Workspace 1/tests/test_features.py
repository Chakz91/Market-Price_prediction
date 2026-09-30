import pandas as pd
import pytest
import torch
from src import features

from src.features import (
    BASE_FEATURE_COLUMNS,
    get_feature_columns,
    MEAN_REVERSION_FEATURE_COLUMNS,
    build_features,
    get_sequence_feature_columns,
    winsorize_series,
)


def test_features_are_shifted_and_finite() -> None:
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    close = pd.Series(range(100, 180), index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )

    result = build_features(prices)

    assert set(BASE_FEATURE_COLUMNS).issubset(result.columns)
    assert "target_return_1d" in result.columns
    assert "target_return_5d" in result.columns
    assert set(result["target"].unique()).issubset({0, 1})
    assert set(result["target_5d"].unique()).issubset({0, 1})
    assert result.index.max() < index.max()
    assert result[BASE_FEATURE_COLUMNS + ["target_return_1d", "target_return_5d"]].notna().all().all()
    winsorized_return = winsorize_series(close.pct_change(fill_method=None))
    pd.testing.assert_series_equal(
        result["return_1d"],
        winsorized_return.reindex(result.index),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["smoothed_return_1d"],
        winsorized_return.rolling(window=3).mean().reindex(result.index),
        check_names=False,
    )

    standard_sequence = get_sequence_feature_columns(result)
    mean_reversion_sequence = get_sequence_feature_columns(result, mean_reversion=True)
    standard_features = get_feature_columns(result)
    mean_reversion_features = get_feature_columns(result, mean_reversion=True)
    assert "return_1d" in standard_sequence
    assert "return_1d" not in mean_reversion_sequence
    assert "smoothed_return_1d" in mean_reversion_sequence
    assert "return_1d" in standard_features
    assert "return_1d" not in mean_reversion_features
    assert "smoothed_return_1d" in mean_reversion_features


def test_zero_volume_does_not_discard_all_feature_rows() -> None:
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    close = pd.Series(range(100, 180), index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 0,
        },
        index=index,
    )

    result = build_features(prices)

    assert not result.empty
    assert result["volume_change_1d"].eq(0).all()


def test_daily_returns_are_winsorized_using_only_prior_observations() -> None:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    returns = pd.Series(
        [0.01 if day % 2 else -0.01 for day in range(120)], index=index
    )
    returns.iloc[80] = 1.76
    close = 100 * (1 + returns).cumprod()
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )

    result = build_features(prices, include_target=False)
    prefix_result = build_features(
        prices.iloc[:80], include_target=False
    )

    assert close.pct_change(fill_method=None).iloc[80] > 1.75
    assert result.loc[index[80], "return_1d"] < 0.02
    pd.testing.assert_series_equal(
        result.loc[:index[79], "return_1d"],
        prefix_result["return_1d"],
    )


def test_mean_reversion_sequence_includes_rsi_and_bollinger_features() -> None:
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    close = pd.Series(range(100, 180), index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )

    dataset = build_features(prices)
    sequence_columns = get_sequence_feature_columns(dataset, mean_reversion=True)

    assert set(MEAN_REVERSION_FEATURE_COLUMNS).issubset(sequence_columns)
    assert dataset[MEAN_REVERSION_FEATURE_COLUMNS].notna().all().all()


def test_news_sentiment_uses_ema_and_one_session_lag(monkeypatch) -> None:
    index = pd.bdate_range("2024-01-01", periods=100)
    close = pd.Series(range(100, 200), index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )
    news = pd.DataFrame({"Text": ["positive earnings"]}, index=[index[60]])
    monkeypatch.setattr(features, "calculate_finbert_score", lambda _text: 0.8)

    result = build_features(prices, include_target=False, news_df=news)

    assert result.loc[index[60], "news_sentiment_ema"] == 0.0
    assert result.loc[index[61], "news_sentiment_ema"] == 0.4
    assert result.loc[index[62], "news_sentiment_ema"] == 0.2
    assert "news_sentiment_ema" in get_feature_columns(result)
    assert "news_sentiment_ema" in get_sequence_feature_columns(result)


def test_finbert_score_uses_positive_minus_negative_probabilities(monkeypatch) -> None:
    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {}

    class Model:
        config = type(
            "Config",
            (),
            {"id2label": {0: "positive", 1: "negative", 2: "neutral"}},
        )()

        def eval(self):
            return self

        def __call__(self, **_kwargs):
            return type(
                "Output", (), {"logits": torch.tensor([[2.0, 1.0, 0.0]])}
            )()

    monkeypatch.setattr(
        features, "_FINBERT_COMPONENTS", (torch, Tokenizer(), Model())
    )

    score = features.calculate_finbert_score("earnings beat expectations")
    probabilities = torch.softmax(torch.tensor([2.0, 1.0, 0.0]), dim=-1)

    assert score == pytest.approx(float(probabilities[0] - probabilities[1]))


def test_macro_sentiment_is_ema_smoothed_lagged_and_used_by_models() -> None:
    index = pd.bdate_range("2024-01-01", periods=100)
    close = pd.Series(range(100, 200), index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )
    macro_sentiment = pd.DataFrame(
        {
            "macro_spy_sentiment": [0.6],
            "macro_sector_sentiment": [-0.4],
        },
        index=[index[60]],
    )

    result = build_features(
        prices, include_target=False, macro_sentiment_df=macro_sentiment
    )

    assert result.loc[index[60], "macro_spy_sentiment_ema"] == 0.0
    assert result.loc[index[61], "macro_spy_sentiment_ema"] == pytest.approx(0.3)
    assert result.loc[index[62], "macro_spy_sentiment_ema"] == pytest.approx(0.15)
    assert result.loc[index[60], "macro_sector_sentiment_ema"] == 0.0
    assert result.loc[index[61], "macro_sector_sentiment_ema"] == pytest.approx(-0.2)
    assert "macro_spy_sentiment_ema" in get_feature_columns(result)
    assert "macro_sector_sentiment_ema" in get_feature_columns(result)
    assert "macro_spy_sentiment_ema" in get_sequence_feature_columns(result)
    assert "macro_sector_sentiment_ema" in get_sequence_feature_columns(result)
