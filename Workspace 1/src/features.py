"""Feature engineering for next-session and one-week market-price prediction."""

from __future__ import annotations

import pandas as pd

BASE_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "volume_change_1d",
    "close_sma_ratio_10d",
    "close_sma_ratio_50d",
    "high_low_range",
]
CONTEXT_FEATURE_COLUMNS = [
    "benchmark_return_1d",
    "benchmark_volatility_20d",
    "sector_return_1d",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS


def get_feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [column for column in FEATURE_COLUMNS if column in dataset.columns]


def build_features(
    prices: pd.DataFrame,
    context: pd.DataFrame | None = None,
    include_target: bool = True,
) -> pd.DataFrame:
    """Build features using information available at the end of each session."""
    frame = prices.copy().sort_index()
    close = frame["Close"]
    volume = frame["Volume"]

    features = pd.DataFrame(index=frame.index)
    features["return_1d"] = close.pct_change(1, fill_method=None)
    features["return_5d"] = close.pct_change(5, fill_method=None)
    features["return_20d"] = close.pct_change(20, fill_method=None)
    features["volatility_20d"] = close.pct_change(fill_method=None).rolling(20).std()
    features["volume_change_1d"] = volume.pct_change(1, fill_method=None)
    features["close_sma_ratio_10d"] = close / close.rolling(10).mean() - 1
    features["close_sma_ratio_50d"] = close / close.rolling(50).mean() - 1
    features["high_low_range"] = (frame["High"] - frame["Low"]) / close

    if context is not None:
        features = features.join(context.reindex(features.index), how="left")

    if include_target:
        features["target_return_1d"] = close.shift(-1) / close - 1
        features["target_return_5d"] = close.shift(-5) / close - 1
    return features.replace([float("inf"), float("-inf")], pd.NA).dropna()
