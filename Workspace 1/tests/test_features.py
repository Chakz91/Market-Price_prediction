import pandas as pd

from src.features import BASE_FEATURE_COLUMNS, build_features


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
