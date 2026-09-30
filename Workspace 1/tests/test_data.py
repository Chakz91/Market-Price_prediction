import pandas as pd
import pytest

from src import data, features


@pytest.mark.parametrize(
    ("sector", "expected"),
    [
        ("Technology", "XLK"),
        ("  Consumer   Cyclical ", "XLY"),
        ("Health Care", "XLV"),
        (None, "SPY"),
    ],
)
def test_get_sector_benchmark_maps_normalized_sectors(
    monkeypatch: pytest.MonkeyPatch, sector: str | None, expected: str
) -> None:
    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.info = {"sector": sector}

    monkeypatch.setattr(data.yf, "Ticker", Ticker)

    assert data.get_sector_benchmark("test") == expected


@pytest.mark.parametrize(("symbol", "expected"), [("uec", "URA"), ("AAL", "JETS")])
def test_get_sector_benchmark_uses_niche_overrides(symbol: str, expected: str) -> None:
    assert data.get_sector_benchmark(symbol) == expected


def test_get_sector_benchmark_falls_back_when_yahoo_finance_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_error(_symbol: str) -> object:
        raise RuntimeError("request failed")

    monkeypatch.setattr(data.yf, "Ticker", raise_error)

    with pytest.warns(RuntimeWarning, match="defaulting to SPY"):
        assert data.get_sector_benchmark("TEST") == "SPY"


def test_aligned_context_features_rebuilds_matrix_for_new_sector_data() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    benchmark = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=index)
    rising_sector = pd.DataFrame({"Close": [100.0, 110.0, 120.0]}, index=index)
    falling_sector = pd.DataFrame({"Close": [100.0, 90.0, 80.0]}, index=index)

    rising_context = data.aligned_context_features(benchmark, rising_sector)
    falling_context = data.aligned_context_features(benchmark, falling_sector)

    assert rising_context is not falling_context
    assert rising_context["sector_return_1d"].iloc[1] > 0
    assert falling_context["sector_return_1d"].iloc[1] < 0


def test_download_ticker_news_sentiment_groups_legacy_and_nested_news(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Ticker:
        news = [
            {
                "providerPublishTime": 1_704_067_200,
                "title": "Quarterly results",
                "summary": "Revenue increased.",
            },
            {
                "content": {
                    "pubDate": "2024-01-01T12:00:00Z",
                    "title": "Company outlook",
                    "summary": "Guidance raised.",
                }
            },
        ]

        def __init__(self, _symbol: str) -> None:
            pass

    monkeypatch.setattr(data.yf, "Ticker", Ticker)

    result = data.download_ticker_news_sentiment("TEST")

    assert result.index.tolist() == [pd.Timestamp("2024-01-01")]
    assert result.iloc[0]["Text"] == (
        "Quarterly results Revenue increased. | "
        "Company outlook Guidance raised."
    )


def test_download_ticker_news_sentiment_returns_empty_frame_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_error(_symbol: str) -> object:
        raise RuntimeError("news unavailable")

    monkeypatch.setattr(data.yf, "Ticker", raise_error)

    with pytest.warns(RuntimeWarning, match="neutral sentiment will be used"):
        result = data.download_ticker_news_sentiment("TEST")

    assert result.empty
    assert result.columns.tolist() == ["Text"]


def test_build_macro_sentiment_features_scores_selected_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    news_by_ticker = {
        "SPY": pd.DataFrame({"Text": ["broad market rally"]}, index=[dates[0]]),
        "XLK": pd.DataFrame({"Text": ["technology earnings"]}, index=[dates[1]]),
    }
    monkeypatch.setattr(
        data,
        "download_ticker_news_sentiment",
        lambda ticker: news_by_ticker[ticker],
    )
    monkeypatch.setattr(
        features,
        "calculate_finbert_score",
        lambda text: 0.7 if "market" in text else -0.4,
    )

    result = data.build_macro_sentiment_features("SPY", "XLK")

    assert result.loc[dates[0], "macro_spy_sentiment"] == pytest.approx(0.7)
    assert pd.isna(result.loc[dates[0], "macro_sector_sentiment"])
    assert result.loc[dates[1], "macro_sector_sentiment"] == pytest.approx(-0.4)