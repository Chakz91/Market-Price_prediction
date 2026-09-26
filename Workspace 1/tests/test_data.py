import pandas as pd
import pytest

from src import data


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