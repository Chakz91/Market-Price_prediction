"""Market-data download and timestamp-aligned context features."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_prices(ticker: str, **kwargs) -> pd.DataFrame:
    prices = yf.download(ticker, auto_adjust=True, progress=False, **kwargs)
    if prices.empty:
        raise RuntimeError(f"No data returned for ticker {ticker!r}")
    if hasattr(prices.columns, "levels"):
        prices.columns = prices.columns.get_level_values(0)
    return prices.dropna(subset=["High", "Low", "Close", "Volume"])


def aligned_context_features(
    benchmark: pd.DataFrame,
    sector: pd.DataFrame,
    volatility_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create same-session benchmark, sector, and volatility context features."""
    indexes = [benchmark.index, sector.index]
    if volatility_index is not None:
        indexes.append(volatility_index.index)
    combined_index = indexes[0].union(indexes[1])
    for index in indexes[2:]:
        combined_index = combined_index.union(index)
    context = pd.DataFrame(index=combined_index.sort_values())
    context["benchmark_return_1d"] = benchmark["Close"].pct_change(fill_method=None)
    context["benchmark_volatility_20d"] = context["benchmark_return_1d"].rolling(20).std()
    context["sector_return_1d"] = sector["Close"].pct_change(fill_method=None)
    if volatility_index is not None:
        vix_close = volatility_index["Close"]
        context["vix_level"] = vix_close
        context["vix_return_1d"] = vix_close.pct_change(fill_method=None)
        context["vix_sma_20d"] = vix_close.rolling(20).mean()
    # Missing sessions remain missing; no future value is backfilled into an earlier date.
    return context.reindex(combined_index).sort_index()
