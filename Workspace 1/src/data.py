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
) -> pd.DataFrame:
    """Create same-session context features for a next-session target."""
    context = pd.DataFrame(index=benchmark.index.union(sector.index).sort_values())
    context["benchmark_return_1d"] = benchmark["Close"].pct_change(fill_method=None)
    context["benchmark_volatility_20d"] = context["benchmark_return_1d"].rolling(20).std()
    context["sector_return_1d"] = sector["Close"].pct_change(fill_method=None)
    # Missing sessions remain missing; no future value is backfilled into an earlier date.
    return context.reindex(benchmark.index.union(sector.index)).sort_index()
