"""Market-data download and timestamp-aligned context features."""

from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf


def get_sector_benchmark(ticker_symbol: str) -> str:
    """Return a sector ETF for a ticker, or SPY when its sector is unknown."""
    ticker_symbol = ticker_symbol.strip().upper()
    niche_overrides = {
        "UEC": "URA",
        "CCJ": "URA",
        "AAL": "JETS",
        "DAL": "JETS",
    }
    if ticker_symbol in niche_overrides:
        return niche_overrides[ticker_symbol]

    sector_mapping = {
        "technology": "XLK",
        "information technology": "XLK",
        "financial services": "XLF",
        "financials": "XLF",
        "healthcare": "XLV",
        "health care": "XLV",
        "consumer cyclical": "XLY",
        "consumer discretionary": "XLY",
        "basic materials": "XLB",
        "materials": "XLB",
        "energy": "XLE",
        "consumer defensive": "XLP",
        "consumer staples": "XLP",
        "industrials": "XLI",
        "communication services": "XLC",
        "utilities": "XLU",
        "real estate": "XLRE",
    }

    try:
        sector = yf.Ticker(ticker_symbol).info.get("sector")
    except Exception:
        warnings.warn(
            f"Could not fetch sector information for {ticker_symbol}; defaulting to SPY.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "SPY"

    normalized_sector = " ".join(sector.split()).casefold() if isinstance(sector, str) else ""
    return sector_mapping.get(normalized_sector, "SPY")


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
    """Build a fresh same-session context matrix from benchmark, sector, and volatility data."""
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

def generate_aligned_targets(df: pd.DataFrame, target_column: str = "return_1d") -> tuple[pd.DataFrame, pd.Series]:
    """
    Shifts the target directional variable backward by 1 step.
    Ensures today's features are explicitly matched with tomorrow's market direction.
    """
    clean_df = df.copy().dropna()
    
    # 1. Look ahead: Tomorrow's directional move becomes today's target label
    target = (clean_df[target_column].shift(-1) > 0).astype(int)
    
    # 2. Drop the final row because tomorrow's real-world data does not exist yet
    clean_df = clean_df.iloc[:-1]
    target = target.iloc[:-1]
    
    return clean_df, target
