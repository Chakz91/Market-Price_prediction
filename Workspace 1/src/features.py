"""Feature engineering for next-session and one-week market-price prediction."""

from __future__ import annotations

import pandas as pd

_FINBERT_COMPONENTS = None

BASE_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "volume_change_1d",
    "close_sma_ratio_10d",
    "close_sma_ratio_50d",
    "high_low_range",
    "macd",
    "macd_signal",
    "rsi_14",
    "bollinger_percent_b",
    "bollinger_bandwidth",
]
CONTEXT_FEATURE_COLUMNS = [
    "benchmark_return_1d",
    "benchmark_volatility_20d",
    "sector_return_1d",
    "vix_level",
    "vix_return_1d",
    "vix_sma_20d",
    "news_sentiment_ema",
    "macro_spy_sentiment_ema",
    "macro_sector_sentiment_ema",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS
SEQUENCE_FEATURE_COLUMNS = [
    "return_1d",
    "volume_change_1d",
    "high_low_range",
    "vix_return_1d",
    "benchmark_return_1d",
    "news_sentiment_ema",
    "macro_spy_sentiment_ema",
    "macro_sector_sentiment_ema",
]
MEAN_REVERSION_FEATURE_COLUMNS = [
    "rsi_14",
    "bollinger_percent_b",
    "bollinger_bandwidth",
]


def calculate_finbert_score(text: str) -> float:
    """Return FinBERT positive-minus-negative probability for one text document."""
    global _FINBERT_COMPONENTS
    if not isinstance(text, str) or not text.strip():
        return 0.0
    if _FINBERT_COMPONENTS is None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "FinBERT sentiment requires the project dependencies; "
                "install requirements.txt."
            ) from error

        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        _FINBERT_COMPONENTS = (torch, tokenizer, model)

    torch, tokenizer, model = _FINBERT_COMPONENTS
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
    labels = {
        int(index): label.casefold()
        for index, label in model.config.id2label.items()
    }
    positive = next(index for index, label in labels.items() if label == "positive")
    negative = next(index for index, label in labels.items() if label == "negative")
    return float(probabilities[positive] - probabilities[negative])


def _point_in_time_news_sentiment(
    news_df: pd.DataFrame | None, price_index: pd.Index
) -> pd.Series:
    sentiment = pd.Series(0.0, index=price_index, name="news_sentiment_ema")
    if news_df is None or news_df.empty:
        return sentiment
    if "Text" not in news_df.columns:
        raise ValueError("news_df must contain a 'Text' column")

    news_dates = news_df["Date"] if "Date" in news_df.columns else news_df.index
    normalized_news_dates = pd.DatetimeIndex(
        pd.to_datetime(news_dates, utc=True)
    ).tz_convert(None).normalize()
    sessions = pd.DatetimeIndex(price_index)
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)
    sessions = sessions.normalize()

    text_by_session: dict[int, list[str]] = {}
    for news_date, text in zip(normalized_news_dates, news_df["Text"]):
        session_position = sessions.searchsorted(news_date, side="left")
        if session_position < len(sessions) and isinstance(text, str) and text.strip():
            text_by_session.setdefault(session_position, []).append(text)

    daily_scores = pd.Series(0.0, index=price_index, name="news_sentiment")
    for position, texts in text_by_session.items():
        daily_scores.iloc[position] = calculate_finbert_score(" | ".join(texts))
    return daily_scores.ewm(span=3, adjust=False).mean().shift(1).fillna(0.0).rename(
        "news_sentiment_ema"
    )


def _point_in_time_scored_sentiment(
    sentiment_by_date: pd.Series | None,
    price_index: pd.Index,
    feature_name: str,
) -> pd.Series:
    daily_scores = pd.Series(0.0, index=price_index, name=feature_name)
    if sentiment_by_date is None or sentiment_by_date.empty:
        return daily_scores

    news_dates = pd.DatetimeIndex(
        pd.to_datetime(sentiment_by_date.index, utc=True)
    ).tz_convert(None).normalize()
    sessions = pd.DatetimeIndex(price_index)
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)
    sessions = sessions.normalize()

    for news_date, score in zip(news_dates, sentiment_by_date):
        if pd.isna(score):
            continue
        session_position = sessions.searchsorted(news_date, side="left")
        if session_position < len(sessions):
            daily_scores.iloc[session_position] = float(score)

    return daily_scores.ewm(span=3, adjust=False).mean().shift(1).fillna(0.0).rename(
        feature_name
    )


def winsorize_series(
    series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99
) -> pd.Series:
    """Clip values to percentile bounds learned from prior observations only."""
    if not 0 <= lower_q <= upper_q <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower_q <= upper_q <= 1")
    lower_bound = series.expanding(min_periods=20).quantile(lower_q).shift(1)
    upper_bound = series.expanding(min_periods=20).quantile(upper_q).shift(1)
    return series.clip(
        lower=lower_bound.fillna(series),
        upper=upper_bound.fillna(series),
    )


def get_feature_columns(
    dataset: pd.DataFrame, mean_reversion: bool = False
) -> list[str]:
    columns = list(FEATURE_COLUMNS)
    if mean_reversion:
        columns[columns.index("return_1d")] = "smoothed_return_1d"
    return [column for column in columns if column in dataset.columns]


def get_sequence_feature_columns(
    dataset: pd.DataFrame, mean_reversion: bool = False
) -> list[str]:
    columns = list(SEQUENCE_FEATURE_COLUMNS)
    if mean_reversion:
        columns[columns.index("return_1d")] = "smoothed_return_1d"
        columns.extend(MEAN_REVERSION_FEATURE_COLUMNS)
    return [column for column in columns if column in dataset.columns]


def build_features(
    prices: pd.DataFrame,
    context: pd.DataFrame | None = None,
    include_target: bool = True,
    news_df: pd.DataFrame | None = None,
    macro_sentiment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build features using information available at the end of each session."""
    frame = prices.copy().sort_index()
    close = frame["Close"]
    volume = frame["Volume"]

    features = pd.DataFrame(index=frame.index)
    features["return_1d"] = winsorize_series(
        close.pct_change(1, fill_method=None)
    )
    features["smoothed_return_1d"] = features["return_1d"].rolling(window=3).mean()
    features["return_5d"] = close.pct_change(5, fill_method=None)
    features["return_20d"] = close.pct_change(20, fill_method=None)
    features["volatility_20d"] = close.pct_change(fill_method=None).rolling(20).std()
    features["volume_change_1d"] = (
        volume.pct_change(1, fill_method=None)
        .replace([float("inf"), float("-inf")], pd.NA)
        .fillna(0)
    )
    features["news_sentiment_ema"] = _point_in_time_news_sentiment(
        news_df, features.index
    )
    for sentiment_column in (
        "macro_spy_sentiment",
        "macro_sector_sentiment",
    ):
        feature_column = f"{sentiment_column}_ema"
        features[feature_column] = _point_in_time_scored_sentiment(
            macro_sentiment_df[sentiment_column]
            if macro_sentiment_df is not None
            and sentiment_column in macro_sentiment_df
            else None,
            features.index,
            feature_column,
        )
    features["close_sma_ratio_10d"] = close / close.rolling(10).mean() - 1
    features["close_sma_ratio_50d"] = close / close.rolling(50).mean() - 1
    features["high_low_range"] = (frame["High"] - frame["Low"]) / close
    sma_200 = close.rolling(200).mean()
    # This regime flag is consumed by strategy evaluation, not model training.
    features["long_allowed"] = sma_200.isna() | (close >= sma_200)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    features["macd"] = macd
    features["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    change = close.diff()
    gains = change.clip(lower=0).rolling(14).mean()
    losses = -change.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + relative_strength))
    features["rsi_14"] = rsi.mask((losses == 0) & (gains > 0), 100).fillna(50)

    bollinger_middle = close.rolling(20).mean()
    bollinger_deviation = close.rolling(20).std()
    bollinger_upper = bollinger_middle + 2 * bollinger_deviation
    bollinger_lower = bollinger_middle - 2 * bollinger_deviation
    bollinger_width = bollinger_upper - bollinger_lower
    features["bollinger_percent_b"] = (
        (close - bollinger_lower) / bollinger_width.replace(0, float("nan"))
    )
    features["bollinger_bandwidth"] = (
        bollinger_width / bollinger_middle.replace(0, float("nan"))
    )

    if context is not None:
        features = features.join(context.reindex(features.index), how="left")

    if include_target:
        features["target_return_1d"] = close.shift(-1) / close - 1
        features["target_return_5d"] = close.shift(-5) / close - 1
        features["target"] = (features["return_1d"].shift(-1) > 0).astype(int)
        features["target_5d"] = (features["target_return_5d"] > 0).astype(int)
    return features.replace([float("inf"), float("-inf")], pd.NA).dropna()
