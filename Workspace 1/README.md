# Workspace 1: Market Price Analysis

A small, reproducible Python starter for researching next-session and one-week market-price movement. It downloads historical data, builds lagged/rolling features plus timestamp-aligned benchmark and sector context, compares regressors, evaluates the selected model with expanding walk-forward folds, and saves models for inference.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

FinBERT weights are fetched from Hugging Face the first time non-empty ticker news is scored and cached locally afterward.

## Train

```powershell
python -m src.train --ticker AAPL --start 2015-01-01
```

The default benchmark is `SPY`; the sector ETF is inferred from Yahoo Finance and can be overridden with `--sector`. Unknown sectors fall back to `SPY`. The command compares Ridge, gradient boosting, random forest, and LSTM models. It writes the latest model to `artifacts/model.joblib`, a reusable ticker-matched model bundle to `artifacts/models/TICKER.joblib`, and metrics to `artifacts/metrics.json`. Bundles include a UTC `trained_at` timestamp and independently selected models for the primary horizon and five trading sessions ahead. The metrics include:

- Holdout comparison for Ridge, gradient boosting, random forest, and LSTM, reporting log loss, Brier score, calibration error, and directional accuracy. Extreme-skew regimes route directly to XGBoost instead of the LSTM.
- Expanding chronological probability comparisons for all four models, separately for the primary and five-session targets. The lower pooled log-loss model is selected for each horizon; Brier score breaks ties and calibration error is diagnostic. The outer holdout is reserved for evaluation and reports all four models for both horizons.
- Expanding walk-forward error and directional accuracy for the selected primary-horizon model.
- Weekly walk-forward metrics for the selected weekly model and all four baselines on identical prediction dates, with a five-session label-maturity purge at each refit. Log loss and Brier score are primary; calibration error and directional accuracy are diagnostics.
- LSTM capacity is compared over three settings on the same folds. LSTM fitting uses a horizon-purged chronological validation tail for early stopping, mini-batches, and feature scaling fitted only on training history.
- A naive buy-and-hold comparison.
- Cost-aware strategy return, maximum drawdown, turnover, and total cost.

Strategy signals use the rolling 70th and 10th probability quantiles over the recent 60 outputs: the upper 30% opens a long (`1`), the lower 10% opens an active short (`-1`), and the middle remains neutral (`0`). Long signals are disabled when price is below its 200-day SMA. The LSTM's direct `predict()` method uses fixed probability cutoffs of `0.70` for long and `0.10` for short.

Example with explicit research assumptions:

```powershell
python -m src.train --ticker MSFT --benchmark SPY --sector XLK --model gradient_boosting --transaction-cost-bps 5 --slippage-bps 5 --max-position 0.5
```

## Multi-Sector Screening

Run the training pipeline for the configured 30-ticker basket, with a two-second pause between tickers to reduce Yahoo Finance throttling:

```powershell
python run_screening.py
```

The screening summary is saved to `artifacts/multi_sector_screening_results.csv`. It includes holdout trade accuracy for Ridge, gradient boosting, random forest, and LSTM, alongside the selected model's walk-forward trade accuracy. Training also writes a separate reusable model bundle for each ticker under `artifacts/models/`. The uranium tickers use `URA` and `AAL` uses `JETS` as sector ETF overrides.

## Predict

```powershell
python -m src.predict --ticker AAPL
```

The models classify whether the next session or next five trading sessions will close higher, using binary cross-entropy/log loss. Direction probabilities are calibrated with historical up/down returns to produce illustrative close estimates. Features include MACD, RSI, moving averages, volume, the SPY benchmark, sector context, VIX volatility context, and lagged FinBERT sentiment from the ticker, SPY, and its selected sector ETF. News is mapped to market sessions, smoothed with a three-session EMA, and shifted one session before use. Yahoo Finance only exposes a recent news feed, not complete historical headlines, so older backtest dates receive neutral sentiment; use a timestamped historical news provider for rigorous historical sentiment evaluation. The one-week horizon means five trading sessions. Context values are joined by session timestamp and are never backfilled from a future session. Treat results as research only.

## Web App

Train a model first, then start the Flask app from the project directory:

```powershell
python -m src.web
```

Open `http://127.0.0.1:5000` in a browser and enter a stock symbol. The JSON API is also available at:

```text
GET /api/predict?ticker=AAPL
GET /health
```

The app downloads the latest available Yahoo Finance history for the requested symbol and returns the last completed close plus next-session and one-week return and close estimates. Yahoo Finance data may be delayed; use a licensed streaming market-data provider for true real-time production predictions. Set `MODEL_PATH` and `PORT` environment variables when deploying the service.

When a ticker is entered in the web app, Flask first checks `artifacts/models/TICKER.joblib`, verifies the embedded ticker, and reuses the bundle only if its UTC `trained_at` timestamp is no more than 60 days old. Missing, invalid, mismatched, or expired bundles trigger fresh per-horizon chronological selection and fitting; Flask then attempts to save the refreshed bundle. In-memory models also expire after 60 days. The API response includes the selected `model` and `weekly_model` names. Use `python -m src.train --ticker SYMBOL` to inspect outer-holdout and walk-forward generalization metrics. Set `TRAIN_START` to change the historical training start date.

## Project Layout

- `src/data.py`: price/news downloads, sector ETF selection, and timestamp-aligned benchmark/sector features, including SPY/sector sentiment.
- `src/features.py`: lagged, rolling, context, and FinBERT sentiment feature construction.
- `src/evaluation.py`: model factories, walk-forward folds, and cost-aware metrics.
- `src/train.py`: comparison, evaluation, and artifact creation.
- `src/predict.py`: latest-session inference using the same saved feature configuration.
- `tests/`: feature and evaluation contract tests.
