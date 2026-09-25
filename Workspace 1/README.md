# Workspace 1: Market Price Analysis

A small, reproducible Python starter for researching next-session and one-week market-price movement. It downloads historical data, builds lagged/rolling features plus timestamp-aligned benchmark and sector context, compares regressors, evaluates the selected model with expanding walk-forward folds, and saves models for inference.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

```powershell
python -m src.train --ticker AAPL --start 2015-01-01
```

The default benchmark is `SPY`, the default sector ETF is `XLK`, and the default selected model is LSTM. The command writes `artifacts/model.joblib` and `artifacts/metrics.json`. The artifact contains separate models for the next session and five trading sessions ahead. The metrics include:

- Holdout comparison for Ridge, gradient boosting, and random forest.
- Expanding walk-forward error and directional accuracy.
- A naive buy-and-hold comparison.
- Cost-aware strategy return, maximum drawdown, turnover, and total cost.

Example with explicit research assumptions:

```powershell
python -m src.train --ticker MSFT --benchmark SPY --sector XLK --model gradient_boosting --transaction-cost-bps 5 --slippage-bps 5 --max-position 0.5
```

## Predict

```powershell
python -m src.predict --ticker AAPL
```

The models predict returns rather than raw prices, which makes evaluation more meaningful across different price levels. The one-week horizon means five trading sessions. Context values are joined by session timestamp and are never backfilled from a future session. Treat results as research only.

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

## Project Layout

- `src/data.py`: data download and timestamp-aligned benchmark/sector features.
- `src/features.py`: lagged, rolling, and context feature construction.
- `src/evaluation.py`: model factories, walk-forward folds, and cost-aware metrics.
- `src/train.py`: comparison, evaluation, and artifact creation.
- `src/predict.py`: latest-session inference using the same saved feature configuration.
- `tests/`: feature and evaluation contract tests.
