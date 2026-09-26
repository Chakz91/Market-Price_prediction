import pandas as pd
import numpy as np

from src.evaluation import (
    LSTMClassifier,
    dynamic_positions,
    strategy_metrics,
    walk_forward_predict,
)
from src.features import FEATURE_COLUMNS


def test_walk_forward_never_predicts_before_initial_window() -> None:
    index = pd.date_range("2020-01-01", periods=30, freq="D")
    dataset = pd.DataFrame({column: 0.01 for column in FEATURE_COLUMNS}, index=index)
    dataset["target_return_1d"] = 0.01

    result = walk_forward_predict(dataset, "ridge", initial_train_size=10, step=3)

    assert result.predictions.index.min() == index[10]
    assert len(result.predictions) == 20


def test_strategy_metrics_charges_turnover() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    actuals = pd.Series([0.01, -0.01, 0.01], index=index)
    predictions = pd.Series([0.01, -0.01, 0.01], index=index)

    metrics = strategy_metrics(actuals, predictions, 10, 20, 1.0)

    assert metrics["total_cost"] > 0
    assert metrics["strategy_total_return"] < metrics["strategy_gross_return"]


def test_dynamic_positions_support_shorting_and_regime_filter() -> None:
    probabilities = pd.Series([0.45, 0.48, 0.52, 0.55, 0.10, 0.20, 0.90, 0.80])
    long_allowed = pd.Series([True, True, True, False, True, True, True, True])

    positions = dynamic_positions(
        probabilities,
        quantile_window=4,
        trade_quantile=0.25,
        long_allowed=long_allowed,
    )

    assert positions.iloc[4] == -1
    assert positions.iloc[6] == 1
    assert positions.iloc[3] == 0


def test_dynamic_positions_use_asymmetric_quantiles_by_default() -> None:
    probabilities = pd.Series(
        [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05, 0.45]
    )

    positions = dynamic_positions(probabilities, quantile_window=10)

    assert positions.iloc[9] == 1
    assert positions.iloc[10] == -1
    assert positions.iloc[11] == 0


def test_lstm_predict_uses_asymmetric_probability_thresholds() -> None:
    model = LSTMClassifier()
    probabilities = np.array([0.05, 0.10, 0.11, 0.69, 0.70, 0.95])
    model.predict_proba = lambda _values: np.column_stack((1 - probabilities, probabilities))

    positions = model.predict(pd.DataFrame(index=range(len(probabilities))))

    assert positions.tolist() == [-1, -1, 0, 0, 1, 1]
