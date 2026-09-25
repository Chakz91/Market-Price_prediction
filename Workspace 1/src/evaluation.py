"""Time-series evaluation and simple cost-aware strategy accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import get_feature_columns

MODEL_NAMES = ("ridge", "gradient_boosting", "random_forest", "lstm")


@dataclass
class WalkForwardResult:
    predictions: pd.Series
    actuals: pd.Series


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """Small PyTorch LSTM with a scikit-learn compatible interface."""

    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 32,
        epochs: int = 40,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ) -> None:
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "The LSTM model requires PyTorch. Install project requirements first."
            ) from error
        return torch

    def _network(self, input_size: int):
        torch = self._torch()
        hidden_size = self.hidden_size

        class Network(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
                self.output = torch.nn.Linear(hidden_size, 1)

            def forward(self, values):
                sequence, _ = self.lstm(values)
                return self.output(sequence[:, -1, :]).squeeze(1)

        return Network()

    def fit(self, X, y):
        torch = self._torch()
        if self.lookback < 1:
            raise ValueError("lookback must be positive")

        values = np.asarray(X, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32)
        if len(values) < self.lookback:
            raise ValueError("LSTM lookback is longer than the training data")

        self.scaler_ = StandardScaler().fit(values)
        scaled_values = self.scaler_.transform(values).astype(np.float32)
        self.history_ = scaled_values.copy()
        windows = np.stack(
            [
                scaled_values[index - self.lookback + 1 : index + 1]
                for index in range(self.lookback - 1, len(values))
            ]
        )
        window_targets = targets[self.lookback - 1 :]

        torch.manual_seed(self.random_state)
        self.network_ = self._network(values.shape[1])
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.MSELoss()
        inputs = torch.from_numpy(windows)
        labels = torch.from_numpy(window_targets)
        self.network_.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(len(inputs))
            for start in range(0, len(inputs), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                loss = loss_function(self.network_(inputs[batch]), labels[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.network_state_ = self.network_.state_dict()
        self.input_size_ = values.shape[1]
        del self.network_
        return self

    def predict(self, X):
        torch = self._torch()
        if not hasattr(self, "network_state_"):
            raise RuntimeError("The LSTM model must be fitted before prediction")

        values = np.asarray(X, dtype=np.float32)
        scaled_values = self.scaler_.transform(values).astype(np.float32)
        if len(scaled_values) == 1 and np.allclose(scaled_values[0], self.history_[-1]):
            combined = self.history_
            end_indices = [len(combined) - 1]
        else:
            combined = np.vstack([self.history_, scaled_values])
            first_new_index = len(self.history_)
            end_indices = range(first_new_index, len(combined))
        windows = np.stack(
            [
                combined[index - self.lookback + 1 : index + 1]
                for index in end_indices
            ]
        )

        network = self._network(self.input_size_)
        network.load_state_dict(self.network_state_)
        network.eval()
        with torch.no_grad():
            return network(torch.from_numpy(windows)).numpy()


def model_factory(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=2,
            random_state=42,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    if name == "lstm":
        return LSTMRegressor()
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def walk_forward_predict(
    dataset: pd.DataFrame,
    model_name: str,
    initial_train_size: int,
    step: int = 5,
) -> WalkForwardResult:
    """Refit on all available history, then predict the next step-sized block."""
    if initial_train_size < 1 or initial_train_size >= len(dataset):
        raise ValueError("initial_train_size must leave at least one test row")
    if step < 1:
        raise ValueError("step must be positive")

    predictions: list[float] = []
    prediction_dates: list[pd.Timestamp] = []
    for start in range(initial_train_size, len(dataset), step):
        stop = min(start + step, len(dataset))
        model = clone(model_factory(model_name))
        train = dataset.iloc[:start]
        test = dataset.iloc[start:stop]
        columns = get_feature_columns(dataset)
        model.fit(train[columns], train["target_return_1d"])
        predictions.extend(model.predict(test[columns]))
        prediction_dates.extend(test.index)

    actuals = dataset.loc[prediction_dates, "target_return_1d"]
    return WalkForwardResult(
        predictions=pd.Series(predictions, index=prediction_dates, name="prediction"),
        actuals=actuals.rename("actual"),
    )


def regression_metrics(actuals: pd.Series, predictions: pd.Series) -> dict[str, float]:
    errors = predictions - actuals
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "directional_accuracy": float(((predictions > 0) == (actuals > 0)).mean()),
    }


def strategy_metrics(
    actuals: pd.Series,
    predictions: pd.Series,
    transaction_cost_bps: float,
    slippage_bps: float,
    max_position: float,
) -> dict[str, float]:
    """Report a fixed-sign strategy after costs and a buy-and-hold benchmark."""
    if min(transaction_cost_bps, slippage_bps) < 0:
        raise ValueError("costs cannot be negative")
    if not 0 < max_position <= 1:
        raise ValueError("max_position must be between 0 and 1")

    positions = pd.Series(np.where(predictions >= 0, max_position, -max_position), index=predictions.index)
    turnover = positions.diff().abs().fillna(positions.abs())
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    gross_returns = positions * actuals
    net_returns = gross_returns - turnover * cost_rate
    benchmark_returns = actuals

    def compound(returns: pd.Series) -> float:
        return float((1 + returns).prod() - 1)

    drawdown_curve = (1 + net_returns).cumprod()
    drawdown = drawdown_curve / drawdown_curve.cummax() - 1
    return {
        "strategy_gross_return": compound(gross_returns),
        "strategy_total_return": compound(net_returns),
        "buy_and_hold_total_return": compound(benchmark_returns),
        "strategy_max_drawdown": float(drawdown.min()),
        "average_position": float(positions.abs().mean()),
        "total_turnover": float(turnover.sum()),
        "total_cost": float((turnover * cost_rate).sum()),
    }
