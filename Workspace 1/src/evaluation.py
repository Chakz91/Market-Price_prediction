"""Time-series evaluation and simple cost-aware strategy accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from .features import get_feature_columns, get_sequence_feature_columns

MODEL_NAMES = ("ridge", "gradient_boosting", "random_forest", "lstm")


@dataclass
class WalkForwardResult:
    predictions: pd.Series
    actuals: pd.Series
    actual_returns: pd.Series | None = None


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Small two-layer PyTorch LSTM classifier trained with binary cross-entropy."""

    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 40,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        random_state: int = 42,
        long_threshold: float = 0.70,
        short_threshold: float = 0.10,
        flip_penalty: float = 0.01,
    ) -> None:
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.flip_penalty = flip_penalty

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
        num_layers = self.num_layers
        dropout = self.dropout

        class Network(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_first=True,
                )
                self.dropout = torch.nn.Dropout(dropout)
                self.output = torch.nn.Linear(hidden_size, 1)

            def forward(self, values):
                sequence, _ = self.lstm(values)
                # Fixed: safely squeeze only the final feature dimension
                return torch.sigmoid(self.output(self.dropout(sequence[:, -1, :])).squeeze(-1))
                
                

        return Network()

    def fit(self, X, y):
        torch = self._torch()
        if self.lookback < 1:
            raise ValueError("lookback must be positive")
        if self.flip_penalty < 0:
            raise ValueError("flip_penalty cannot be negative")

        values = np.asarray(X, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32)
        if len(values) < self.lookback:
            raise ValueError("LSTM lookback is longer than the training data")

        self.history_ = values.copy()
        windows = np.stack(
            [
                values[index - self.lookback + 1 : index + 1]
                for index in range(self.lookback - 1, len(values))
            ]
        )
        window_targets = targets[self.lookback - 1 :]

        torch.manual_seed(self.random_state)
        self.network_ = self._network(values.shape[1])
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.BCELoss()
        inputs = torch.from_numpy(windows)
        labels = torch.from_numpy(window_targets).float()
        inputs = self._normalize_windows(inputs)
        self.network_.train()
        for _ in range(self.epochs):
            # Keep windows chronological so adjacent outputs represent adjacent days.
            probabilities = self.network_(inputs)
            loss = self._transaction_aware_loss(probabilities, labels, loss_function)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network_.parameters(), max_norm=1.0)
            optimizer.step()

        self.network_state_ = {
            name: value.detach().clone()
            for name, value in self.network_.state_dict().items()
        }
        self.input_size_ = values.shape[1]
        del self.network_
        return self

    def _transaction_aware_loss(self, probabilities, labels, classification_loss):
        torch = self._torch()
        loss = classification_loss(probabilities, labels)
        if len(probabilities) < 2:
            return loss
        previous_down = torch.relu(0.5 - probabilities[:-1])
        current_up = torch.relu(probabilities[1:] - 0.5)
        flip_loss = (previous_down * current_up).mean()
        return loss + self.flip_penalty * flip_loss

    def predict_proba(self, X):
        torch = self._torch()
        if not hasattr(self, "network_state_"):
            raise RuntimeError("The LSTM model must be fitted before prediction")

        values = np.asarray(X, dtype=np.float32)
        if len(values) == 1 and np.allclose(values[0], self.history_[-1]):
            combined = self.history_
            end_indices = [len(combined) - 1]
        else:
            combined = np.vstack([self.history_, values])
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
        windows = self._normalize_windows(torch.from_numpy(windows))
        network.eval()
        with torch.no_grad():
            probs = network(windows).numpy()
            # Returns an [N, 2] array containing [P(Down), P(Up)]
            return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        """Return long, short, or neutral signals using asymmetric cutoffs."""
        short_threshold = getattr(self, "short_threshold", 0.10)
        long_threshold = getattr(self, "long_threshold", 0.70)
        if not 0 <= short_threshold < long_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= short < long <= 1")

        probabilities = self.predict_proba(X)[:, 1]
        positions = np.zeros_like(probabilities, dtype=np.int32)
        positions[probabilities >= long_threshold] = 1
        positions[probabilities <= short_threshold] = -1
        return positions

    @staticmethod
    def _normalize_windows(windows):
        # Normalize each feature channel across its time steps independently.
        #channels_first = windows.transpose(1, 2)
        #mean = channels_first.mean(dim=2, keepdim=True)
        #standard_deviation = channels_first.std(
        #    dim=2,
        #    keepdim=True,
        #    unbiased=False,
        #).clamp_min(1e-6)
        #return ((channels_first - mean) / standard_deviation).transpose(1, 2)

        # Correctly normalises across the lookback dimension (dim=1) 
        # while keeping feature dimensions entirely isolated.
        mean = windows.mean(dim=1, keepdim=True)
        standard_deviation = windows.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (windows - mean) / standard_deviation




def predict_probability(model, X) -> np.ndarray:
    """Return the probability of a positive next-price movement."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.shape[1] == 1:
            return np.full(len(X), float(model.classes_[0] >= 0.5))
        return probabilities[:, 1]
    return np.asarray(model.predict(X), dtype=float)


# Keep older joblib artifacts loadable after the regression model was replaced.
LSTMRegressor = LSTMClassifier


def model_factory(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=2,
            loss="log_loss",
            random_state=42,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            criterion="log_loss",
            random_state=42,
            n_jobs=-1,
        )
    if name == "lstm":
        return LSTMClassifier()
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def walk_forward_predict(
    dataset: pd.DataFrame,
    model_name: str,
    initial_train_size: int,
    step: int = 5,
    target_column: str = "target",
    return_column: str | None = None,
) -> WalkForwardResult:
    """Refit on all available history, then predict the next step-sized block."""
    if initial_train_size < 1 or initial_train_size >= len(dataset):
        raise ValueError("initial_train_size must leave at least one test row")
    if step < 1:
        raise ValueError("step must be positive")

    if target_column not in dataset.columns and target_column == "target":
        target_column = "target_return_1d"
    predictions: list[float] = []
    prediction_dates: list[pd.Timestamp] = []
    for start in range(initial_train_size, len(dataset), step):
        stop = min(start + step, len(dataset))
        model = clone(model_factory(model_name))
        train = dataset.iloc[:start]
        test = dataset.iloc[start:stop]
        columns = (
            get_sequence_feature_columns(dataset)
            if model_name == "lstm"
            else get_feature_columns(dataset)
        )
        fit_target = train[target_column]
        if fit_target.nunique() < 2:
            model = DummyClassifier(strategy="constant", constant=[fit_target.iloc[0]])
        model.fit(train[columns], fit_target)
        if target_column in {"target", "target_5d"}:
            predictions.extend(predict_probability(model, test[columns]))
        else:
            predictions.extend(model.predict(test[columns]))
        prediction_dates.extend(test.index)

    actuals = dataset.loc[prediction_dates, target_column]
    actual_returns = None
    if return_column is not None:
        actual_returns = dataset.loc[prediction_dates, return_column]
    return WalkForwardResult(
        predictions=pd.Series(predictions, index=prediction_dates, name="prediction"),
        actuals=actuals.rename("actual"),
        actual_returns=actual_returns,
    )


def classification_metrics(actuals: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    clipped = probabilities.clip(1e-7, 1 - 1e-7)
    return {
        "log_loss": float(log_loss(actuals, clipped, labels=[0, 1])),
        "directional_accuracy": float(((clipped >= 0.5) == (actuals >= 0.5)).mean()),
    }


def directional_return_calibration(
    dataset: pd.DataFrame,
    direction_column: str,
    return_column: str,
) -> dict[str, float]:
    up_returns = dataset.loc[dataset[direction_column] >= 0.5, return_column]
    down_returns = dataset.loc[dataset[direction_column] < 0.5, return_column]
    return {
        "up_return": float(up_returns.mean()),
        "down_return": float(down_returns.mean()),
    }


def regression_metrics(actuals: pd.Series, predictions: pd.Series) -> dict[str, float]:
    errors = predictions - actuals
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "directional_accuracy": float(((predictions > 0) == (actuals > 0)).mean()),
    }


def dynamic_positions(
    probabilities: pd.Series,
    max_position: float = 1.0,
    quantile_window: int = 60,
    trade_quantile: float | None = None,
    long_allowed: pd.Series | None = None,
    long_quantile: float = 0.70,
    short_quantile: float = 0.10,
) -> pd.Series:
    """Use asymmetric rolling quantiles to generate long, short, or neutral positions."""
    if quantile_window < 2:
        raise ValueError("quantile_window must be at least 2")
    if trade_quantile is not None:
        if not 0 < trade_quantile < 0.5:
            raise ValueError("trade_quantile must be between 0 and 0.5")
        long_quantile = 1 - trade_quantile
        short_quantile = trade_quantile
    if not 0 <= short_quantile < long_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= short < long <= 1")

    values = pd.Series(probabilities, dtype=float)
    upper = values.rolling(quantile_window, min_periods=quantile_window).quantile(
        long_quantile
    )
    lower = values.rolling(quantile_window, min_periods=quantile_window).quantile(
        short_quantile
    )
    positions = pd.Series(0.0, index=values.index)
    positions[values >= upper] = max_position
    positions[values <= lower] = -max_position
    if long_allowed is not None:
        allowed = pd.Series(long_allowed, index=values.index).fillna(False).astype(bool)
        positions[(positions > 0) & ~allowed] = 0.0
    return positions


def strategy_metrics(
    actuals: pd.Series,
    predictions: pd.Series,
    transaction_cost_bps: float,
    slippage_bps: float,
    max_position: float,
    long_allowed: pd.Series | None = None,
    quantile_window: int = 60,
    trade_quantile: float | None = None,
    long_quantile: float = 0.70,
    short_quantile: float = 0.10,
) -> dict[str, float]:
    """Report a quantile-triggered long/short strategy after costs."""
    if min(transaction_cost_bps, slippage_bps) < 0:
        raise ValueError("costs cannot be negative")
    if not 0 < max_position <= 1:
        raise ValueError("max_position must be between 0 and 1")
    if quantile_window < 2:
        raise ValueError("quantile_window must be at least 2")
    if trade_quantile is not None:
        if not 0 < trade_quantile < 0.5:
            raise ValueError("trade_quantile must be between 0 and 0.5")
        long_quantile = 1 - trade_quantile
        short_quantile = trade_quantile
    if not 0 <= short_quantile < long_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= short < long <= 1")

    values = pd.Series(predictions, index=predictions.index, dtype=float)
    if ((values < 0) | (values > 1)).any():
        # Compatibility path for callers that still provide signed signals.
        positions = pd.Series(
            np.where(values >= 0, max_position, -max_position), index=values.index
        )
    else:
        positions = dynamic_positions(
            values,
            max_position=max_position,
            quantile_window=quantile_window,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            long_allowed=long_allowed,
        )

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
