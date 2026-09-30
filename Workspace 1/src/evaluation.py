"""Time-series evaluation and simple cost-aware strategy accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from .features import (
    MEAN_REVERSION_FEATURE_COLUMNS,
    get_feature_columns,
    get_sequence_feature_columns,
)

MODEL_NAMES = ("ridge", "gradient_boosting", "random_forest", "lstm", "xgboost")


@dataclass
class WalkForwardResult:
    predictions: pd.Series
    actuals: pd.Series
    actual_returns: pd.Series | None = None
    is_mean_reverter: pd.Series | None = None


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Small two-layer PyTorch LSTM classifier trained with binary cross-entropy."""

    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 40,
        batch_size: int = 128,
        learning_rate: float = 0.0001,
        random_state: int = 42,
        validation_fraction: float = 0.15,
        validation_gap: int = 1,
        patience: int = 6,
        min_delta: float = 0.0001,
        long_threshold: float = 0.70,
        short_threshold: float = 0.10,
        flip_penalty: float = 0.0,
    ) -> None:
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.validation_gap = validation_gap
        self.patience = patience
        self.min_delta = min_delta
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
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        class Network(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers=num_layers,
                    dropout=recurrent_dropout,
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
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if not 0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        if self.validation_gap < 0:
            raise ValueError("validation_gap cannot be negative")
        if self.patience < 1:
            raise ValueError("patience must be positive")
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

        validation_size = 0
        maximum_validation_size = len(windows) - self.validation_gap - 2
        if self.validation_fraction and maximum_validation_size >= 1:
            validation_size = min(
                max(1, int(len(windows) * self.validation_fraction)),
                maximum_validation_size,
            )
        training_size = len(windows) - validation_size - self.validation_gap
        validation_start = training_size + self.validation_gap
        validation_loss = None
        best_epoch = self.epochs

        if validation_size:
            training_row_count = self.lookback + training_size - 1
            self._fit_normalizer(values[:training_row_count])
            all_windows = self._normalize_windows(torch.from_numpy(windows))
            labels = torch.from_numpy(window_targets).float()
            inputs = all_windows[:training_size]
            validation_inputs = all_windows[validation_start:]
            validation_labels = labels[validation_start:]

            torch.manual_seed(self.random_state)
            self.network_ = self._network(values.shape[1])
            optimizer = torch.optim.Adam(
                self.network_.parameters(), lr=self.learning_rate
            )
            loss_function = torch.nn.BCELoss()
            best_loss = float("inf")
            best_epoch = 1
            epochs_without_improvement = 0
            self.optimization_steps_ = 0
            for epoch in range(1, self.epochs + 1):
                self._fit_epoch(inputs, labels[:training_size], optimizer, torch)
                self.network_.eval()
                with torch.no_grad():
                    probabilities = self.network_(validation_inputs)
                    current_loss = float(
                        loss_function(probabilities, validation_labels).item()
                    )
                if current_loss < best_loss - self.min_delta:
                    best_loss = current_loss
                    best_epoch = epoch
                    validation_loss = current_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        break

        self.best_epoch_ = best_epoch
        self.validation_log_loss_ = validation_loss
        self._fit_normalizer(values)
        inputs = self._normalize_windows(torch.from_numpy(windows))
        labels = torch.from_numpy(window_targets).float()
        torch.manual_seed(self.random_state)
        self.network_ = self._network(values.shape[1])
        self.optimization_steps_ = 0
        self._fit_epochs(inputs, labels, best_epoch, torch)

        self.network_state_ = {
            name: value.detach().clone()
            for name, value in self.network_.state_dict().items()
        }
        self.input_size_ = values.shape[1]
        del self.network_
        return self

    def _fit_normalizer(self, values: np.ndarray) -> None:
        self.feature_mean_ = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        feature_scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale_ = np.where(feature_scale < 1e-6, 1.0, feature_scale)

    def _fit_epochs(self, inputs, labels, epochs: int, torch) -> None:
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
        for _ in range(epochs):
            self._fit_epoch(inputs, labels, optimizer, torch)

    def _fit_epoch(self, inputs, labels, optimizer, torch) -> None:
        loss_function = torch.nn.BCELoss()
        self.network_.train()
        for start in range(0, len(inputs), self.batch_size):
            batch_inputs = inputs[start : start + self.batch_size]
            batch_labels = labels[start : start + self.batch_size]
            probabilities = self.network_(batch_inputs)
            loss = self._transaction_aware_loss(
                probabilities, batch_labels, loss_function
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network_.parameters(), max_norm=1.0)
            optimizer.step()
            self.optimization_steps_ += 1

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

    def _normalize_windows(self, windows):
        torch = self._torch()
        if not hasattr(self, "feature_mean_"):
            mean = windows.mean(dim=1, keepdim=True)
            scale = windows.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return (windows - mean) / scale
        mean = torch.as_tensor(self.feature_mean_, dtype=windows.dtype).view(1, 1, -1)
        scale = torch.as_tensor(self.feature_scale_, dtype=windows.dtype).view(1, 1, -1)
        return (windows - mean) / scale




def predict_probability(model, X) -> np.ndarray:
    """Return the probability of a positive next-price movement."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.shape[1] == 1:
            return np.full(len(X), float(model.classes_[0] >= 0.5))
        return probabilities[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def fit_classifier_or_constant(model, features, target):
    """Fit a classifier, falling back to its only observed class when necessary."""
    target = pd.Series(target)
    if target.empty:
        raise ValueError("Cannot fit a classifier with no training targets")
    if target.nunique() < 2:
        model = DummyClassifier(strategy="constant", constant=[target.iloc[0]])
    model.fit(features, target)
    return model


# Keep older joblib artifacts loadable after the regression model was replaced.
LSTMRegressor = LSTMClassifier


def model_factory(name: str, **kwargs: Any):
    if name not in {"lstm", "xgboost"} and kwargs:
        raise TypeError("Model parameters are only supported for lstm")
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError(
                "The XGBoost model requires xgboost. Install project requirements first."
            ) from error
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
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
        return LSTMClassifier(**kwargs)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def walk_forward_predict(
    dataset: pd.DataFrame,
    model_name: str,
    initial_train_size: int,
    step: int = 5,
    target_column: str = "target",
    return_column: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    sequence_columns: list[str] | None = None,
    feature_columns: list[str] | None = None,
    close_prices: pd.Series | None = None,
    regime_selector: Callable[[pd.Series], dict[str, Any]] | None = None,
    target_horizon: int = 0,
) -> WalkForwardResult:
    """Refit on available history, optionally selecting a point-in-time regime."""
    if initial_train_size < 1 or initial_train_size >= len(dataset):
        raise ValueError("initial_train_size must leave at least one test row")
    if step < 1:
        raise ValueError("step must be positive")
    if target_horizon < 0:
        raise ValueError("target_horizon cannot be negative")
    if (close_prices is None) != (regime_selector is None):
        raise ValueError("close_prices and regime_selector must be provided together")

    if target_column not in dataset.columns and target_column == "target":
        target_column = "target_return_1d"
    predictions: list[float] = []
    prediction_dates: list[pd.Timestamp] = []
    actual_values: list[int | float] = []
    actual_return_values: list[float] = []
    regime_values: list[bool] = []
    for start in range(initial_train_size, len(dataset), step):
        stop = min(start + step, len(dataset))
        test = dataset.iloc[start:stop]
        current_model_name = model_name
        if close_prices is not None and regime_selector is not None:
            cutoff = dataset.index[start - 1]
            history = close_prices.loc[:cutoff]
            regime_params = regime_selector(history)
            is_mean_reverter = bool(regime_params["is_mean_reverter"])
            if regime_params.get("force_xgboost", False):
                current_model_name = "xgboost"
            elif regime_params.get("force_linear", False):
                current_model_name = "ridge"
            target_for_window = "target_5d" if is_mean_reverter else "target"
            return_for_window = (
                "target_return_5d" if is_mean_reverter else "target_return_1d"
            )
            horizon = 5 if is_mean_reverter else 1
            train = dataset.iloc[: start - horizon]
            if current_model_name == "lstm":
                columns = list(
                    sequence_columns
                    if sequence_columns is not None
                    else get_sequence_feature_columns(dataset)
                )
                if is_mean_reverter:
                    if "return_1d" in columns and "smoothed_return_1d" in dataset:
                        columns[columns.index("return_1d")] = "smoothed_return_1d"
                    columns.extend(
                        column
                        for column in MEAN_REVERSION_FEATURE_COLUMNS
                        if column in dataset.columns and column not in columns
                    )
            else:
                columns = list(
                    feature_columns
                    if feature_columns is not None
                    else get_feature_columns(dataset)
                )
                if (
                    is_mean_reverter
                    and "return_1d" in columns
                    and "smoothed_return_1d" in dataset
                ):
                    columns[columns.index("return_1d")] = "smoothed_return_1d"
            current_model_kwargs = {
                key: value
                for key, value in regime_params.items()
                if key
                not in {"is_mean_reverter", "force_linear", "force_xgboost"}
            }
            regime_values.extend([is_mean_reverter] * len(test))
        else:
            is_mean_reverter = False
            target_for_window = target_column
            return_for_window = return_column
            train_stop = start - target_horizon
            if train_stop < 1:
                raise ValueError(
                    "initial_train_size must exceed target_horizon"
                )
            train = dataset.iloc[:train_stop]
            if current_model_name == "lstm":
                columns = (
                    sequence_columns
                    if sequence_columns is not None
                    else get_sequence_feature_columns(dataset)
                )
            else:
                columns = (
                    feature_columns
                    if feature_columns is not None
                    else get_feature_columns(dataset)
                )
            current_model_kwargs = model_kwargs or {}

        factory_kwargs = current_model_kwargs if current_model_name == "lstm" else {}
        model = clone(model_factory(current_model_name, **factory_kwargs))
        fit_target = train[target_for_window]
        model = fit_classifier_or_constant(model, train[columns], fit_target)
        if target_for_window in {"target", "target_5d"}:
            prediction_features = test[columns]
            if (
                close_prices is None
                and target_horizon
                and current_model_name == "lstm"
            ):
                prediction_features = dataset.iloc[train_stop:stop][columns]
                fold_predictions = predict_probability(model, prediction_features)
                predictions.extend(fold_predictions[-len(test):])
            else:
                predictions.extend(predict_probability(model, prediction_features))
        else:
            predictions.extend(model.predict(test[columns]))
        prediction_dates.extend(test.index)
        actual_values.extend(test[target_for_window].tolist())
        if return_for_window is not None:
            actual_return_values.extend(test[return_for_window].tolist())

    actuals = pd.Series(actual_values, index=prediction_dates, name="actual")
    actual_returns = (
        pd.Series(actual_return_values, index=prediction_dates, name="actual_return")
        if return_for_window is not None
        else None
    )
    return WalkForwardResult(
        predictions=pd.Series(predictions, index=prediction_dates, name="prediction"),
        actuals=actuals,
        actual_returns=actual_returns,
        is_mean_reverter=(
            pd.Series(regime_values, index=prediction_dates, name="is_mean_reverter")
            if close_prices is not None
            else None
        ),
    )


def classification_metrics(actuals: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    clipped = probabilities.clip(1e-7, 1 - 1e-7)
    actual_values = actuals.to_numpy(dtype=float)
    probability_values = clipped.to_numpy(dtype=float)
    bin_indices = np.minimum((probability_values * 10).astype(int), 9)
    calibration_error = sum(
        float(np.mean(bin_indices == bin_index))
        * abs(
            float(actual_values[bin_indices == bin_index].mean())
            - float(probability_values[bin_indices == bin_index].mean())
        )
        for bin_index in range(10)
        if np.any(bin_indices == bin_index)
    )
    return {
        "log_loss": float(log_loss(actuals, clipped, labels=[0, 1])),
        "brier_score": float(np.mean((probability_values - actual_values) ** 2)),
        "expected_calibration_error": float(calibration_error),
        "directional_accuracy": float(((clipped >= 0.5) == (actuals >= 0.5)).mean()),
    }


def compare_probability_models(
    features_by_model: dict[str, pd.DataFrame],
    target: pd.Series,
    candidates_by_model: dict[str, list[dict[str, Any]]],
    *,
    folds: int = 3,
    gap: int = 1,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Compare probability models on identical expanding chronological folds."""
    if not features_by_model:
        raise ValueError("at least one model feature set is required")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if gap < 0:
        raise ValueError("gap cannot be negative")
    if len(target) == 0 or any(len(features) != len(target) for features in features_by_model.values()):
        raise ValueError("all feature sets and target must have the same number of rows")
    reference_index = target.index
    if any(
        not features.index.equals(reference_index)
        for features in features_by_model.values()
    ):
        raise ValueError("all feature sets must have the target's chronological index")
    if set(features_by_model) != set(candidates_by_model):
        raise ValueError("feature sets and model candidates must have matching model names")
    if any(not candidates for candidates in candidates_by_model.values()):
        raise ValueError("each model needs at least one parameter candidate")

    maximum_lookback = max(
        (
            int(candidate.get("lookback", 20))
            for model_name, candidates in candidates_by_model.items()
            if model_name == "lstm"
            for candidate in candidates
        ),
        default=1,
    )
    initial_train_size = max(maximum_lookback + 10, int(len(target) * 0.55))
    if initial_train_size + gap + folds > len(target):
        raise ValueError("not enough rows for the requested chronological tuning folds")
    validation_indices = np.array_split(
        np.arange(initial_train_size, len(target)), folds
    )

    results = []
    for model_name, candidates in candidates_by_model.items():
        model_features = features_by_model[model_name]
        for candidate in candidates:
            actual_values: list[float] = []
            probability_values: list[float] = []
            completed_folds = 0
            for fold_indices in validation_indices:
                if not len(fold_indices):
                    continue
                start = int(fold_indices[0])
                stop = int(fold_indices[-1]) + 1
                train_stop = start - gap
                if train_stop <= int(candidate.get("lookback", 0)):
                    continue

                model = fit_classifier_or_constant(
                    model_factory(model_name, **candidate),
                    model_features.iloc[:train_stop],
                    target.iloc[:train_stop],
                )
                if model_name == "lstm":
                    prediction_inputs = model_features.iloc[train_stop:stop]
                    predictions = predict_probability(model, prediction_inputs)[
                        -(stop - start):
                    ]
                else:
                    predictions = predict_probability(
                        model, model_features.iloc[start:stop]
                    )
                probability_values.extend(predictions.tolist())
                actual_values.extend(target.iloc[start:stop].astype(float).tolist())
                completed_folds += 1

            if not completed_folds:
                raise ValueError(f"no usable chronological folds for {model_name}")
            metrics = classification_metrics(
                pd.Series(actual_values), pd.Series(probability_values)
            )
            results.append(
                {
                    "model": model_name,
                    "parameters": candidate.copy(),
                    "folds": completed_folds,
                    "validation_rows": len(actual_values),
                    **metrics,
                }
            )

    best_result = min(
        results, key=lambda result: (result["log_loss"], result["brier_score"])
    )
    return best_result["model"], best_result["parameters"].copy(), results


def tune_lstm_hyperparameters(
    features: pd.DataFrame,
    target: pd.Series,
    candidates: list[dict[str, Any]],
    *,
    baseline_features: pd.DataFrame | None = None,
    folds: int = 3,
    gap: int = 1,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Compatibility wrapper for the earlier Ridge-versus-LSTM tuner."""
    features_by_model = {}
    candidates_by_model = {}
    if baseline_features is not None:
        features_by_model["ridge"] = baseline_features
        candidates_by_model["ridge"] = [{}]
    features_by_model["lstm"] = features
    candidates_by_model["lstm"] = candidates
    return compare_probability_models(
        features_by_model,
        target,
        candidates_by_model,
        folds=folds,
        gap=gap,
    )


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
    invert_positions: bool | pd.Series = False,
    sentiment_volatility: pd.Series | None = None,
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
    if sentiment_volatility is None:
        upper = values.rolling(quantile_window, min_periods=quantile_window).quantile(
            long_quantile
        )
        lower = values.rolling(quantile_window, min_periods=quantile_window).quantile(
            short_quantile
        )
    else:
        volatility = pd.Series(
            sentiment_volatility, index=values.index, dtype=float
        ).replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
        volatility = volatility.clip(lower=0.0)
        multipliers = 1.0 + volatility
        upper_quantiles = (long_quantile * multipliers).clip(upper=0.95)
        lower_quantiles = (short_quantile / multipliers).clip(lower=0.05)
        upper = pd.Series(np.nan, index=values.index, dtype=float)
        lower = pd.Series(np.nan, index=values.index, dtype=float)
        for position in range(quantile_window - 1, len(values)):
            probability_window = values.iloc[
                position - quantile_window + 1 : position + 1
            ].to_numpy(dtype=float)
            upper.iloc[position] = np.quantile(
                probability_window, upper_quantiles.iloc[position]
            )
            lower.iloc[position] = np.quantile(
                probability_window, lower_quantiles.iloc[position]
            )
    positions = pd.Series(0.0, index=values.index)
    positions[values >= upper] = max_position
    positions[values <= lower] = -max_position
    if isinstance(invert_positions, pd.Series):
        invert_mask = pd.Series(invert_positions, index=values.index).fillna(False)
        positions.loc[invert_mask.astype(bool)] *= -1
    elif invert_positions:
        positions *= -1
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
    invert_positions: bool | pd.Series = False,
    sentiment_volatility: pd.Series | None = None,
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
            invert_positions=invert_positions,
            sentiment_volatility=sentiment_volatility,
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
