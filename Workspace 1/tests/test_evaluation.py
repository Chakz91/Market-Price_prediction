import pandas as pd
import numpy as np
import pytest

from src.evaluation import (
    LSTMClassifier,
    classification_metrics,
    compare_probability_models,
    dynamic_positions,
    fit_classifier_or_constant,
    model_factory,
    strategy_metrics,
    tune_lstm_hyperparameters,
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


def test_walk_forward_forwards_model_parameters_and_sequence_columns(
    monkeypatch,
) -> None:
    from src import evaluation

    factory_calls = []

    def configured_factory(name, **kwargs):
        from sklearn.linear_model import LogisticRegression

        factory_calls.append((name, kwargs))
        return LogisticRegression()

    monkeypatch.setattr(evaluation, "model_factory", configured_factory)
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    dataset = pd.DataFrame(
        {
            "custom_sequence_feature": range(12),
            "target": [0, 1] * 6,
        },
        index=index,
    )

    result = walk_forward_predict(
        dataset,
        "lstm",
        initial_train_size=8,
        step=2,
        model_kwargs={"lookback": 5},
        sequence_columns=["custom_sequence_feature"],
    )

    assert len(result.predictions) == 4
    assert factory_calls
    assert all(name == "lstm" and kwargs == {"lookback": 5} for name, kwargs in factory_calls)


def test_walk_forward_fits_requested_target_column(monkeypatch) -> None:
    from src import evaluation

    fitted_targets = []

    class RecordingModel:
        def fit(self, _features, target):
            fitted_targets.append(target.copy())
            return self

        def predict_proba(self, features):
            return np.tile([0.5, 0.5], (len(features), 1))

    monkeypatch.setattr(
        evaluation,
        "model_factory",
        lambda _name, **_kwargs: RecordingModel(),
    )
    monkeypatch.setattr(evaluation, "clone", lambda model: model)
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    target = pd.Series([0, 1] * 6, index=index)
    target_5d = pd.Series([1, 1, 0, 0] * 3, index=index, name="target_5d")
    dataset = pd.DataFrame(
        {"feature": range(12), "target": target, "target_5d": target_5d},
        index=index,
    )

    result = walk_forward_predict(
        dataset,
        "ridge",
        initial_train_size=8,
        step=2,
        target_column="target_5d",
        feature_columns=["feature"],
    )

    assert len(fitted_targets) == 2
    pd.testing.assert_series_equal(fitted_targets[0], target_5d.iloc[:8])
    pd.testing.assert_series_equal(fitted_targets[1], target_5d.iloc[:10])
    pd.testing.assert_series_equal(
        result.actuals,
        target_5d.iloc[8:].rename("actual"),
        check_freq=False,
    )


@pytest.mark.parametrize("model_name", ["ridge", "lstm"])
def test_weekly_walk_forward_purges_five_session_labels(
    monkeypatch, model_name: str
) -> None:
    from src import evaluation

    fitted_targets = []
    prediction_indexes = []

    class RecordingModel:
        def fit(self, features, target):
            fitted_targets.append((features.index.copy(), target.copy()))
            return self

        def predict_proba(self, features):
            prediction_indexes.append(features.index.copy())
            return np.tile([0.45, 0.55], (len(features), 1))

    monkeypatch.setattr(
        evaluation,
        "model_factory",
        lambda _name, **_kwargs: RecordingModel(),
    )
    monkeypatch.setattr(evaluation, "clone", lambda model: model)
    index = pd.date_range("2023-01-01", periods=30, freq="D")
    target = pd.Series(np.arange(30) % 2, index=index, name="target_5d")
    dataset = pd.DataFrame(
        {"feature": np.arange(30), "target_5d": target}, index=index
    )

    result = walk_forward_predict(
        dataset,
        model_name,
        initial_train_size=12,
        step=4,
        target_column="target_5d",
        feature_columns=["feature"],
        sequence_columns=["feature"],
        model_kwargs={"lookback": 5} if model_name == "lstm" else None,
        target_horizon=5,
    )

    expected_starts = [12, 16, 20, 24, 28]
    assert [len(targets) for _features, targets in fitted_targets] == [
        start - 5 for start in expected_starts
    ]
    for (training_index, targets), start in zip(fitted_targets, expected_starts):
        pd.testing.assert_index_equal(training_index, index[: start - 5])
        pd.testing.assert_series_equal(targets, target.iloc[: start - 5])
    for prediction_index, start in zip(prediction_indexes, expected_starts):
        prediction_start = start - 5 if model_name == "lstm" else start
        pd.testing.assert_index_equal(
            prediction_index,
            index[prediction_start : start + 4],
        )
    pd.testing.assert_index_equal(result.predictions.index, index[12:])
    pd.testing.assert_index_equal(result.actuals.index, result.predictions.index)


def test_walk_forward_selects_regimes_and_mature_targets_per_window(monkeypatch) -> None:
    from src import evaluation

    fitted_targets = []
    fitted_features = []
    model_settings = []

    class RecordingModel:
        def fit(self, features, target):
            fitted_features.append(features.copy())
            fitted_targets.append(target.copy())
            return self

        def predict_proba(self, features):
            return np.tile([0.4, 0.6], (len(features), 1))

    def recording_factory(_name, **kwargs):
        model_settings.append(kwargs)
        return RecordingModel()

    monkeypatch.setattr(evaluation, "model_factory", recording_factory)
    monkeypatch.setattr(evaluation, "clone", lambda model: model)
    index = pd.date_range("2021-01-01", periods=16, freq="D")
    dataset = pd.DataFrame(
        {
            "feature": range(16),
            "return_1d": [0.01] * 16,
            "smoothed_return_1d": [0.02] * 16,
            "target": [0, 1] * 8,
            "target_5d": [1, 0, 1, 0] * 4,
            "target_return_1d": [0.01] * 16,
            "target_return_5d": [0.05] * 16,
        },
        index=index,
    )
    close = pd.Series([100.0] * 11 + [200.0] * 5, index=index)
    selected_history_lengths = []

    def select_regime(history):
        selected_history_lengths.append(len(history))
        is_mean_reverter = history.iloc[-1] < 150
        return {
            "is_mean_reverter": is_mean_reverter,
            "lookback": 10 if is_mean_reverter else 20,
        }

    result = walk_forward_predict(
        dataset,
        "lstm",
        initial_train_size=8,
        step=4,
        sequence_columns=["return_1d", "feature"],
        close_prices=close,
        regime_selector=select_regime,
    )

    assert selected_history_lengths == [8, 12]
    pd.testing.assert_series_equal(fitted_targets[0], dataset["target_5d"].iloc[:3])
    pd.testing.assert_series_equal(fitted_targets[1], dataset["target"].iloc[:11])
    assert fitted_features[0].columns.tolist() == ["smoothed_return_1d", "feature"]
    assert fitted_features[1].columns.tolist() == ["return_1d", "feature"]
    assert model_settings == [{"lookback": 10}, {"lookback": 20}]
    expected_actuals = pd.concat(
        [dataset["target_5d"].iloc[8:12], dataset["target"].iloc[12:16]]
    ).rename("actual")
    expected_returns = pd.concat(
        [
            dataset["target_return_5d"].iloc[8:12],
            dataset["target_return_1d"].iloc[12:16],
        ]
    ).rename("actual_return")
    pd.testing.assert_series_equal(result.actuals, expected_actuals, check_freq=False)
    pd.testing.assert_series_equal(
        result.actual_returns, expected_returns, check_freq=False
    )
    assert result.is_mean_reverter.tolist() == [True] * 4 + [False] * 4


def test_walk_forward_forces_linear_model_when_regime_requests_it(monkeypatch) -> None:
    from src import evaluation

    factory_calls = []
    fitted_columns = []

    class RecordingModel:
        def fit(self, features, _target):
            fitted_columns.append(features.columns.tolist())
            return self

        def predict_proba(self, features):
            return np.tile([0.4, 0.6], (len(features), 1))

    def recording_factory(name, **kwargs):
        factory_calls.append((name, kwargs))
        return RecordingModel()

    monkeypatch.setattr(evaluation, "model_factory", recording_factory)
    monkeypatch.setattr(evaluation, "clone", lambda model: model)
    index = pd.date_range("2022-01-01", periods=12, freq="D")
    dataset = pd.DataFrame(
        {
            "feature": range(12),
            "sequence_feature": range(12),
            "target": [0, 1] * 6,
            "target_5d": [0, 1] * 6,
            "target_return_1d": [0.01] * 12,
            "target_return_5d": [0.05] * 12,
        },
        index=index,
    )
    close = pd.Series([100.0] * 12, index=index)

    result = walk_forward_predict(
        dataset,
        "lstm",
        initial_train_size=8,
        step=2,
        sequence_columns=["sequence_feature"],
        feature_columns=["feature"],
        close_prices=close,
        regime_selector=lambda _history: {
            "is_mean_reverter": True,
            "force_linear": True,
            "lookback": 5,
        },
    )

    assert factory_calls == [("ridge", {}), ("ridge", {})]
    assert fitted_columns == [["feature"], ["feature"]]
    assert result.is_mean_reverter.tolist() == [True] * 4


def test_walk_forward_forces_xgboost_for_skewed_regime(monkeypatch) -> None:
    from src import evaluation

    factory_calls = []
    fitted_columns = []

    class RecordingModel:
        def fit(self, features, _target):
            fitted_columns.append(features.columns.tolist())
            return self

        def predict_proba(self, features):
            return np.tile([0.45, 0.55], (len(features), 1))

    def recording_factory(name, **kwargs):
        factory_calls.append((name, kwargs))
        return RecordingModel()

    monkeypatch.setattr(evaluation, "model_factory", recording_factory)
    monkeypatch.setattr(evaluation, "clone", lambda model: model)
    index = pd.date_range("2022-02-01", periods=12, freq="D")
    dataset = pd.DataFrame(
        {
            "feature": range(12),
            "sequence_feature": range(12),
            "target": [0, 1] * 6,
            "target_return_1d": [0.01] * 12,
        },
        index=index,
    )
    close = pd.Series([100.0] * 12, index=index)

    result = walk_forward_predict(
        dataset,
        "lstm",
        initial_train_size=8,
        step=2,
        sequence_columns=["sequence_feature"],
        feature_columns=["feature"],
        close_prices=close,
        regime_selector=lambda _history: {
            "is_mean_reverter": False,
            "force_xgboost": True,
        },
    )

    assert factory_calls == [("xgboost", {}), ("xgboost", {})]
    assert fitted_columns == [["feature"], ["feature"]]
    assert result.is_mean_reverter.tolist() == [False] * 4


def test_model_factory_accepts_regime_lstm_parameters() -> None:
    model = model_factory(
        "lstm", lookback=5, hidden_size=16, num_layers=1, epochs=25
    )

    assert model.lookback == 5
    assert model.hidden_size == 16
    assert model.num_layers == 1
    assert model.epochs == 25


def test_model_factory_builds_xgboost_classifier() -> None:
    from xgboost import XGBClassifier

    model = model_factory("xgboost")

    assert isinstance(model, XGBClassifier)
    assert model.objective == "binary:logistic"


def test_single_class_fit_uses_constant_classifier() -> None:
    features = pd.DataFrame({"feature": [0.1, 0.2, 0.3]})
    target = pd.Series([1, 1, 1])

    model = fit_classifier_or_constant(model_factory("ridge"), features, target)

    np.testing.assert_array_equal(model.predict(features), target)
    np.testing.assert_array_equal(
        model.predict_proba(features), np.ones((len(features), 1))
    )


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


def test_dynamic_positions_tighten_gates_during_sentiment_volatility() -> None:
    probabilities = pd.Series(
        [*np.arange(0.01, 0.15, 0.01), *np.arange(0.16, 0.21, 0.01), 0.15]
    )
    calm_volatility = pd.Series(0.0, index=probabilities.index)
    high_volatility = calm_volatility.copy()
    high_volatility.iloc[-1] = 0.8

    calm_positions = dynamic_positions(
        probabilities,
        quantile_window=20,
        sentiment_volatility=calm_volatility,
    )
    adaptive_positions = dynamic_positions(
        probabilities,
        quantile_window=20,
        sentiment_volatility=high_volatility,
    )

    assert calm_positions.iloc[-2] == adaptive_positions.iloc[-2]
    assert calm_positions.iloc[-1] == 1.0
    assert adaptive_positions.iloc[-1] == 0.0


def test_dynamic_positions_inverts_execution_without_transforming_scores() -> None:
    probabilities = pd.Series(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.05]
    )

    normal_positions = dynamic_positions(probabilities, quantile_window=10)
    mean_reversion_positions = dynamic_positions(
        probabilities, quantile_window=10, invert_positions=True
    )

    assert mean_reversion_positions.tolist() == (-normal_positions).tolist()
    assert probabilities.iloc[-1] == 0.05


def test_lstm_predict_uses_asymmetric_probability_thresholds() -> None:
    model = LSTMClassifier()
    probabilities = np.array([0.05, 0.10, 0.11, 0.69, 0.70, 0.95])
    model.predict_proba = lambda _values: np.column_stack((1 - probabilities, probabilities))

    positions = model.predict(pd.DataFrame(index=range(len(probabilities))))

    assert positions.tolist() == [-1, -1, 0, 0, 1, 1]


def test_lstm_uses_chronological_validation_and_train_only_scaling(monkeypatch) -> None:
    pytest.importorskip("torch")
    feature_values = np.column_stack(
        [
            np.linspace(-1, 1, 80),
            np.concatenate([np.linspace(0, 1, 70), np.linspace(100, 120, 10)]),
        ]
    ).astype(np.float32)
    features = pd.DataFrame(feature_values, columns=["first", "second"])
    target = pd.Series(np.arange(len(features)) % 2)
    model = LSTMClassifier(
        lookback=4,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        epochs=8,
        batch_size=8,
        validation_fraction=0.2,
        validation_gap=2,
        patience=2,
    )
    captured_means = []
    normalize_windows = model._normalize_windows

    def capture_normalizer(windows):
        captured_means.append(model.feature_mean_.copy())
        return normalize_windows(windows)

    monkeypatch.setattr(model, "_normalize_windows", capture_normalizer)
    model.fit(features, target)

    window_count = len(features) - model.lookback + 1
    validation_size = int(window_count * model.validation_fraction)
    training_size = window_count - validation_size - model.validation_gap
    training_row_count = model.lookback + training_size - 1
    np.testing.assert_allclose(
        captured_means[0], feature_values[:training_row_count].mean(axis=0), atol=1e-6
    )
    np.testing.assert_allclose(
        captured_means[-1], feature_values.mean(axis=0), atol=1e-6
    )
    assert model.validation_log_loss_ is not None
    assert 1 <= model.best_epoch_ <= model.epochs
    assert model.optimization_steps_ > model.best_epoch_

    future_features = pd.DataFrame([[1.5, 2.0], [1.6, 2.1]], columns=features.columns)
    probabilities = model.predict_proba(future_features)
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_single_layer_lstm_avoids_unused_recurrent_dropout_warning() -> None:
    pytest.importorskip("torch")
    single_layer = LSTMClassifier(num_layers=1, dropout=0.2)._network(3)
    multiple_layers = LSTMClassifier(num_layers=2, dropout=0.2)._network(3)

    assert single_layer.lstm.dropout == 0.0
    assert single_layer.dropout.p == 0.2
    assert multiple_layers.lstm.dropout == 0.2


def test_lstm_tuning_uses_expanding_folds_and_selects_lower_log_loss(monkeypatch) -> None:
    from src import evaluation

    fitted_lengths = []

    class ConstantProbabilityModel:
        def __init__(self, probability):
            self.probability = probability

        def fit(self, _features, target):
            fitted_lengths.append(len(target))
            return self

        def predict_proba(self, features):
            return np.tile(
                [1 - self.probability, self.probability], (len(features), 1)
            )

    monkeypatch.setattr(
        evaluation,
        "model_factory",
        lambda _name, probability=0.6, **_kwargs: ConstantProbabilityModel(probability),
    )
    features = pd.DataFrame({"feature": np.arange(80)})
    target = pd.Series(np.tile([0, 1, 1, 1, 1], 16))
    candidates = [
        {"lookback": 5, "probability": 0.55},
        {"lookback": 5, "probability": 0.8},
    ]

    selected_model, selected, results = tune_lstm_hyperparameters(
        features,
        target,
        candidates,
        baseline_features=features,
        folds=3,
        gap=2,
    )

    assert selected_model == "lstm"
    assert selected == candidates[1]
    assert len(results) == 3
    assert results[0]["model"] == "ridge"
    assert results[1]["model"] == results[2]["model"] == "lstm"
    assert all(result["folds"] == 3 for result in results)
    assert results[2]["log_loss"] < results[1]["log_loss"]
    assert fitted_lengths[:3] == sorted(fitted_lengths[:3])


def test_probability_tuner_can_select_ridge_over_lstm(monkeypatch) -> None:
    from src import evaluation

    class ConstantProbabilityModel:
        def __init__(self, probability):
            self.probability = probability

        def fit(self, _features, _target):
            return self

        def predict_proba(self, features):
            return np.tile(
                [1 - self.probability, self.probability], (len(features), 1)
            )

    monkeypatch.setattr(
        evaluation,
        "model_factory",
        lambda name, **_kwargs: ConstantProbabilityModel(
            0.6 if name == "ridge" else 0.4
        ),
    )
    features = pd.DataFrame({"feature": np.arange(80)})
    target = pd.Series(np.tile([0, 1, 1, 1, 1], 16))

    selected_model, selected_parameters, results = tune_lstm_hyperparameters(
        features,
        target,
        [{"lookback": 5}],
        baseline_features=features,
        folds=3,
        gap=1,
    )

    assert selected_model == "ridge"
    assert selected_parameters == {}
    assert results[0]["log_loss"] < results[1]["log_loss"]


@pytest.mark.parametrize(("target_column", "gap"), [("target", 1), ("target_5d", 5)])
def test_four_probability_models_share_expanding_folds(
    monkeypatch, target_column: str, gap: int
) -> None:
    from src import evaluation

    fitted_rows = {}

    class RecordingModel:
        def __init__(self, name):
            self.name = name

        def fit(self, features, target):
            fitted_rows.setdefault(self.name, []).append(
                (len(features), len(target))
            )
            return self

        def predict_proba(self, features):
            return np.tile([0.5, 0.5], (len(features), 1))

    monkeypatch.setattr(
        evaluation,
        "model_factory",
        lambda name, **_kwargs: RecordingModel(name),
    )
    index = pd.date_range("2024-01-01", periods=80, freq="D")
    feature_sets = {
        name: pd.DataFrame({f"{name}_feature": np.arange(80)}, index=index)
        for name in ("ridge", "gradient_boosting", "random_forest", "lstm")
    }
    target = pd.Series(np.tile([0, 1], 40), index=index, name=target_column)
    candidates = {
        "ridge": [{}],
        "gradient_boosting": [{}],
        "random_forest": [{}],
        "lstm": [{"lookback": 5}, {"lookback": 10}],
    }

    selected_model, _parameters, results = compare_probability_models(
        feature_sets,
        target,
        candidates,
        folds=3,
        gap=gap,
    )

    expected_training_rows = [44 - gap, 56 - gap, 68 - gap]
    assert selected_model in candidates
    assert len(results) == 5
    assert {result["folds"] for result in results} == {3}
    assert {result["validation_rows"] for result in results} == {36}
    for model_name in ("ridge", "gradient_boosting", "random_forest"):
        assert [rows for rows, _ in fitted_rows[model_name]] == expected_training_rows
    assert [rows for rows, _ in fitted_rows["lstm"]] == expected_training_rows * 2


def test_classification_metrics_include_brier_score() -> None:
    metrics = classification_metrics(
        pd.Series([0, 1]), pd.Series([0.25, 0.75])
    )

    assert metrics["brier_score"] == pytest.approx(0.0625)
    assert metrics["expected_calibration_error"] == pytest.approx(0.25)
