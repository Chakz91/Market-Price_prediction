from run_screening import build_screening_row


def test_screening_row_includes_trade_accuracy_for_each_model() -> None:
    metrics = {
        "sector": "XLK",
        "walk_forward": {
            "trade_directional_accuracy": 0.51,
            "strategy_total_return": 0.12,
            "buy_and_hold_total_return": 0.2,
            "strategy_max_drawdown": -0.1,
            "total_turnover": 8,
        },
        "holdout_model_comparison": {
            "ridge": {"trade_directional_accuracy": 0.48},
            "gradient_boosting": {"trade_directional_accuracy": 0.52},
            "random_forest": {"trade_directional_accuracy": 0.5},
            "lstm": {"trade_directional_accuracy": 0.54},
        },
    }

    row = build_screening_row("Technology", "NVDA", metrics)

    assert row["Trade Accuracy"] == 0.51
    assert row["Ridge Trade Accuracy"] == 0.48
    assert row["Gradient Boosting Trade Accuracy"] == 0.52
    assert row["Random Forest Trade Accuracy"] == 0.5
    assert row["LSTM Trade Accuracy"] == 0.54