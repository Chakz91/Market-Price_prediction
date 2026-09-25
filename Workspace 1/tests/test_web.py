from pathlib import Path

import joblib
from sklearn.dummy import DummyRegressor

from src.web import create_app


def test_health_endpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model = DummyRegressor(strategy="constant", constant=0.01)
    model.fit([[0] * 8], [0.01])
    joblib.dump({"model": model, "features": [], "ticker": "AAPL"}, model_path)

    client = create_app(model_path).test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_invalid_ticker_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model = DummyRegressor(strategy="constant", constant=0.01)
    model.fit([[0]], [0.01])
    joblib.dump({"model": model, "features": [], "ticker": "AAPL"}, model_path)

    response = create_app(model_path).test_client().get("/api/predict?ticker=not%20valid")

    assert response.status_code == 400


def test_health_does_not_require_ticker_training(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model = DummyRegressor(strategy="constant", constant=0.01)
    model.fit([[0]], [0.01])
    joblib.dump({"model": model, "features": [], "ticker": "AAPL"}, model_path)

    client = create_app(model_path).test_client()
    response = client.get("/health")

    assert response.status_code == 200
