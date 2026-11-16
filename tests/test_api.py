"""API integration tests using the saved model artifacts."""

import copy
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.predict import score_customers
from src.prepare_data import PROJECT_ROOT


@pytest.fixture
def payload():
    path = PROJECT_ROOT / "examples/request.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.fixture
def client():
    # The context manager runs startup and loads the model.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_matches_batch_scoring(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    predictions = response.json()["predictions"]

    expected = score_customers(
        pd.DataFrame(payload["customers"]),
        app.state.model,
        app.state.metadata,
    ).to_dict(orient="records")

    assert len(predictions) == len(expected)

    for actual, reference in zip(predictions, expected):
        assert actual["customer_id"] == reference["customer_id"]
        assert actual["risk_band"] == reference["risk_band"]
        assert actual["p_churn"] == pytest.approx(
            reference["p_churn"], abs=1e-12
        )


def test_negative_tenure_rejected(client, payload):
    payload["customers"][0]["tenure_months"] = -1

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_required_field_rejected(client, payload):
    del payload["customers"][0]["county"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_outcome_field_rejected(client, payload):
    payload["customers"][0]["churn_value"] = 1

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_empty_batch_rejected(client):
    response = client.post("/predict", json={"customers": []})

    assert response.status_code == 422


def test_duplicate_customer_ids_rejected(client, payload):
    payload["customers"].append(
        copy.deepcopy(payload["customers"][0])
    )

    response = client.post("/predict", json=payload)

    assert response.status_code == 422