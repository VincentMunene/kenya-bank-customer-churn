# Project checkpoint — 16 November 2026

## Completed
- Analysis notebook updated for CBCustomer_Churn.csv.
- Reusable data preparation and feature engineering modules.
- Standalone model training with sigmoid calibration.
- Serialized model and metadata; reload verified.
- Batch prediction script.
- FastAPI health and prediction endpoints.
- Example CSV and JSON requests.
- Seven API tests passed.

## Known issue
- TestClient emits a dependency deprecation warning.
- It does not prevent the current tests from passing.

## Next session
- Add standalone evaluation script.
- Add and test Dockerfile and docker-compose.yml.
- Complete README with architecture and reproduction instructions.
- Complete AI_USE.md.
- Prepare final report.
- Verify reproduction from a fresh clone.

## Commands
Train: python -m src.train
Predict: python -m src.predict --input examples/customers.csv --output artifacts/example_predictions.csv
Serve: python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Test: python -m pytest tests/test_api.py -q