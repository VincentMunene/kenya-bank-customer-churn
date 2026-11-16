"""FastAPI service using the saved churn prediction pipeline."""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from src.predict import load_artifacts, score_customers


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, metadata = load_artifacts()

    app.state.model = model
    app.state.metadata = metadata

    yield

    del app.state.model
    del app.state.metadata


app = FastAPI(
    title="Kenya Bank Customer Churn API",
    description="Customer churn scoring for an academic demonstration.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": app.state.metadata["model"],
        "calibration": app.state.metadata["calibration"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    records = [
        customer.model_dump()
        for customer in request.customers
    ]

    raw = pd.DataFrame(records)

    try:
        predictions = score_customers(
            raw,
            app.state.model,
            app.state.metadata,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    return {
        "predictions": predictions.to_dict(orient="records")
    }