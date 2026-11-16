"""Score raw customer CSV files using the saved churn pipeline."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.prepare_data import PROJECT_ROOT, clean_data


def load_artifacts(artifact_dir=None):
    """Load trusted, locally generated model artifacts."""
    artifact_dir = (
        Path(artifact_dir)
        if artifact_dir is not None
        else PROJECT_ROOT / "artifacts"
    )

    metadata = json.loads(
        (artifact_dir / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    model = joblib.load(artifact_dir / "churn_model.joblib")

    return model, metadata


def score_customers(raw, model, metadata):
    """Score raw inputs without requiring or using churn labels."""
    if raw.empty:
        raise ValueError("The input contains no customers.")

    clean = clean_data(raw, require_target=False)
    inputs = clean[metadata["input_columns"]]

    probabilities = model.predict_proba(inputs)[:, 1]

    cutoffs = metadata["risk_band_cutoffs"]
    risk_bands = np.select(
        [
            probabilities >= cutoffs["high_ge"],
            probabilities >= cutoffs["medium_ge"],
        ],
        ["High", "Medium"],
        default="Low",
    )

    # Preserve the input row order.
    return pd.DataFrame({
        "customer_id": clean["customer_id"].to_numpy(),
        "p_churn": probabilities,
        "risk_band": risk_bands,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Generate customer churn predictions."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Output must not overwrite the input CSV.")

    model, metadata = load_artifacts()
    raw = pd.read_csv(args.input)
    predictions = score_customers(raw, model, metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print(f"Customers scored: {len(predictions):,}")
    print(f"Saved: {args.output}")
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()