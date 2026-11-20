"""Compare an incoming customer batch with the training reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.predict import load_artifacts
from src.prepare_data import (
    PROJECT_ROOT,
    NUMERIC_COLUMNS,
    YES_NO_COLUMNS,
    CATEGORICAL_COLUMNS,
    clean_data,
)


# Demonstration thresholds; validate before operational use.
DRIFT_THRESHOLD = 0.10
MISSING_CHANGE_THRESHOLD = 0.05
MIN_BATCH_ROWS = 200


def category_distribution(series):
    """Include missing values as a separate category."""
    return (
        series.astype("string")
        .fillna("__MISSING__")
        .value_counts(normalize=True)
    )


def total_variation(reference, current):
    """0 = identical proportions; 1 = completely different."""
    ref = category_distribution(reference)
    cur = category_distribution(current)
    categories = ref.index.union(cur.index)

    return float(
        0.5 * (
            ref.reindex(categories, fill_value=0)
            - cur.reindex(categories, fill_value=0)
        ).abs().sum()
    )


def numeric_distance(reference, current):
    """KS statistic: largest difference between cumulative distributions."""
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return None

    return float(ks_2samp(ref, cur).statistic)


def risk_bands(scores, metadata):
    cuts = metadata["risk_band_cutoffs"]
    return pd.Series(
        np.select(
            [
                scores >= cuts["high_ge"],
                scores >= cuts["medium_ge"],
            ],
            ["High", "Medium"],
            default="Low",
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/monitoring",
    )
    args = parser.parse_args()

    artifact_dir = PROJECT_ROOT / "artifacts"
    source_path = PROJECT_ROOT / "data/raw/CBCustomer_Churn.csv"
    model, metadata = load_artifacts(artifact_dir)

    # Verify the source used to reconstruct the training reference.
    checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if checksum != metadata["data_sha256"]:
        raise ValueError("Reference dataset differs from training data.")

    source = clean_data(pd.read_csv(source_path), require_target=True)
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv")

    if not manifest["customer_id"].is_unique:
        raise ValueError("Duplicate IDs in split manifest.")

    if not manifest["split"].isin(["train", "test"]).all():
        raise ValueError("Unexpected split label.")

    if set(manifest["customer_id"]) != set(source["customer_id"]):
        raise ValueError("Split manifest does not match reference data.")

    train_ids = manifest.loc[
        manifest["split"].eq("train"), "customer_id"
    ]
    reference = source.set_index("customer_id").loc[train_ids].copy()

    if len(reference) != metadata["train_rows"]:
        raise ValueError("Unexpected reference size.")

    # Uses the shared validation rules; invalid batches fail explicitly.
    current = clean_data(
        pd.read_csv(args.input),
        require_target=False,
    )

    if current.empty:
        raise ValueError("The incoming batch is empty.")

    enough_rows = len(current) >= MIN_BATCH_ROWS
    rows = []

    for column in metadata["input_columns"]:
        ref = reference[column]
        cur = current[column]

        reference_missing = float(ref.isna().mean())
        current_missing = float(cur.isna().mean())
        missing_change = current_missing - reference_missing

        unseen_rate = 0.0

        if column in NUMERIC_COLUMNS:
            method = "KS statistic"
            distance = numeric_distance(ref, cur)
        else:
            method = "Total variation"
            distance = total_variation(ref, cur)

            known = set(ref.dropna().unique())
            unseen_rate = float(
                (cur.notna() & ~cur.isin(known)).mean()
            )

        distribution_alert = (
            enough_rows
            and distance is not None
            and distance >= DRIFT_THRESHOLD
        )

        quality_alert = (
            abs(missing_change) >= MISSING_CHANGE_THRESHOLD
            or unseen_rate > 0
            or cur.isna().all()
        )

        rows.append({
            "feature": column,
            "method": method,
            "distance": distance,
            "reference_missing_rate": reference_missing,
            "current_missing_rate": current_missing,
            "missing_rate_change": missing_change,
            "unseen_category_rate": unseen_rate,
            "distribution_alert": bool(distribution_alert),
            "quality_alert": bool(quality_alert),
        })

    features = metadata["input_columns"]

    # Use the SAME fitted model for both score distributions.
    reference_scores = model.predict_proba(reference[features])[:, 1]
    current_scores = model.predict_proba(current[features])[:, 1]

    score_distance = numeric_distance(
        pd.Series(reference_scores),
        pd.Series(current_scores),
    )
    score_alert = bool(
        enough_rows and score_distance >= DRIFT_THRESHOLD
    )

    feature_report = pd.DataFrame(rows)
    distribution_alerts = feature_report.loc[
        feature_report["distribution_alert"], "feature"
    ].tolist()
    quality_alerts = feature_report.loc[
        feature_report["quality_alert"], "feature"
    ].tolist()

    if not enough_rows:
        status = "insufficient_batch_size"
    elif distribution_alerts or quality_alerts or score_alert:
        status = "review_required"
    else:
        status = "no_alert_detected"

    summary = {
        "status": status,
        "model": metadata["model"],
        "reference_customers": len(reference),
        "current_customers": len(current),
        "minimum_batch_rows": MIN_BATCH_ROWS,
        "distribution_threshold": DRIFT_THRESHOLD,
        "missing_change_threshold": MISSING_CHANGE_THRESHOLD,
        "features_with_distribution_alerts": distribution_alerts,
        "features_with_quality_alerts": quality_alerts,
        "reference_mean_score": float(reference_scores.mean()),
        "current_mean_score": float(current_scores.mean()),
        "score_ks_distance": score_distance,
        "score_distribution_alert": score_alert,
        "reference_risk_band_proportions": (
            risk_bands(reference_scores, metadata)
            .value_counts(normalize=True).to_dict()
        ),
        "current_risk_band_proportions": (
            risk_bands(current_scores, metadata)
            .value_counts(normalize=True).to_dict()
        ),
        "reference_data_sha256": checksum,
        "current_data_sha256": hashlib.sha256(
            args.input.read_bytes()
        ).hexdigest(),
        "model_sha256": hashlib.sha256(
            (artifact_dir / "churn_model.joblib").read_bytes()
        ).hexdigest(),
        "notes": [
            "Thresholds are illustrative and require operational validation.",
            "Distances are descriptive; no significance claim is made.",
            "Numeric distances exclude missing values; missingness is reported separately.",
            "Reference scores are fitted-model scores on training customers, not OOF scores.",
            "Training-reference score drift can reflect differences in generalization.",
            "No churn labels were used to calculate monitoring metrics.",
            "Input or prediction drift does not establish concept drift or performance decline.",
            "No automatic retraining or deployment is performed.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = args.output_dir / "feature_drift.csv"
    summary_path = args.output_dir / "summary.json"

    feature_report.to_csv(feature_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Status: {status}")
    print(f"Reference customers: {len(reference):,}")
    print(f"Current customers: {len(current):,}")
    print(f"Distribution alerts: {distribution_alerts}")
    print(f"Data quality alerts: {quality_alerts}")
    print(f"Score KS distance: {score_distance:.4f}")

    if not enough_rows:
        print("Batch too small: distribution alerts were not assessed.")

    print(f"Saved: {feature_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()