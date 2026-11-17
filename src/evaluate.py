"""Evaluate the saved model on the recorded test partition."""

import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.predict import load_artifacts
from src.prepare_data import PROJECT_ROOT, clean_data


def main():
    artifact_dir = PROJECT_ROOT / "artifacts"
    data_path = PROJECT_ROOT / "data/raw/CBCustomer_Churn.csv"

    model, metadata = load_artifacts(artifact_dir)

    # Ensure evaluation uses the same dataset as training.
    checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if checksum != metadata["data_sha256"]:
        raise ValueError("Dataset differs from the training dataset.")

    data = clean_data(
        pd.read_csv(data_path),
        require_target=True,
    )

    manifest = pd.read_csv(
        artifact_dir / "split_manifest.csv"
    )

    if not manifest["customer_id"].is_unique:
        raise ValueError("Duplicate customer IDs in split manifest.")

    if not manifest["split"].isin(["train", "test"]).all():
        raise ValueError("Unexpected partition in split manifest.")

    if set(manifest["customer_id"]) != set(data["customer_id"]):
        raise ValueError("Split manifest does not match the dataset.")

    test_ids = manifest.loc[
        manifest["split"].eq("test"), "customer_id"
    ]

    # Preserve the recorded test order.
    test = data.set_index("customer_id").loc[test_ids].copy()

    if len(test) != metadata["test_rows"]:
        raise ValueError("Unexpected test partition size.")

    actual = test["churn"].to_numpy()
    probabilities = model.predict_proba(
        test[metadata["input_columns"]]
    )[:, 1]

    if len(np.unique(actual)) != 2:
        raise ValueError("Test evaluation requires both target classes.")

    # Strict capacity: never exceed 10% of the batch.
    capacity = metadata["contact_capacity"]
    contact_count = int(np.floor(capacity * len(test)))

    if contact_count < 1:
        raise ValueError("Test partition is too small for this capacity.")

    ranked = np.argsort(-probabilities, kind="mergesort")

    selected = np.zeros(len(test), dtype=bool)
    selected[ranked[:contact_count]] = True

    reached = int(actual[selected].sum())
    precision = float(actual[selected].mean())
    prevalence = float(actual.mean())

    metrics = {
        "test_customers": len(test),
        "test_churners": int(actual.sum()),
        "test_churn_rate": prevalence,
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "average_precision": float(
            average_precision_score(actual, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(actual, probabilities)
        ),
        "contact_capacity": capacity,
        "customers_selected": contact_count,
        "churners_reached": reached,
        "precision_at_capacity": precision,
        "recall_at_capacity": float(reached / actual.sum()),
        "lift_at_capacity": float(precision / prevalence),
        "evaluation_note": (
            "Reproduction check on the previously examined holdout; "
            "not a new independent validation."
        ),
    }

    metrics_path = artifact_dir / "script_test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame({
        "customer_id": test.index,
        "actual_churn": actual,
        "p_churn": probabilities,
        "selected_at_capacity": selected,
    }).to_csv(
        artifact_dir / "script_test_predictions.csv",
        index=False,
    )

    print(f"Test customers: {len(test):,}")
    print(f"Test churners: {actual.sum():,}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Customers selected: {contact_count}")
    print(f"Churners reached: {reached}")
    print(f"Recall at capacity: {metrics['recall_at_capacity']:.2%}")
    print(f"Precision at capacity: {precision:.2%}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()