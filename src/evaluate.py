"""Evaluate the saved classifier on the recorded test partition."""

import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.predict import load_artifacts
from src.prepare_data import PROJECT_ROOT, clean_data


def main():
    artifact_dir = PROJECT_ROOT / "artifacts"
    data_path = PROJECT_ROOT / "data/raw/CBCustomer_Churn.csv"

    model, metadata = load_artifacts(artifact_dir)

    # Check that the dataset has not changed since training.
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

    X_test = test[metadata["input_columns"]]
    actual = test["churn"].to_numpy()

    if len(np.unique(actual)) != 2:
        raise ValueError("Test evaluation requires both target classes.")

    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Rows = actual class; columns = predicted class.
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "model": metadata["model"],
        "test_customers": len(test),
        "test_churners": int(actual.sum()),
        "test_churn_rate": float(actual.mean()),
        "classification_rule": (
            "model.predict: highest-probability class; "
            "ties assigned to class 0."
        ),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "evaluation_note": (
            "Evaluation on the previously examined holdout; "
            "not a new independent validation. "
            "Model selection used training cross-validation."
        ),
    }

    metrics_path = artifact_dir / "script_test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    predictions_path = artifact_dir / "script_test_predictions.csv"

    pd.DataFrame({
        "customer_id": test.index,
        "actual_churn": actual,
        "predicted_churn": predicted,
        "p_churn": probabilities,
    }).to_csv(predictions_path, index=False)

    confusion_table = pd.DataFrame(
        matrix,
        index=["Actual: No churn", "Actual: Churn"],
        columns=["Predicted: No churn", "Predicted: Churn"],
    )

    confusion_path = artifact_dir / "test_confusion_matrix.csv"
    confusion_table.to_csv(confusion_path, index_label="actual_class")

    print(f"Model: {metadata['model']}")
    print(f"Test customers: {len(test):,}")
    print(f"Test churners: {int(actual.sum()):,}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    print("\nConfusion matrix:")
    print(confusion_table.to_string())

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {confusion_path}")


if __name__ == "__main__":
    main()