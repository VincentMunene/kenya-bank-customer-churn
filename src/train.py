"""Train the selected Random Forest and save its prediction pipeline."""

import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from src.features import build_simple_pipeline
from src.prepare_data import PROJECT_ROOT, MODEL_INPUTS, clean_data


SEED = 42
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = PROJECT_ROOT / "data/raw/CBCustomer_Churn.csv"
    data = clean_data(pd.read_csv(data_path), require_target=True)

    train, test = train_test_split(
        data,
        test_size=0.20,
        stratify=data["churn"],
        random_state=SEED,
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    assert set(train["customer_id"]).isdisjoint(test["customer_id"])

    X_train = train[MODEL_INPUTS]
    y_train = train["churn"].to_numpy()

    parameters = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": 1,
    }

    model = build_simple_pipeline(
        RandomForestClassifier(**parameters)
    )

    # Training-only scores for the existing descriptive risk bands.
    print("Generating training out-of-fold scores...", flush=True)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    oof_scores = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    print("Fitting the selected Random Forest...", flush=True)
    model.fit(X_train, y_train)

    model_path = ARTIFACT_DIR / "churn_model.joblib"
    joblib.dump(model, model_path)

    # Check serialization using training rows only.
    reloaded = joblib.load(model_path)
    sample = X_train.head(10)

    np.testing.assert_allclose(
        model.predict_proba(sample),
        reloaded.predict_proba(sample),
    )

    metadata = {
        "model": "Random Forest",
        "calibration": "none",
        "parameters": parameters,
        "selection_metric": "mean cross-validation F1",
        "selection_results": "simple_model_cv_results.csv",
        "random_state": SEED,
        "input_columns": MODEL_INPUTS,
        "train_rows": len(train),
        "test_rows": len(test),
        "training_churn_rate": float(y_train.mean()),
        "classification_threshold": 0.5,
        "risk_band_cutoffs": {
            "high_ge": float(np.quantile(oof_scores, 0.90)),
            "medium_ge": float(np.quantile(oof_scores, 0.70)),
        },
        "contact_capacity": 0.10,
        "data_sha256": hashlib.sha256(
            data_path.read_bytes()
        ).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "The dataset is synthetic or illustrative.",
            "Churn horizon and feature observation times are unverified.",
            "Existing holdout has already been examined during analysis.",
            "Class-weighted scores are not calibrated churn probabilities.",
            "Risk bands are descriptive and differ from predicted classes.",
            "Risk bands do not enforce a strict contact capacity.",
            "One row per customer is required.",
        ],
    }

    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    manifest = pd.concat(
        [
            train[["customer_id"]].assign(split="train"),
            test[["customer_id"]].assign(split="test"),
        ],
        ignore_index=True,
    )

    manifest.to_csv(
        ARTIFACT_DIR / "split_manifest.csv",
        index=False,
    )

    pd.DataFrame({
        "customer_id": train["customer_id"],
        "actual_churn": y_train,
        "oof_probability": oof_scores,
    }).to_csv(
        ARTIFACT_DIR / "training_oof_scores.csv",
        index=False,
    )

    print(f"Training customers: {len(train):,}")
    print(f"Reserved test customers: {len(test):,}")
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    print("Model reload check passed.")


if __name__ == "__main__":
    main()