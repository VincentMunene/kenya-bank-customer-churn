"""Train the selected churn model and save its prediction pipeline."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from src.features import build_pipeline
from src.prepare_data import PROJECT_ROOT, MODEL_INPUTS, clean_data


SEED = 42
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = PROJECT_ROOT / "data/raw/CBCustomer_Churn.csv"
    parameter_path = ARTIFACT_DIR / "selected_hgb_params.json"

    if not parameter_path.is_file():
        raise FileNotFoundError(
            "Export selected_hgb_params.json from the notebook first."
        )

    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))

    # Reuse the shared cleaner directly from the original CSV.
    data = clean_data(
        pd.read_csv(data_path),
        require_target=True,
    )

    # Reproduce the notebook's split.
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
    groups = train["customer_id"].to_numpy()

    classifier = HistGradientBoostingClassifier(
        random_state=SEED,
        **parameters,
    )

    pipeline = build_pipeline(classifier)

    # Each calibration fold fits its own complete preprocessing pipeline.
    model = CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        ),
        ensemble=True,
    )

    # Training-only predictions for descriptive risk-band cutoffs.
    print("Generating training out-of-fold predictions...", flush=True)

    outer_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    oof_scores = cross_val_predict(
        model,
        X_train,
        y_train,
        groups=groups,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    high_cut = float(np.quantile(oof_scores, 0.90))
    medium_cut = float(np.quantile(oof_scores, 0.70))

    print("Fitting the final calibrated model...", flush=True)
    model.fit(X_train, y_train)

    model_path = ARTIFACT_DIR / "churn_model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model": "Histogram Gradient Boosting",
        "calibration": "sigmoid",
        "parameters": parameters,
        "random_state": SEED,
        "input_columns": MODEL_INPUTS,
        "train_rows": len(train),
        "test_rows": len(test),
        "training_churn_rate": float(y_train.mean()),
        "risk_band_cutoffs": {
            "high_ge": high_cut,
            "medium_ge": medium_cut,
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
            "Churn horizon and feature observation times are unverified.",
            "Risk bands do not enforce a strict contact capacity.",
            "Existing holdout has already been examined during analysis.",
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

    # Verify serialization without evaluating the test partition.
    reloaded = joblib.load(model_path)
    sample = X_train.head(10)

    np.testing.assert_allclose(
        model.predict_proba(sample),
        reloaded.predict_proba(sample),
    )

    print(f"Training customers: {len(train):,}")
    print(f"Reserved test customers: {len(test):,}")
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    print("Model reload check passed.")


if __name__ == "__main__":
    main()