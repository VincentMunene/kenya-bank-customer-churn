"""Compare demographic inputs using training-only cross-validation."""

import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.features import build_simple_pipeline
from src.prepare_data import (
    PROJECT_ROOT,
    MODEL_INPUTS,
    CATEGORICAL_COLUMNS,
    clean_data,
    to_snake,
)


SEED = 42


def main():
    artifact_dir = PROJECT_ROOT / "artifacts"
    data_path = PROJECT_ROOT / "data/raw/CBCustomer_Churn.csv"

    metadata = json.loads(
        (artifact_dir / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if checksum != metadata["data_sha256"]:
        raise ValueError("Dataset differs from the training dataset.")

    raw = pd.read_csv(data_path)
    data = clean_data(raw, require_target=True)

    # Normalize demographic fields without changing the shared cleaner.
    demographics = raw.copy()
    demographics.columns = [
        to_snake(column) for column in demographics.columns
    ]
    demographics = demographics.rename(
        columns={"customerid": "customer_id"}
    )

    demographics["customer_id"] = (
        demographics["customer_id"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    for column in ["age_band", "gender"]:
        values = demographics[column].astype("string").str.strip()
        values = values.replace("", pd.NA)
        demographics[column] = (
            values.astype(object).where(values.notna(), np.nan)
        )

    data = data.merge(
        demographics[["customer_id", "age_band", "gender"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # Use the recorded training customers in their original order.
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv")

    if not manifest["customer_id"].is_unique:
        raise ValueError("Duplicate customer IDs in split manifest.")

    if not manifest["split"].isin(["train", "test"]).all():
        raise ValueError("Unexpected partition in split manifest.")

    if set(manifest["customer_id"]) != set(data["customer_id"]):
        raise ValueError("Split manifest does not match the dataset.")

    train_ids = manifest.loc[
        manifest["split"].eq("train"), "customer_id"
    ]

    train = (
        data.set_index("customer_id")
        .loc[train_ids]
        .reset_index()
    )

    if len(train) != metadata["train_rows"]:
        raise ValueError("Unexpected training partition size.")

    y = train["churn"].to_numpy()

    # Materialize the folds once so every version uses identical folds.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    folds = list(cv.split(train, y))

    variants = {
        "Baseline": [],
        "Add age": ["age_band"],
        "Add gender": ["gender"],
        "Add age and gender": ["age_band", "gender"],
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    summary_rows = []
    fold_rows = []
    baseline_f1 = None

    for name, extra_columns in variants.items():
        print(f"Evaluating {name}...", flush=True)

        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        )

        pipeline = build_simple_pipeline(classifier)

        # Extend only this pipeline's categorical column list.
        preprocessing = pipeline.named_steps["preprocess"]
        preprocessing.transformers = [
            (
                transformer_name,
                transformer,
                CATEGORICAL_COLUMNS + extra_columns
                if transformer_name == "categorical"
                else columns,
            )
            for transformer_name, transformer, columns
            in preprocessing.transformers
        ]

        X = train[MODEL_INPUTS + extra_columns]

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=folds,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        f1_values = scores["test_f1"]
        if baseline_f1 is None:
            baseline_f1 = f1_values.copy()

        f1_change = f1_values - baseline_f1

        row = {
            "variant": name,
            "input_count": len(MODEL_INPUTS) + len(extra_columns),
        }

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[metric] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))

        row["f1_change_vs_baseline"] = float(f1_change.mean())
        row["folds_with_higher_f1"] = int((f1_change > 0).sum())
        summary_rows.append(row)

        for fold_index in range(len(folds)):
            fold_row = {
                "variant": name,
                "fold": fold_index + 1,
                "f1_change_vs_baseline": float(
                    f1_change[fold_index]
                ),
            }
            for metric in scoring:
                fold_row[metric] = float(
                    scores[f"test_{metric}"][fold_index]
                )
            fold_rows.append(fold_row)

    summary = pd.DataFrame(summary_rows)

    summary_path = artifact_dir / "demographic_cv_results.csv"
    folds_path = artifact_dir / "demographic_cv_folds.csv"

    summary.to_csv(summary_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)

    display_columns = [
        "variant",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_sd",
        "roc_auc",
        "f1_change_vs_baseline",
        "folds_with_higher_f1",
    ]

    print(f"\nTraining customers: {len(train):,}")
    print(summary[display_columns].round(4).to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {folds_path}")
    print("Test partition was not scored. Saved model was not changed.")


if __name__ == "__main__":
    main()