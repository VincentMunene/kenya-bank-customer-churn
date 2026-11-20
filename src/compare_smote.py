"""Compare class weighting and SMOTENC using training-only CV."""

import hashlib
import json

import pandas as pd

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as SamplingPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import build_simple_pipeline
from src.prepare_data import (
    PROJECT_ROOT,
    MODEL_INPUTS,
    NUMERIC_COLUMNS,
    YES_NO_COLUMNS,
    CATEGORICAL_COLUMNS,
    clean_data,
)


SEED = 42


def forest(class_weight=None):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=SEED,
        n_jobs=1,
    )


def build_smote_pipeline():
    # Output order: numeric columns, flags, then text categories.
    before_sampling = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                NUMERIC_COLUMNS,
            ),
            (
                "flags",
                SimpleImputer(strategy="most_frequent"),
                YES_NO_COLUMNS,
            ),
            (
                "categorical",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        sparse_threshold=0,
    )

    numeric_count = len(NUMERIC_COLUMNS)
    flag_end = numeric_count + len(YES_NO_COLUMNS)
    total_count = flag_end + len(CATEGORICAL_COLUMNS)

    # Treat flags and text categories as categorical during sampling.
    sampler = SMOTENC(
        categorical_features=list(range(numeric_count, total_count)),
        sampling_strategy=1.0,
        k_neighbors=5,
        random_state=SEED,
    )

    # Preserve numeric values and binary flags; encode text categories.
    after_sampling = ColumnTransformer(
        transformers=[
            (
                "numeric_and_flags",
                "passthrough",
                list(range(flag_end)),
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                list(range(flag_end, total_count)),
            ),
        ],
        sparse_threshold=0,
    )

    return SamplingPipeline([
        ("prepare", before_sampling),
        ("smote", sampler),
        ("encode", after_sampling),
        ("model", forest()),
    ])


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

    data = clean_data(pd.read_csv(data_path), require_target=True)
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

    X = train[MODEL_INPUTS]
    y = train["churn"].to_numpy()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    folds = list(cv.split(X, y))

    models = {
        "RF unweighted": build_simple_pipeline(forest()),
        "RF balanced": build_simple_pipeline(forest("balanced")),
        "RF SMOTENC": build_smote_pipeline(),
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

    for name, model in models.items():
        print(f"Evaluating {name}...", flush=True)

        scores = cross_validate(
            model,
            X,
            y,
            cv=folds,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[metric] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
        summary_rows.append(row)

        for fold_index in range(len(folds)):
            fold_row = {
                "model": name,
                "fold": fold_index + 1,
            }
            for metric in scoring:
                fold_row[metric] = float(
                    scores[f"test_{metric}"][fold_index]
                )
            fold_rows.append(fold_row)

    summary = pd.DataFrame(summary_rows).sort_values(
        "f1", ascending=False
    )

    summary_path = artifact_dir / "smote_cv_results.csv"
    folds_path = artifact_dir / "smote_cv_folds.csv"

    summary.to_csv(summary_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)

    print(f"\nTraining customers: {len(train):,}")
    print(
        summary[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "f1_sd",
                "roc_auc",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {folds_path}")
    print("Test partition was not scored.")
    print("Saved model and metadata were not changed.")


if __name__ == "__main__":
    main()