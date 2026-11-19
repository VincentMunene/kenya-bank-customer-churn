"""Compare simple classifiers using training-only cross-validation."""

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from src.features import build_simple_pipeline
from src.prepare_data import PROJECT_ROOT, MODEL_INPUTS, clean_data


SEED = 42


def main():
    data_path = PROJECT_ROOT / "data/raw/CBCustomer_Churn.csv"
    data = clean_data(pd.read_csv(data_path), require_target=True)

    # Preserve the same split used by the existing training script.
    train, _ = train_test_split(
        data,
        test_size=0.20,
        stratify=data["churn"],
        random_state=SEED,
    )

    X_train = train[MODEL_INPUTS]
    y_train = train["churn"]

    models = {
        "Dummy baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        ),
        "Logistic Regression (balanced)": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=1,
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        ),
    }

    # Explicitly define precision and F1 behavior if no churn is predicted.
    from sklearn.metrics import (
        make_scorer,
        precision_score,
        recall_score,
        f1_score,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    rows = []

    for name, estimator in models.items():
        print(f"Evaluating {name}...", flush=True)

        scores = cross_validate(
            build_simple_pipeline(estimator),
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[metric] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))

        rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["f1", "roc_auc"],
        ascending=False,
    )

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    output_path = artifact_dir / "simple_model_cv_results.csv"
    results.to_csv(output_path, index=False)

    print(f"\nTraining customers: {len(train):,}")
    print("Mean scores across five validation folds:")
    print(
        results[
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
        ].round(4).to_string(index=False)
    )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()