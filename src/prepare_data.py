"""Validate and clean customer data without fitting model transforms."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

YES_NO_COLUMNS = [
    "debit_card",
    "mobile_banking",
    "internet_banking",
    "sms_alerts",
    "standing_orders",
    "loan_product",
    "savings_product",
    "insurance_product",
    "e_statements",
]

NUMERIC_COLUMNS = [
    "tenure_months",
    "monthly_transaction_value_kes",
    "total_relationship_value_kes",
]

CATEGORICAL_COLUMNS = [
    "primary_account",
    "customer_segment",
    "preferred_transaction_channel",
    "county",
]

MODEL_INPUTS = YES_NO_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def to_snake(name):
    name = name.strip().lower().replace("(kes)", "kes")
    return re.sub(r"[^0-9a-z]+", "_", name).strip("_")


def clean_data(raw, require_target=False):
    """Use the same cleaning rules for training and future predictions."""
    df = raw.copy()
    df.columns = [to_snake(column) for column in df.columns]

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization.")

    df = df.rename(columns={"customerid": "customer_id"})

    required = ["customer_id"] + MODEL_INPUTS
    if require_target:
        required.append("churn_value")

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ids = df["customer_id"].astype("string").str.strip().str.upper()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Customer IDs must be present and unique.")
    df["customer_id"] = ids

    for column in YES_NO_COLUMNS:
        values = df[column].astype("string").str.strip().str.lower()
        if not values.isin(["yes", "no"]).all():
            raise ValueError(f"{column} must contain Yes or No.")
        df[column] = values.map({"yes": 1, "no": 0}).astype(int)

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[column], errors="raise")
        if ((values < 0) | np.isinf(values)).any():
            raise ValueError(f"{column} contains negative or infinite values.")
        df[column] = values

    for column in CATEGORICAL_COLUMNS:
        values = df[column].astype("string").str.strip()
        df[column] = values.replace("", pd.NA).astype(object)
        df[column] = df[column].where(df[column].notna(), np.nan)

    # Explicit allowlist prevents outcome fields entering model inputs.
    result = df[["customer_id"] + MODEL_INPUTS].copy()

    if require_target:
        target = pd.to_numeric(df["churn_value"], errors="raise")
        if not target.isin([0, 1]).all():
            raise ValueError("Churn Value must contain only 0 or 1.")
        result["churn"] = target.astype(int)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/CBCustomer_Churn.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/customers_clean.csv",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    clean = clean_data(raw, require_target=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)

    print(f"Customers: {len(clean):,}")
    print(f"Model inputs: {len(MODEL_INPUTS)}")
    print(f"Churn rate: {clean['churn'].mean():.2%}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()