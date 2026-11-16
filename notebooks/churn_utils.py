"""churn_utils.py - stateless cleaning, feature engineering and scoring helpers.

Every function here is a pure, row-wise transformation: no statistics are learned from
the data, so it is safe to call before the train/test split and inside cross-validation.
"""
import re
import numpy as np
import pandas as pd

YES_NO_COLS = ["debit_card", "mobile_banking", "internet_banking", "sms_alerts",
               "standing_orders", "loan_product", "savings_product", "insurance_product",
               "e_statements"]
AGE_ORDER = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5}
CONSTANT_COLS = ["count", "country"]
REDUNDANT_GEO = ["city", "branch_code", "lat_long", "latitude", "longitude"]  # 1:1 with `county`


def to_snake(name: str) -> str:
    s = name.strip().lower().replace("(kes)", "kes")
    return re.sub(r"[^0-9a-z]+", "_", s).strip("_")


def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw CRM extract -> tidy frame. Works with or without the outcome columns."""
    df = raw.copy()
    df.columns = [to_snake(c) for c in df.columns]

        # Shared normalization for notebook, training and prediction.
    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = df[column].str.strip()
        df[column] = df[column].replace("", np.nan)

    for column in ["age_band", "marital_status"]:
        if column in df.columns:
            df[column] = df[column].replace(
                {"Not stated": np.nan}
            )

    # Outcome fields are optional when scoring new customers.
    if "churn_reason" in df.columns:
        df["churn_reason"] = df["churn_reason"].replace(
            {"Not applicable": np.nan}
        )

    # canonical customer id (entity resolution, as in Lab 2)
    df["customer_id"] = df["customerid"].astype(str).str.strip().str.upper()
    drop = ["customerid"] + [c for c in CONSTANT_COLS + REDUNDANT_GEO if c in df.columns]
    df = df.drop(columns=drop)

    # Yes/No -> 1/0, failing loudly on anything unexpected (a data-contract check)
    for c in YES_NO_COLS:
        s = df[c].astype(str).str.strip().str.lower()
        bad = ~s.isin(["yes", "no"])
        if bad.any():
            raise ValueError(f"Unexpected values in '{c}': {sorted(set(df.loc[bad, c].astype(str)))}")
        df[c] = s.eq("yes").astype(int)

    df["age_ord"] = df["age_band"].map(AGE_ORDER)          # ordered categorical -> integer rank
    if "churn_value" in df.columns:                         # the outcome only exists in training data
        df["churn"] = df["churn_value"].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features (Section 4). Uses only the row's own columns: never the target."""
    out = df.copy()
    mtv = out["monthly_transaction_value_kes"].astype(float)
    trv = out["total_relationship_value_kes"].astype(float)
    tenure = out["tenure_months"].clip(lower=1).astype(float)

    out["log_mtv"] = np.log1p(mtv)                              # tame heavy right tails
    out["log_trv"] = np.log1p(trv)
    out["log_avg_rel_per_month"] = np.log1p(trv / tenure)       # value built per month of relationship
    out["mtv_to_trv"] = mtv / (trv + 1.0)                       # scale-free: how "active" vs. balance

    out["n_products"] = out[["loan_product", "savings_product", "insurance_product", "standing_orders"]].sum(axis=1)
    out["n_digital"] = out[["mobile_banking", "internet_banking", "sms_alerts", "e_statements"]].sum(axis=1)
    out["new_customer"] = (out["tenure_months"] <= 12).astype(int)

    ch = out["preferred_transaction_channel"]
    out["channel_mismatch"] = (
        (ch.eq("Mobile Banking") & out["mobile_banking"].eq(0))
        | (ch.eq("Internet Banking") & out["internet_banking"].eq(0))
        | (ch.eq("Card/POS") & out["debit_card"].eq(0))
    ).astype(int)
    return out


def score_customers(raw_df: pd.DataFrame, model, input_cols, high_cut: float, med_cut: float) -> pd.DataFrame:
    """Raw CRM extract -> churn probability and risk band, highest risk first (Section 11)."""
    df = clean_raw(raw_df)
    p = model.predict_proba(df[list(input_cols)])[:, 1]
    out = pd.DataFrame({"customer_id": df["customer_id"].values, "p_churn": p})
    out["risk_band"] = np.select([p >= high_cut, p >= med_cut], ["High", "Medium"], default="Low")
    return out.sort_values("p_churn", ascending=False).reset_index(drop=True)
