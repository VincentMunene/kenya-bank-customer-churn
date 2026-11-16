"""Shared feature engineering and preprocessing for training and scoring."""

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.prepare_data import YES_NO_COLUMNS, CATEGORICAL_COLUMNS


CONTINUOUS_FEATURES = [
    "tenure_months",
    "log_mtv",
    "log_trv",
    "log_avg_rel_per_month",
    "mtv_to_trv",
]

DISCRETE_FEATURES = YES_NO_COLUMNS + [
    "n_products",
    "n_digital",
    "new_customer",
    "channel_mismatch",
]


def engineer_features(data):
    """Apply the notebook's row-wise features to cleaned model inputs."""
    out = data.copy()

    mtv = out["monthly_transaction_value_kes"].astype(float)
    trv = out["total_relationship_value_kes"].astype(float)
    tenure = out["tenure_months"].clip(lower=1).astype(float)

    out["log_mtv"] = np.log1p(mtv)
    out["log_trv"] = np.log1p(trv)
    out["log_avg_rel_per_month"] = np.log1p(trv / tenure)
    out["mtv_to_trv"] = mtv / (trv + 1.0)

    # Preserve the notebook's feature definitions.
    out["n_products"] = out[
        [
            "loan_product",
            "savings_product",
            "insurance_product",
            "standing_orders",
        ]
    ].sum(axis=1, min_count=4)

    out["n_digital"] = out[
        [
            "mobile_banking",
            "internet_banking",
            "sms_alerts",
            "e_statements",
        ]
    ].sum(axis=1, min_count=4)

    out["new_customer"] = np.where(
        out["tenure_months"].isna(),
        np.nan,
        (out["tenure_months"] <= 12).astype(float),
    )

    channel = out["preferred_transaction_channel"]

    out["channel_mismatch"] = (
        (channel.eq("Mobile Banking") & out["mobile_banking"].eq(0))
        | (
            channel.eq("Internet Banking")
            & out["internet_banking"].eq(0)
        )
        | (channel.eq("Card/POS") & out["debit_card"].eq(0))
    ).astype(int)

    return out.replace([np.inf, -np.inf], np.nan)


def build_pipeline(estimator):
    """Create the preprocessing pipeline for the selected tree model."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "continuous",
                SimpleImputer(strategy="median"),
                CONTINUOUS_FEATURES,
            ),
            (
                "discrete",
                SimpleImputer(strategy="most_frequent"),
                DISCRETE_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="UNK",
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=30,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "features",
                FunctionTransformer(
                    engineer_features,
                    validate=False,
                ),
            ),
            ("preprocess", preprocessing),
            ("model", estimator),
        ]
    )