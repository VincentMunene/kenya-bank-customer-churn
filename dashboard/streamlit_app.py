"""Dashboard for the saved churn model and batch monitoring demo."""
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_artifacts, score_customers


@st.cache_resource
def load_model():
    return load_artifacts()


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def predict(payload):
    model, metadata = load_model()
    return score_customers(
        pd.DataFrame(payload["customers"]), model, metadata
    ).to_dict(orient="records")


st.set_page_config(page_title="Kenya Bank Churn", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp [data-testid="stMainBlockContainer"] {max-width: 1120px;}
.churn-hero {background: linear-gradient(115deg, #10253b, #114c54);
    color: #fff; padding: 1.6rem 2rem; border-radius: 1rem; margin-bottom: 1rem;}
.churn-hero h1 {color: #fff; margin: 0; padding: 0; font-size: 2.2rem;}
.churn-hero p {color: #c4e7e5; margin: .4rem 0 0;}
</style>
<section class="churn-hero">
<h1>Kenya Bank Churn</h1>
<p>Random Forest · Synthetic banking data · Academic demonstration</p>
</section>
""", unsafe_allow_html=True)

predictions, evaluation, monitoring = st.tabs(
    ["Customer scoring", "Model performance", "Drift monitoring"]
)

with predictions:
    st.subheader("Score a customer")
    st.caption("The form starts with a synthetic example. Change any inputs and select Score customer.")
    try:
        sample = read_json("examples/request.json")["customers"][0]
    except (OSError, ValueError, KeyError, IndexError) as error:
        st.error(f"Could not load the sample request: {error}")
    else:
        with st.form("customer_scoring"):
            st.markdown("**Customer profile**")
            profile, activity = st.columns(2)
            with profile:
                customer_id = st.text_input("Customer ID", sample["customer_id"], max_chars=120)
                account = st.text_input("Primary account", sample["primary_account"], max_chars=120)
                segment = st.text_input("Customer segment", sample["customer_segment"], max_chars=120)
                county = st.text_input("County", sample["county"], max_chars=120)
            with activity:
                tenure = st.number_input("Tenure (months)", min_value=0, value=int(sample["tenure_months"]))
                transactions = st.number_input("Monthly transaction value (KES)", min_value=0.0,
                                               value=float(sample["monthly_transaction_value_kes"]))
                relationship = st.number_input("Total relationship value (KES)", min_value=0.0,
                                               value=float(sample["total_relationship_value_kes"]))
                channel = st.text_input("Preferred channel", sample["preferred_transaction_channel"],
                                        max_chars=120)

            st.markdown("**Products and services**")
            services = {
                "debit_card": "Debit card", "mobile_banking": "Mobile banking",
                "internet_banking": "Internet banking", "sms_alerts": "SMS alerts",
                "standing_orders": "Standing orders", "loan_product": "Loan product",
                "savings_product": "Savings product", "insurance_product": "Insurance product",
                "e_statements": "E-statements",
            }
            service_columns = st.columns(3)
            flags = {}
            for index, (field, label) in enumerate(services.items()):
                enabled = service_columns[index % 3].checkbox(
                    label, value=sample[field] == "Yes"
                )
                flags[field] = "Yes" if enabled else "No"
            submitted = st.form_submit_button("Score customer", type="primary")

        if submitted:
            st.session_state.pop("latest_prediction", None)
            customer = {
                "customer_id": customer_id.strip(),
                "primary_account": account.strip(),
                "customer_segment": segment.strip(),
                "county": county.strip(),
                "preferred_transaction_channel": channel.strip(),
                "tenure_months": tenure,
                "monthly_transaction_value_kes": transactions,
                "total_relationship_value_kes": relationship,
                **flags,
            }
            if any(value == "" for value in customer.values()):
                st.error("Complete all customer profile fields.")
            else:
                try:
                    with st.spinner("Scoring customer…"):
                        st.session_state["latest_prediction"] = predict({"customers": [customer]})[0]
                except (OSError, ValueError, KeyError, ImportError) as error:
                    st.error(f"Could not score this customer: {error}")


        if result := st.session_state.get("latest_prediction"):
            with st.container(border=True):
                st.markdown("**Latest prediction**")
                st.caption(f"Customer {result['customer_id']}")
                score, band = st.columns(2)
                score.metric("Churn score", f"{result['p_churn']:.3f}")
                band.metric("Risk band", result["risk_band"])
                st.progress(float(result["p_churn"]))
                st.caption("Use this uncalibrated score to rank customers for review.")

with evaluation:
    st.subheader("Recorded test results")
    try:
        metrics = read_json("artifacts/script_test_metrics.json")
    except (OSError, ValueError) as error:
        st.error(f"Could not load evaluation results: {error}")
    else:
        for column, key, label, fmt in zip(
            st.columns(5),
            ["accuracy", "precision", "recall", "f1", "roc_auc"],
            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [".1%", ".1%", ".1%", ".3f", ".3f"],
        ):
            column.metric(label, format(metrics[key], fmt))
        cm = metrics["confusion_matrix"]
        st.info(f"The model caught {cm['true_positives']} of {metrics['test_churners']} churners in the test set.")
        st.markdown("**Confusion matrix**")
        st.table(pd.DataFrame(
            [[cm["true_negatives"], cm["false_positives"]],
             [cm["false_negatives"], cm["true_positives"]]],
            index=["Actual: stayed", "Actual: churned"],
            columns=["Predicted: stayed", "Predicted: churned"],
        ))
        st.caption(metrics["evaluation_note"])

with monitoring:
    st.subheader("Simulated batch comparison")
    st.caption("These saved reports come from the monitoring demonstration.")
    batch = st.radio("Batch", ["Baseline", "Shifted"], horizontal=True)
    try:
        report = read_json(f"reports/monitoring/{batch.lower()}/summary.json")
    except (OSError, ValueError) as error:
        st.error(f"Could not load the monitoring report: {error}")
    else:
        size, distance, alerts = st.columns(3)
        size.metric("Customers", report["current_customers"])
        distance.metric("Score KS distance", f"{report['score_ks_distance']:.4f}")
        alerts.metric("Input distribution alerts", len(report["features_with_distribution_alerts"]))
        names = report["features_with_distribution_alerts"]
        if names:
            st.warning("Review " + ", ".join(name.replace("_", " ") for name in names) + ".")
        else:
            st.success("No input distribution alerts in this batch.")
        st.bar_chart(pd.DataFrame({
            "Training reference": report["reference_risk_band_proportions"],
            "Selected batch": report["current_risk_band_proportions"],
        }).reindex(["Low", "Medium", "High"]).fillna(0))
        st.caption(
            f"Distribution alert threshold ≥ {report['distribution_threshold']:.2f}. "
            "Actual churn outcomes are needed to measure performance changes."
        )
