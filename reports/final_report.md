# Customer Churn Prediction for a Kenyan Financial Institution

**Student:** Vincent Munene Kinyua  
**Course:** DSA 8401 — Applied Machine Learning  
**Repository:** https://github.com/VincentMunene/kenya-bank-customer-churn

## 1. Problem framing

The project develops a reproducible system for ranking customers by
estimated churn risk. Its operational demonstration prioritizes customers
for retention review when contact capacity is limited.

The objective is prioritization rather than certainty about individual
customers or proof that contacting them will prevent churn.

The analysis notebook documents experimentation. Separate Python modules
provide data processing, training, evaluation and inference.

## 2. Dataset and scope

The submitted dataset, CBCustomer_Churn.csv, contains 7,043 customers,
including 494 churners: a positive-class prevalence of approximately 7.01%.

It includes account characteristics, service subscriptions, customer
segments, tenure and monetary values expressed in KES.

The dataset is treated as synthetic/illustrative. Its precise generation
process, observation dates and churn horizon are not independently
verified. It is not presented as an extract of actual customer records
from a named Kenyan bank.

Missing values were identified in geographic and demographic columns.
Customer identifiers were unique. "Not applicable" churn reasons were
normalized to missing values for analysis.

## 3. Leakage controls and preprocessing

Customer IDs are retained for traceability and partition checks but are
not model predictors.

Churn Label and Churn Reason were excluded because they encode or describe
the outcome. Churn Score and Customer Lifetime Value were excluded because
their calculation timing and lineage were unverified.

The final model also excludes demographic fields and redundant geographic
detail. County remains a predictor.

The system uses an explicit allowlist of 16 raw model inputs. Shared
cleaning validates IDs, service flags and numeric values.

Feature engineering creates logarithmic monetary features, tenure-related
features, an activity ratio, service counts and channel-mismatch flags.
These definitions are exploratory proxies, not verified financial
measures. For example, the feature named n_products includes standing
orders alongside loan, savings and insurance indicators.

Imputation and categorical encoding are fitted inside the model pipeline.
The same fitted transformations are reused for inference.

## 4. Experimental design and model selection

A stratified split allocated 5,634 customers to training and 1,409 to
testing. The test partition contains 99 churners.

Training cross-validation compared:

- A dummy prevalence baseline.
- Logistic regression with and without class weighting.
- Random Forest.
- Histogram Gradient Boosting.

Average precision was the primary selection metric because churn is rare.
ROC-AUC, Brier score and contact-capacity metrics provided complementary
views. Average precision is labelled PR-AUC in some notebook tables;
it is not identical to trapezoidal area under the precision–recall curve.

Randomized search tuned logistic regression and boosting using training
data. Tuned Histogram Gradient Boosting had the highest mean average
precision in the finalist comparison.

Sigmoid calibration produced only marginal improvements in training
out-of-fold metrics and was retained. Matching overall prevalence alone
was not treated as proof of calibration.

Cross-validation estimates remain subject to model-selection optimism.
The test partition was examined during analysis; subsequent scripted
evaluation is a reproduction check, not a new independent validation.

## 5. Results

The standalone evaluation script checks the dataset checksum and evaluates
the saved model using the recorded test partition.

| Metric | Result |
|---|---:|
| Test customers | 1,409 |
| Test churners | 99 |
| ROC-AUC | 0.7639 |
| Average precision | 0.2036 |
| Brier score | 0.0604 |
| Customers selected under strict 10% capacity | 140 |
| Churners reached | 36 |
| Recall at capacity | 36.36% |
| Precision at capacity | 25.71% |

The model ranks customers meaningfully above the prevalence baseline,
but many selected customers do not churn and most churners remain outside
the contact list.

The notebook's rounded-up top-decile calculation selects 141 customers;
the strict-capacity script selects 140. Their results should not be
presented as interchangeable.

## 6. Interpretation and subgroup review

Permutation importance identified tenure as the strongest contributor
to test-set ranking, followed by savings products and mobile banking.

These are predictive associations, not evidence that changing a service
subscription causes a change in churn.

Subgroup analysis included missing demographic values as a separate group.
Observed recall differed across genders and age groups. Youth customers
had higher average predicted risk than their observed churn rate, while
Corporate and SME customers showed the opposite pattern.

Small subgroup churn counts limit conclusions. Excluding demographic
predictors does not guarantee fairness, and no subgroup-specific thresholds
were chosen from these test results.

## 7. Retention-policy scenario

Two policies were compared under the same maximum capacity of 140 contacts:

| Policy | Churners reached | Recall | Precision |
|---|---:|---:|---:|
| Risk ranking | 36 | 36.4% | 25.7% |
| Value ranking | 19 | 19.2% | 13.6% |

Value ranking produced higher hypothetical financial benefits under
assumed contact costs, save rates and recoverable customer value.
These figures are not realized savings.

Risk ranking is the primary operational demonstration. Value-based
targeting would require verified pre-decision customer values and
experimental evidence of intervention effectiveness.

## 8. System implementation

The operational system consists of:

- Reusable preparation and feature-engineering modules.
- A standalone training script using exported selected parameters.
- A serialized calibrated pipeline and model metadata.
- A standalone evaluation script.
- Batch prediction.
- A FastAPI service with health and prediction endpoints.
- Docker and Docker Compose configuration.

The API loads the saved model at startup. It does not retrain per request
or require the notebook.

Input schemas reject missing required fields, negative values and
unexpected fields. Shared cleaning rejects duplicate customer IDs.

Risk bands describe training-derived score ranges. The API returns scores
and bands; it does not enforce campaign capacity or initiate outreach.

## 9. Verification and reproducibility

Development checks confirmed:

- Successful training and model reload.
- Standalone evaluation matching notebook results to displayed precision.
- Working batch predictions.
- Agreement between API and shared batch scoring.
- Seven passing API integration tests.
- Successful Docker image build and prediction through the container.

The repository records selected hyperparameters, dataset checksum,
library versions and split assignments.

The listed runtime packages are pinned, but the Docker base-image tag
and transitive dependencies are not fully immutable. Bit-for-bit
reproducibility across platforms is therefore not claimed.

## 10. Limitations and next steps

The main limitations are synthetic/illustrative data, unverified feature
timing and churn horizon, random rather than temporal validation, few
positive labels, and uncertain intervention effectiveness.

Before real-world use:

1. Establish a clear churn event and prediction horizon.
2. Obtain verified pre-outcome features and temporal customer snapshots.
3. Evaluate on a genuinely new future-period dataset.
4. Reassess calibration and subgroup errors.
5. Measure retention effectiveness through a randomized pilot.
6. Add production authentication, monitoring and operational controls.

The project demonstrates an end-to-end ML implementation, not a
production-approved banking decision system.

## 11. AI assistance

See AI_USE.md for disclosure of AI assistance and verification performed.