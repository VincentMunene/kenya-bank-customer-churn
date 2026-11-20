# Customer Churn Prediction for a Kenyan Financial Institution

**Student:** Vincent Munene Kinyua  
**Course:** DSA 8401 — Applied Machine Learning  
**Repository:** https://github.com/VincentMunene/kenya-bank-customer-churn

## 1. Problem framing

Customer churn can reduce a financial institution’s customer base and
future revenue. Identifying customers who may leave can support retention
review and help staff prioritize further investigation.

This project develops an end-to-end customer churn classification system
using illustrative banking data. It compares Logistic Regression and
Random Forest, evaluates the selected model and serves predictions through
a containerized API.

The system produces customer-level scores and descriptive risk bands.
It does not establish whether contacting a customer will prevent churn.

Data preparation, model comparison, training, evaluation and prediction
are implemented in separate Python scripts. The submitted workflow does
not require a notebook.

## 2. Dataset and scope

The dataset, `CBCustomer_Churn.csv`, contains 7,043 customers, including
494 churners and 6,549 non-churners. The churn prevalence is 7.01%.

The target is `Churn Value`, where 1 represents churn and 0 represents
no churn. Available fields include tenure, account characteristics,
service subscriptions, customer segment, demographics and monetary
values expressed in Kenyan shillings.

The dataset is treated as synthetic/illustrative academic data.
Its precise generation process, observation dates and churn prediction
horizon are not independently verified. It is not presented as actual
customer records from a named Kenyan bank.

Initial inspection identified missing values in geographic and demographic
fields. Customer identifiers were unique. The small proportion of churners
makes this an imbalanced classification problem.

## 3. Input selection and preprocessing

The selected model uses 16 raw inputs:

- Tenure in months.
- Monthly transaction value and total relationship value.
- Nine service flags covering debit cards, mobile banking, internet
  banking, SMS alerts, standing orders, loans, savings, insurance and
  e-statements.
- Primary account type, customer segment, preferred transaction channel
  and county.

Customer IDs are retained for traceability and partition checks but are
excluded from model predictors.

Churn Label and Churn Reason are excluded because they describe the
outcome and could cause target leakage. Churn Score and Customer Lifetime
Value are excluded because their calculation timing and data origins
are unverified.

Total Relationship Value remains an illustrative monetary input. Its
underlying components and timing are not documented, which limits its
business interpretation and requires verification before real-world use.

Age band and gender were investigated separately through training-only
cross-validation. The findings and exclusion decision are reported
in Section 6.

Shared cleaning checks required columns, unique customer IDs, valid
Yes/No service flags and nonnegative, finite observed numeric values.

The preprocessing pipeline applies:

- Median imputation and standard scaling to numeric inputs.
- Most-frequent imputation to service flags.
- Constant-value imputation and one-hot encoding to categorical inputs.

Scaling supports Logistic Regression and is retained in the shared
Random Forest pipeline. The selected model uses the original inputs
without additional engineered ratios or service-count features.

All learned preprocessing is fitted within each cross-validation training
fold. The final fitted transformations are saved with the classifier
and reused during prediction.

## 4. Experimental design and model selection

A stratified split with random seed 42 allocated 5,634 customers to
training and 1,409 to testing. The test partition contains 99 churners.

Five-fold stratified cross-validation on the training partition compared:

1. A dummy classifier that always predicts the majority class.
2. Logistic Regression without class weighting.
3. Logistic Regression with balanced class weights.
4. Random Forest without class weighting.
5. Random Forest with balanced class weights.

Logistic Regression provides a simple linear comparison. Random Forest
can capture nonlinear relationships and interactions between inputs.
Balanced class weights give churn cases more influence during training
without generating additional customer records.

F1 was established as the primary selection metric because it balances
precision and recall for the churn class. Accuracy, precision, recall
and ROC-AUC were also reported.

Mean cross-validation results were:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy baseline | 0.9299 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.9299 | 0.0000 | 0.0000 | 0.0000 | 0.6856 |
| Logistic Regression, balanced | 0.6555 | 0.1200 | 0.6177 | 0.2009 | 0.6774 |
| Random Forest | 0.9299 | 0.0000 | 0.0000 | 0.0000 | 0.7339 |
| Random Forest, balanced | 0.8568 | 0.1987 | 0.3443 | 0.2519 | 0.7195 |

Balanced Random Forest achieved the highest mean F1 and was selected.
Balanced Logistic Regression achieved higher recall but lower precision.

The unweighted classifiers achieved approximately 93% accuracy while
identifying no churners at their default classification threshold.
Their ROC-AUC scores indicate some ranking ability despite this
classification behavior.

The selected Random Forest uses 200 trees, maximum depth 8, minimum
samples per leaf 5, balanced class weights and random seed 42.
No probability calibration is applied.

These settings were used in a limited comparison rather than an exhaustive
hyperparameter search. More complex algorithms were not retained in the
final workflow to keep the implementation understandable and defensible
within the project scope.

A further training-only experiment compared unweighted Random Forest,
balanced class weights and SMOTENC using identical validation folds
and Random Forest parameters. SMOTENC was applied only inside training
folds, with five neighbours and a minority-to-majority ratio of 1:1.
It achieved mean F1 of 0.2138, compared with 0.2519 for balanced weights.
Although recall increased from 34.43% to 37.72%, precision decreased
from 19.87% to 14.93%. Balanced class weights were therefore retained.
This experiment evaluated one SMOTENC configuration and does not
establish that class weighting is universally superior.

## 5. Evaluation metrics

The evaluation uses five metrics:

- **Accuracy:** proportion of all customers classified correctly.
- **Precision:** proportion of predicted churners who actually churned.
- **Recall:** proportion of actual churners identified.
- **F1:** harmonic mean of churn precision and recall.
- **ROC-AUC:** ability to rank churners above non-churners across thresholds.

Accuracy alone is insufficient because most customers do not churn.
Precision reflects false-alarm burden, while recall reflects missed
churners. F1 summarizes their trade-off without assigning an explicit
financial cost to either error.

Accuracy, precision, recall and F1 use the model’s default class
predictions. ROC-AUC uses probability scores. No classification threshold
was selected using the test results.

## 6. Age and gender experiment

A separate experiment compared the baseline Random Forest with versions
including age band, gender, or both. All versions used the same training
customers, model parameters and five validation folds.

Age band and gender were encoded as categorical variables. Missing age
values were handled within the preprocessing pipeline.

| Inputs added | Mean F1 | F1 fold SD | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| None: baseline | 0.2519 | 0.0344 | 0.3443 | 0.7195 |
| Age band | 0.2580 | 0.0317 | 0.3392 | 0.7140 |
| Gender | 0.2496 | 0.0456 | 0.3342 | 0.7162 |
| Age band and gender | 0.2589 | 0.0363 | 0.3316 | 0.7202 |

Including both fields increased mean F1 by 0.0069 and improved F1 in
four of five folds. However, recall decreased, and ROC-AUC changed only
slightly.

The original 16 inputs were retained as a simplicity trade-off given
the small observed improvement. This does not mean the baseline had the
highest F1 across demographic variants. The fold variation is descriptive;
no formal statistical significance claim is made.

The experiment did not score the test partition or replace the deployed
model. Excluding age and gender does not guarantee fairness, and this
feature comparison is not a subgroup fairness audit.

## 7. Test results and interpretation

The evaluation script verifies the dataset checksum and evaluates the
saved model on the recorded test partition.

| Metric | Result |
|---|---:|
| Test customers | 1,409 |
| Test churners | 99 |
| Accuracy | 86.59% |
| Precision | 25.54% |
| Recall | 47.47% |
| F1 score | 0.3322 |
| ROC-AUC | 0.7609 |

The confusion matrix is:

| Actual class | Predicted no churn | Predicted churn |
|---|---:|---:|
| No churn | 1,173 | 137 |
| Churn | 52 | 47 |

The model correctly identified 47 of 99 churners and missed 52.
Of the 184 customers predicted to churn, 137 did not churn.
Approximately one in four flagged customers was an actual churner.

Predicting no churn for everyone would achieve approximately 92.97%
test accuracy but zero churn recall. The selected model has lower overall
accuracy while identifying a meaningful portion of churners.

Performance remains modest. The model could support an illustrative
review process, but the false positives and missed churners limit its
usefulness for operational decisions.

These classification results are not a fixed-capacity contact policy.
They do not estimate retention effectiveness or financial savings.

The test partition was examined during earlier project development.
These results are therefore not a new independent validation.
Model and demographic comparisons used training cross-validation,
but repeated development still limits the strength of generalization
claims.

## 8. System implementation

The system consists of reusable Python modules for preparation,
preprocessing, model comparison, training, evaluation and prediction.

The training script saves the fitted pipeline as
`artifacts/churn_model.joblib`. Metadata records the model configuration,
dataset checksum, input columns and library versions. A split manifest
records training and test customer IDs.

Batch prediction and the FastAPI service use the same scoring function.
The API provides health and prediction endpoints and loads the saved
model at startup.

Input validation rejects missing required fields, unexpected fields,
negative values and duplicate customer IDs.

Predictions contain:

- Customer ID.
- An uncalibrated model probability score named `p_churn`.
- A descriptive Low, Medium or High risk band.

Risk-band cutoffs are the 70th and 90th percentiles of training
out-of-fold scores. They are separate from the default classification
rule used for test evaluation and do not enforce contact capacity.

Because class weighting is used without calibration, the scores should
not be presented as verified absolute churn probabilities.

Docker packages the application, model and metadata. The serving
container does not require the training dataset or retrain per request.

## 9. Verification and reproducibility

Development checks confirmed:

- Successful Random Forest training and model reload.
- Evaluation using the recorded test partition.
- Successful batch prediction.
- Agreement between API and batch scoring.
- Seven passing API integration tests.
- Successful Docker image build and a healthy container.
- Agreement between container predictions and local batch predictions.

The API tests cover health, prediction consistency, negative tenure,
missing fields, unexpected outcome fields, empty batches and duplicate
customer IDs.

The workflow is run from the repository root:

    python -m src.prepare_data
    python -m src.compare_models
    python -m src.train
    python -m src.evaluate

The demographic comparison can be reproduced separately:

    python -m src.compare_demographics

A fresh-clone reproduction check remains to be completed.
Development dependency versions are recorded, but the Docker base-image
tag and transitive runtime dependencies are not fully immutable.
Bit-for-bit reproducibility across platforms is not claimed.

## CI/CD and model monitoring

GitHub Actions runs API integration tests, builds the Docker image and
verifies container health and prediction consistency on pushes and pull
requests to main.

Version tags trigger publication of the tested image to GitHub Container
Registry. Version v1.0.0 was published, downloaded and run locally.
Its prediction matched local batch scoring. This provides automated
image delivery with manual deployment; it does not deploy a public API.

Batch monitoring compares incoming inputs with the training reference.
Numeric changes use the Kolmogorov–Smirnov statistic, while categorical
changes use total variation distance. The script also reports missingness,
unseen categories, prediction-score changes and risk-band proportions.

The illustrative distribution threshold is 0.10. Missing-rate changes
of at least 0.05 trigger a quality warning. Distribution alerts require
at least 200 incoming records.

A demonstration used 500 sampled training customers and a modified copy
with reduced monthly transaction values and altered banking-service
usage. The baseline produced no alerts and a score KS distance of 0.0291.
The modified batch triggered alerts for all four deliberately changed
inputs and a score KS distance of 0.3529.

These batches demonstrate monitoring behavior, not independent validation.
The thresholds require operational validation. Input and prediction drift
do not prove concept drift or performance decline. Actual churn outcomes
would be needed to monitor precision, recall and other performance metrics.

Monitoring currently runs as a command-line batch process. It does not
collect live API traffic or automatically retrain or deploy models.

## 10. Limitations and next steps

The main limitations are:

- Synthetic/illustrative data with an unverified generation process.
- Unverified feature timing, monetary-field definitions and churn horizon.
- Random rather than future-period validation.
- A relatively small number of churn examples.
- Previously examined test data and possible model-selection optimism.
- Modest precision and recall.
- Uncalibrated probability scores.
- No current-model subgroup fairness assessment.
- No evidence that acting on predictions prevents churn.

Before real-world deployment, the institution would need to define churn
and its prediction horizon, verify feature availability before the
outcome, and validate on genuinely new customer data.

Further work should assess probability calibration, subgroup errors and
decision thresholds using training or validation data. Retention benefits
would require a controlled intervention study.

Production deployment would also require authentication, access controls,
monitoring and operational support.

## 11. Conclusion

The project demonstrates a scripted machine-learning workflow from data
validation to containerized prediction.

Balanced Random Forest was selected from the main model comparison using
training cross-validation F1. On the recorded test partition, it achieved
86.59% accuracy, 25.54% precision, 47.47% recall, an F1 score of 0.3322 and
ROC-AUC of 0.7609.

The implementation demonstrates the required pipeline and deployment
capabilities. Its predictive results and data limitations do not establish
readiness for production banking decisions.

## 12. AI assistance

AI assistance supported code drafting, debugging, evaluation explanations
and documentation.

See `AI_USE.md` for the detailed disclosure of assistance and verification.