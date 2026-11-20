# Simulated monitoring batches

These files reuse 500 training customers sampled with seed 42.
They are demonstrations, not new customer observations or independent validation data.

- baseline.csv: sampled source inputs with outcome fields removed.
- shifted.csv: the same inputs with monthly transaction values reduced by 80%, mobile and internet banking set to No, and preferred transaction channel set to ATM.

No churn labels are generated. The changes demonstrate detection of distribution shifts, not changes in actual churn or proof of model performance degradation.
