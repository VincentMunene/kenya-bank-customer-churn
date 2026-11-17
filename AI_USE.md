# AI Assistance Disclosure

## Tool used

ChatGPT and Claude, assisted with this project.

## Scope of assistance

AI assistance included:

- Planning the project structure and development workflow.
- Explaining preprocessing, leakage, evaluation and calibration.
- Drafting Python modules for data preparation, feature engineering,
  training, evaluation and prediction.
- Drafting the FastAPI service, request schemas and integration tests.
- Drafting Docker configuration and troubleshooting setup errors.

## Verification performed

During development, the project was run locally in VS Code.

Verification included:

- Checking dataset size, target distribution and customer-ID uniqueness.
- Running training and evaluating the serialized model.
- Confirming that the saved model reloads successfully.
- Running batch predictions.
- Checking API predictions against batch scoring.
- Confirming rejection of invalid API inputs.
- Running seven passing API integration tests.
- Building the Docker image and receiving predictions from the container.