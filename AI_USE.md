# AI Assistance Disclosure

## Tool used

Copilot, ChatGPT and Claude.

## Scope of assistance

AI assistance included:

- Copilot was used for generation synthetic data since it is embeded in the FI system.
- Planning the project structure and development workflow.
- Drafting the FastAPI service, request schemas and integration tests.
- Drafting Docker configuration and troubleshooting errors.

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