"""Create reproducible monitoring demo batches from training rows."""

import hashlib
import json

import pandas as pd

from src.prepare_data import PROJECT_ROOT, clean_data


def main():
    source_path = PROJECT_ROOT / "data/raw/CBCustomer_Churn.csv"
    artifact_dir = PROJECT_ROOT / "artifacts"
    output_dir = PROJECT_ROOT / "examples/monitoring"

    metadata = json.loads(
        (artifact_dir / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if checksum != metadata["data_sha256"]:
        raise ValueError("Dataset differs from the training dataset.")

    raw = pd.read_csv(source_path)
    clean = clean_data(raw, require_target=True)
    manifest = pd.read_csv(artifact_dir / "split_manifest.csv")

    if not manifest["customer_id"].is_unique:
        raise ValueError("Duplicate IDs in split manifest.")

    if not manifest["split"].isin(["train", "test"]).all():
        raise ValueError("Unexpected split label.")

    if set(manifest["customer_id"]) != set(clean["customer_id"]):
        raise ValueError("Split manifest does not match the dataset.")

    train_ids = manifest.loc[
        manifest["split"].eq("train"), "customer_id"
    ]

    if len(train_ids) != metadata["train_rows"]:
        raise ValueError("Unexpected training partition size.")

    # Map raw rows to canonical IDs while retaining raw input formats.
    raw = raw.copy()
    raw.index = clean["customer_id"].to_numpy()

    sampled_ids = train_ids.sample(n=500, random_state=42).to_numpy()
    baseline = raw.loc[sampled_ids].copy().reset_index(drop=True)

    # Remove target and outcome-related fields from demonstration inputs.
    baseline = baseline.drop(
        columns=[
            "Churn Label",
            "Churn Value",
            "Churn Score",
            "Churn Reason",
            "Customer Lifetime Value (KES)",
        ],
        errors="raise",
    )

    shifted = baseline.copy()

    # Deliberate distribution changes for demonstration only.
    shifted["Monthly Transaction Value (KES)"] = (
        pd.to_numeric(
            shifted["Monthly Transaction Value (KES)"],
            errors="raise",
        ) * 0.20
    ).round(2)

    shifted["Mobile Banking"] = "No"
    shifted["Internet Banking"] = "No"
    shifted["Preferred Transaction Channel"] = "ATM"

    # Validate both batches using the operational cleaning rules.
    clean_data(baseline)
    clean_data(shifted)

    output_dir.mkdir(parents=True, exist_ok=True)

    baseline.to_csv(output_dir / "baseline.csv", index=False)
    shifted.to_csv(output_dir / "shifted.csv", index=False)

    (output_dir / "README.md").write_text(
        "# Simulated monitoring batches\n\n"
        "These files reuse 500 training customers sampled with seed 42.\n"
        "They are demonstrations, not new customer observations or "
        "independent validation data.\n\n"
        "- baseline.csv: sampled source inputs with outcome fields removed.\n"
        "- shifted.csv: the same inputs with monthly transaction values "
        "reduced by 80%, mobile and internet banking set to No, "
        "and preferred transaction channel set to ATM.\n\n"
        "No churn labels are generated. The changes demonstrate detection "
        "of distribution shifts, not changes in actual churn or proof "
        "of model performance degradation.\n",
        encoding="utf-8",
    )

    print(f"Created baseline batch: {len(baseline)} customers")
    print(f"Created shifted batch: {len(shifted)} customers")
    print(f"Saved in: {output_dir}")


if __name__ == "__main__":
    main()