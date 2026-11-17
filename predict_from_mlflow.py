"""
predict_from_mlflow.py
----------------------
Loads the Champion model purely through MLflow (by registry alias),
runs it on brand-new data that was NOT part of the evaluation set,
and prints which model version / HF model ID was loaded.

This script intentionally does NOT import or call the Hugging Face
pipeline directly. All prediction goes through the MLflow pyfunc wrapper.

Author: S Avantika (1MS25CI108)
"""

import os
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
import pandas as pd

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-model"
NEW_DATA_PATH = os.path.join("data", "new_data.csv")
OUTPUT_PATH = os.path.join("outputs", "new_data_predictions.csv")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # ── Load the champion model by alias ──
    champion_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    print(f"Loading champion model from: {champion_uri}")
    champion_model = mlflow.pyfunc.load_model(champion_uri)
    print("  Model loaded successfully.\n")

    # ── Figure out what was loaded ──
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    run_id = mv.run_id
    run = client.get_run(run_id)
    hf_model_id = run.data.params.get("hf_model_id", "unknown")

    print(f"  Registry name : {REGISTERED_MODEL_NAME}")
    print(f"  Alias         : champion")
    print(f"  Model version : {mv.version}")
    print(f"  Source run    : {run_id[:8]}...")
    print(f"  HF model ID   : {hf_model_id}")
    print()

    # ── Load new data ──
    new_data = pd.read_csv(NEW_DATA_PATH)
    print(f"New data ({len(new_data)} examples) from {NEW_DATA_PATH}:\n")

    # prepare input: the pyfunc wrapper expects a DataFrame with a 'text' column
    input_df = pd.DataFrame({"text": new_data["sentence"]})

    # ── Predict ──
    predictions = champion_model.predict(input_df)

    # merge predictions with ground truth for a nice table
    result_df = predictions.copy()
    result_df["expected_label"] = new_data["label"].values

    # print a clean table
    print("-" * 100)
    print(f"{'Sentence':<70} {'Predicted':<12} {'Expected':<10} {'Conf':<6}")
    print("-" * 100)
    for _, row in result_df.iterrows():
        sent = row["text"][:67] + "..." if len(row["text"]) > 70 else row["text"]
        print(f"{sent:<70} {row['prediction']:<12} {row['expected_label']:<10} {row['confidence']:<6}")
    print("-" * 100)

    # ── Save output ──
    os.makedirs("outputs", exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved predictions to {OUTPUT_PATH}")

    # ── summary ──
    correct = sum(
        (1 if pred == "POSITIVE" else 0) == exp
        for pred, exp in zip(result_df["prediction"], result_df["expected_label"])
    )
    print(f"Accuracy on new data: {correct}/{len(result_df)} ({correct/len(result_df):.0%})")
    print(f"\nNote: this script loaded the model entirely through MLflow (no direct HF calls).")


if __name__ == "__main__":
    main()
