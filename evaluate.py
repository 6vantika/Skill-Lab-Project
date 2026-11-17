"""
evaluate.py
-----------
Loads two Hugging Face sentiment models, runs them on the same 200-sample
evaluation subset from SST-2, computes metrics, generates comparison plots,
and logs everything (parameters, metrics, tags, models, artifacts) to MLflow.

Author: S Avantika (1MS25CI108)
"""

import os
# Use HF_TOKEN from environment if set (public models do not require authentication)
if "HF_TOKEN" not in os.environ and os.path.exists(".env"):
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.strip().split("=", 1)[1].strip("\"'")
    except Exception:
        pass
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import time

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import pipeline
import transformers
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"

SEED = 42
EVAL_SAMPLES = 200
DEVICE = "cpu"
BATCH_SIZE = 32

MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B_ID = "siebert/sentiment-roberta-large-english"

MODEL_A_SHORT = "distilbert-sst2"
MODEL_B_SHORT = "roberta-large"

LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": 0}

# Champion rule (written before the final run, do not change after seeing results)
# Primary metric: macro-F1 (higher is better)
# Margin: challenger must beat champion by at least 1 percentage point (0.01)
# Tiebreaker: if within margin, the model with lower avg_latency_ms wins
# Rationale: with only 200 samples a gap smaller than 1pp is likely noise
PRIMARY_METRIC = "macro_f1"
MARGIN = 0.01
TIEBREAKER = "avg_latency_ms"

STUDENT_USN = "1MS25CI108"
DATASET_NAME = "sst2"

OUTPUT_DIR = "outputs"
DATA_DIR = "data"


# ----------------------------------------------------------------
# Custom MLflow PythonModel wrapper
# ----------------------------------------------------------------
class HFSentimentModel(mlflow.pyfunc.PythonModel):
    """
    Wraps a Hugging Face sentiment-analysis pipeline so that it can
    be logged, registered, and loaded through MLflow without the
    caller needing to touch transformers directly.
    """

    def __init__(self, model_id):
        self.model_id = model_id

    def load_context(self, context):
        from transformers import pipeline as hf_pipeline
        self.pipe = hf_pipeline(
            "sentiment-analysis",
            model=self.model_id,
            device=-1,
        )

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        else:
            texts = list(model_input)

        results = self.pipe(texts, truncation=True, batch_size=32)

        rows = []
        for text, res in zip(texts, results):
            rows.append({
                "text": text,
                "prediction": res["label"],
                "confidence": round(res["score"], 4),
            })
        return pd.DataFrame(rows)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def load_eval_data():
    """Load SST-2 validation split and sample EVAL_SAMPLES rows."""
    try:
        ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
    except Exception:
        ds = load_dataset("stanfordnlp/sst2", split="validation")
    df = ds.to_pandas()
    df = df.sample(n=EVAL_SAMPLES, random_state=SEED).reset_index(drop=True)
    return df


def run_predictions(pipe, texts):
    """Run the pipeline on a list of texts in batches. Return results + timing."""
    start = time.time()
    all_results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_out = pipe(batch, truncation=True)
        all_results.extend(batch_out)
    total_time = time.time() - start
    avg_latency_ms = (total_time / len(texts)) * 1000
    return all_results, total_time, avg_latency_ms


def compute_metrics(y_true, y_pred):
    """Compute standard classification metrics."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro"), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro"), 4),
    }


def plot_confusion_matrix(y_true, y_pred, title, filepath):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["NEGATIVE", "POSITIVE"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_metric_comparison(metrics_a, metrics_b, filepath):
    """Side-by-side bar chart comparing the two models on every metric."""
    metric_names = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    vals_a = [metrics_a[m] for m in metric_names]
    vals_b = [metrics_b[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_a = ax.bar(x - width / 2, vals_a, width,
                    label=f"A: {MODEL_A_SHORT}", color="#4285F4")
    bars_b = ax.bar(x + width / 2, vals_b, width,
                    label=f"B: {MODEL_B_SHORT}", color="#EA4335")

    ax.set_ylabel("Score")
    ax.set_title("Champion vs Challenger: Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.legend()

    for bar in bars_a + bars_b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def determine_winner(metrics_a, metrics_b, latency_a, latency_b):
    """
    Apply the champion rule:
      1) Compare macro_f1. Challenger (B) must beat Champion (A) by >= MARGIN.
      2) If the gap is within the margin in either direction, pick the faster model.
    Returns (winner_letter, reason_string, delta_f1).
    """
    f1_a = metrics_a["macro_f1"]
    f1_b = metrics_b["macro_f1"]
    delta = f1_b - f1_a   # positive = Challenger is better

    if delta >= MARGIN:
        winner = "B"
        reason = (
            f"Challenger (B) beats Champion (A) on macro-F1 by "
            f"{delta:.4f} (>= margin of {MARGIN})"
        )
    elif delta <= -MARGIN:
        winner = "A"
        reason = (
            f"Champion (A) beats Challenger (B) on macro-F1 by "
            f"{-delta:.4f} (>= margin of {MARGIN})"
        )
    else:
        # within margin, tiebreaker: lower latency wins
        if latency_a <= latency_b:
            winner = "A"
            reason = (
                f"Scores within margin (delta={delta:.4f}); "
                f"Champion (A) wins tiebreaker on latency "
                f"({latency_a:.1f}ms vs {latency_b:.1f}ms)"
            )
        else:
            winner = "B"
            reason = (
                f"Scores within margin (delta={delta:.4f}); "
                f"Challenger (B) wins tiebreaker on latency "
                f"({latency_b:.1f}ms vs {latency_a:.1f}ms)"
            )

    return winner, reason, delta


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── MLflow setup ──
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI : {TRACKING_URI}")
    print(f"Experiment          : {EXPERIMENT_NAME}\n")

    # ── 1. Load evaluation data ──
    print("[1/7] Loading SST-2 evaluation data ...")
    eval_df = load_eval_data()
    eval_path = os.path.join(DATA_DIR, "eval_inputs.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"  Saved {len(eval_df)} samples to {eval_path}")

    texts = eval_df["sentence"].tolist()
    true_labels = eval_df["label"].tolist()

    # ── 2. Load HF models ──
    print("\n[2/7] Loading Hugging Face pipelines ...")
    print(f"  Model A: {MODEL_A_ID}")
    pipe_a = pipeline("sentiment-analysis", model=MODEL_A_ID, device=-1)
    print(f"  Model B: {MODEL_B_ID}")
    pipe_b = pipeline("sentiment-analysis", model=MODEL_B_ID, device=-1)

    # ── 3. Run predictions ──
    print("\n[3/7] Running predictions ...")
    print("  Model A ...")
    results_a, time_a, latency_a = run_predictions(pipe_a, texts)
    print(f"    {time_a:.2f}s total, {latency_a:.1f}ms per sample")

    print("  Model B ...")
    results_b, time_b, latency_b = run_predictions(pipe_b, texts)
    print(f"    {time_b:.2f}s total, {latency_b:.1f}ms per sample")

    # Convert predictions to numeric labels
    preds_a_labels = [r["label"] for r in results_a]
    preds_b_labels = [r["label"] for r in results_b]
    preds_a_num = [LABEL_MAP[lbl] for lbl in preds_a_labels]
    preds_b_num = [LABEL_MAP[lbl] for lbl in preds_b_labels]
    confs_a = [round(r["score"], 4) for r in results_a]
    confs_b = [round(r["score"], 4) for r in results_b]

    # ── 4. Compute metrics ──
    print("\n[4/7] Computing metrics ...")
    metrics_a = compute_metrics(true_labels, preds_a_num)
    metrics_b = compute_metrics(true_labels, preds_b_num)

    # add timing info to the metrics dict
    metrics_a["avg_latency_ms"] = round(latency_a, 2)
    metrics_a["total_inference_s"] = round(time_a, 2)
    metrics_b["avg_latency_ms"] = round(latency_b, 2)
    metrics_b["total_inference_s"] = round(time_b, 2)

    print(f"  Model A: {metrics_a}")
    print(f"  Model B: {metrics_b}")

    # ── 5. Decide the winner ──
    winner, reason, delta = determine_winner(
        metrics_a, metrics_b, latency_a, latency_b
    )
    winner_name = MODEL_A_SHORT if winner == "A" else MODEL_B_SHORT
    print(f"\n  >> Winner: Model {winner} ({winner_name})")
    print(f"     {reason}")

    # ── 6. Save files and plots ──
    print("\n[5/7] Saving predictions, plots, and artifacts ...")

    # predictions CSVs
    pred_df_a = pd.DataFrame({
        "sentence": texts,
        "reference": true_labels,
        "prediction": preds_a_num,
        "prediction_label": preds_a_labels,
        "confidence": confs_a,
        "correct": [int(t == p) for t, p in zip(true_labels, preds_a_num)],
    })
    pred_path_a = os.path.join(OUTPUT_DIR, "predictions_distilbert.csv")
    pred_df_a.to_csv(pred_path_a, index=False)

    pred_df_b = pd.DataFrame({
        "sentence": texts,
        "reference": true_labels,
        "prediction": preds_b_num,
        "prediction_label": preds_b_labels,
        "confidence": confs_b,
        "correct": [int(t == p) for t, p in zip(true_labels, preds_b_num)],
    })
    pred_path_b = os.path.join(OUTPUT_DIR, "predictions_roberta.csv")
    pred_df_b.to_csv(pred_path_b, index=False)

    # metrics summary JSON
    summary = {
        "model_a": {"hf_model_id": MODEL_A_ID, "metrics": metrics_a},
        "model_b": {"hf_model_id": MODEL_B_ID, "metrics": metrics_b},
        "champion_rule": {
            "primary_metric": PRIMARY_METRIC,
            "margin": MARGIN,
            "tiebreaker": TIEBREAKER,
            "direction": "higher macro_f1 is better; lower latency is better",
        },
        "delta_macro_f1": round(delta, 4),
        "winner": winner,
        "reason": reason,
    }
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    # confusion matrices
    cm_path_a = os.path.join(OUTPUT_DIR, "confusion_matrix_distilbert.png")
    plot_confusion_matrix(
        true_labels, preds_a_num,
        f"Confusion Matrix: Model A ({MODEL_A_SHORT})", cm_path_a,
    )
    cm_path_b = os.path.join(OUTPUT_DIR, "confusion_matrix_roberta.png")
    plot_confusion_matrix(
        true_labels, preds_b_num,
        f"Confusion Matrix: Model B ({MODEL_B_SHORT})", cm_path_b,
    )

    # metric comparison bar chart
    bar_path = os.path.join(OUTPUT_DIR, "champion_vs_challenger_metrics.png")
    plot_metric_comparison(metrics_a, metrics_b, bar_path)

    # disagreement examples (pick first 5 where models disagree)
    disagree_rows = []
    for i in range(len(texts)):
        if preds_a_num[i] != preds_b_num[i]:
            correct_a = preds_a_num[i] == true_labels[i]
            correct_b = preds_b_num[i] == true_labels[i]
            if correct_a and not correct_b:
                comment = (
                    f"A got it right ({preds_a_labels[i]}, "
                    f"{confs_a[i]:.0%}); B predicted {preds_b_labels[i]} incorrectly"
                )
            elif correct_b and not correct_a:
                comment = (
                    f"B got it right ({preds_b_labels[i]}, "
                    f"{confs_b[i]:.0%}); A predicted {preds_a_labels[i]} incorrectly"
                )
            else:
                comment = "Both models got this wrong"
            disagree_rows.append({
                "sentence": texts[i],
                "reference_label": true_labels[i],
                "model_a_pred": preds_a_labels[i],
                "model_a_conf": confs_a[i],
                "model_b_pred": preds_b_labels[i],
                "model_b_conf": confs_b[i],
                "comment": comment,
            })

    disagree_df = pd.DataFrame(disagree_rows[:5])
    disagree_path = os.path.join(OUTPUT_DIR, "disagreement_examples.csv")
    disagree_df.to_csv(disagree_path, index=False)
    print(f"  {len(disagree_rows)} total disagreements; saved top 5 to {disagree_path}")

    # ── 7. Log everything to MLflow ──
    print("\n[6/7] Logging to MLflow ...")

    lib_versions = (
        f"transformers={transformers.__version__},"
        f"torch={torch.__version__},"
        f"mlflow={mlflow.__version__}"
    )

    pip_reqs = [
        f"transformers=={transformers.__version__}",
        f"torch=={torch.__version__}",
        f"pandas=={pd.__version__}",
    ]

    # Model signature (avoids needing to call predict during logging)
    input_schema = Schema([ColSpec("string", "text")])
    output_schema = Schema([
        ColSpec("string", "text"),
        ColSpec("string", "prediction"),
        ColSpec("double", "confidence"),
    ])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    with mlflow.start_run(run_name="compare_sentiment") as parent_run:

        # -- parent run: comparison summary --
        mlflow.log_param("primary_metric", PRIMARY_METRIC)
        mlflow.log_param("margin", MARGIN)
        mlflow.log_param("tiebreaker", TIEBREAKER)
        mlflow.log_param("eval_samples", EVAL_SAMPLES)
        mlflow.log_param("model_a", MODEL_A_ID)
        mlflow.log_param("model_b", MODEL_B_ID)

        mlflow.log_metric("delta_macro_f1", round(delta, 4))
        mlflow.log_metric("model_a_macro_f1", metrics_a["macro_f1"])
        mlflow.log_metric("model_b_macro_f1", metrics_b["macro_f1"])
        mlflow.log_metric("model_a_accuracy", metrics_a["accuracy"])
        mlflow.log_metric("model_b_accuracy", metrics_b["accuracy"])
        mlflow.log_metric("model_a_avg_latency_ms", metrics_a["avg_latency_ms"])
        mlflow.log_metric("model_b_avg_latency_ms", metrics_b["avg_latency_ms"])

        mlflow.set_tag("winner", winner)
        mlflow.set_tag("student_usn", STUDENT_USN)

        mlflow.log_artifact(bar_path)
        mlflow.log_artifact(disagree_path)
        mlflow.log_artifact(metrics_path)
        if os.path.exists("README.md"):
            mlflow.log_artifact("README.md")
        mlflow.log_artifact("requirements.txt")

        # -- child run A --
        with mlflow.start_run(
            run_name=f"A_sentiment_{MODEL_A_SHORT}", nested=True
        ) as run_a:
            mlflow.log_param("hf_model_id", MODEL_A_ID)
            mlflow.log_param("task", "sentiment-analysis")
            mlflow.log_param("role", "champion_candidate")
            mlflow.log_param("eval_samples", EVAL_SAMPLES)
            mlflow.log_param("seed", SEED)
            mlflow.log_param("device", DEVICE)
            mlflow.log_param("batch_size", BATCH_SIZE)

            for key, val in metrics_a.items():
                mlflow.log_metric(key, val)

            mlflow.set_tag("student_usn", STUDENT_USN)
            mlflow.set_tag("library_versions", lib_versions)
            mlflow.set_tag("dataset_name", DATASET_NAME)

            mlflow.log_artifact(pred_path_a)
            mlflow.log_artifact(cm_path_a)
            mlflow.log_artifact(eval_path)

            pyfunc_a = HFSentimentModel(MODEL_A_ID)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pyfunc_a,
                pip_requirements=pip_reqs,
                signature=signature,
            )
            print(f"  Model A logged (run: {run_a.info.run_id[:8]}...)")

        # -- child run B --
        with mlflow.start_run(
            run_name=f"B_sentiment_{MODEL_B_SHORT}", nested=True
        ) as run_b:
            mlflow.log_param("hf_model_id", MODEL_B_ID)
            mlflow.log_param("task", "sentiment-analysis")
            mlflow.log_param("role", "challenger_candidate")
            mlflow.log_param("eval_samples", EVAL_SAMPLES)
            mlflow.log_param("seed", SEED)
            mlflow.log_param("device", DEVICE)
            mlflow.log_param("batch_size", BATCH_SIZE)

            for key, val in metrics_b.items():
                mlflow.log_metric(key, val)

            mlflow.set_tag("student_usn", STUDENT_USN)
            mlflow.set_tag("library_versions", lib_versions)
            mlflow.set_tag("dataset_name", DATASET_NAME)

            mlflow.log_artifact(pred_path_b)
            mlflow.log_artifact(cm_path_b)
            mlflow.log_artifact(eval_path)

            pyfunc_b = HFSentimentModel(MODEL_B_ID)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pyfunc_b,
                pip_requirements=pip_reqs,
                signature=signature,
            )
            print(f"  Model B logged (run: {run_b.info.run_id[:8]}...)")

        print(f"  Summary run   : {parent_run.info.run_id[:8]}...")

    # ── Done ──
    print("\n[7/7] All done!")
    print(f"  Winner       : Model {winner} ({winner_name})")
    print(f"  delta(F1)    : {delta:+.4f}")
    print(f"  View results : {TRACKING_URI}")
    print()
    print("Next steps:")
    print("  1. python register_champion.py   (register models + set aliases)")
    print("  2. python predict_from_mlflow.py  (load champion, predict new data)")


if __name__ == "__main__":
    main()
