"""
register_champion.py
--------------------
Finds the model runs logged by evaluate.py, registers both models in the
MLflow Model Registry under a single registered-model name, and sets the
'champion' alias on the winner and 'challenger' on the other.

Does NOT hard-code any run IDs. It searches the experiment by run name
and reads the 'winner' tag from the summary run.

Author: S Avantika (1MS25CI108)
"""

import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # find the experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found. Run evaluate.py first.")
        return
    exp_id = experiment.experiment_id
    print(f"Experiment: {EXPERIMENT_NAME} (id={exp_id})")

    # pull all runs from the experiment
    all_runs = client.search_runs(
        experiment_ids=[exp_id],
        order_by=["start_time DESC"],
    )

    # identify the three runs by their naming convention
    run_a = None
    run_b = None
    summary_run = None

    for run in all_runs:
        name = run.info.run_name or ""
        if name.startswith("A_sentiment_"):
            run_a = run
        elif name.startswith("B_sentiment_"):
            run_b = run
        elif name.startswith("compare_"):
            summary_run = run

    if run_a is None or run_b is None:
        print("Could not find Model A and/or Model B runs. Check evaluate.py output.")
        return
    if summary_run is None:
        print("Could not find the summary (compare_*) run.")
        return

    print(f"  Run A : {run_a.info.run_name} ({run_a.info.run_id[:8]}...)")
    print(f"  Run B : {run_b.info.run_name} ({run_b.info.run_id[:8]}...)")
    print(f"  Summary: {summary_run.info.run_name} ({summary_run.info.run_id[:8]}...)")

    # read the winner tag set by evaluate.py
    winner = summary_run.data.tags.get("winner", None)
    if winner is None:
        print("No 'winner' tag on the summary run. Re-run evaluate.py.")
        return
    print(f"\n  Winner from summary run: Model {winner}")

    # register Model A
    model_a_uri = f"runs:/{run_a.info.run_id}/model"
    mv_a = mlflow.register_model(model_a_uri, REGISTERED_MODEL_NAME)
    print(f"\n  Registered Model A as {REGISTERED_MODEL_NAME} v{mv_a.version}")

    # register Model B
    model_b_uri = f"runs:/{run_b.info.run_id}/model"
    mv_b = mlflow.register_model(model_b_uri, REGISTERED_MODEL_NAME)
    print(f"  Registered Model B as {REGISTERED_MODEL_NAME} v{mv_b.version}")

    # assign aliases
    if winner == "A":
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "champion", mv_a.version
        )
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "challenger", mv_b.version
        )
        print(f"\n  champion   -> v{mv_a.version} (Model A, {run_a.data.params.get('hf_model_id', '?')})")
        print(f"  challenger -> v{mv_b.version} (Model B, {run_b.data.params.get('hf_model_id', '?')})")
    else:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "champion", mv_b.version
        )
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "challenger", mv_a.version
        )
        print(f"\n  champion   -> v{mv_b.version} (Model B, {run_b.data.params.get('hf_model_id', '?')})")
        print(f"  challenger -> v{mv_a.version} (Model A, {run_a.data.params.get('hf_model_id', '?')})")

    print("\nDone. You can verify the aliases in the MLflow UI under Models.")
    print(f"  {TRACKING_URI}/#/models/{REGISTERED_MODEL_NAME}")


if __name__ == "__main__":
    main()
