import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from .feature_engineering import train_test_split_stratified
from .data_extraction import load_data
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger(__name__)


def evaluate_model(name: str, model: Any, X_test, y_test) -> dict:
    """Evaluate a single fitted model and return its metrics."""
    is_binary = len(np.unique(y_test)) == 2

    LOGGER.info("Evaluating model", extra={"model": name})
    y_pred = model.predict(X_test)

    # Probability for ROC-AUC (binary only)
    auc = None
    if is_binary and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = None

    # Aggregate metrics
    acc = accuracy_score(y_test, y_pred)
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
    report_txt = classification_report(y_test, y_pred, zero_division=0)

    return {
        "accuracy": acc,
        "precision_weighted": pr_w,
        "recall_weighted": rc_w,
        "f1_weighted": f1_w,
        "precision_macro": pr_m,
        "recall_macro": rc_m,
        "f1_macro": f1_m,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "report": report_txt,
    }
def evaluate_models(models: Dict[str, Any], X_test, y_test) -> Dict[str, dict]:
    results = {}
    is_binary = len(np.unique(y_test)) == 2
    for name, pipe in models.items():
        LOGGER.info("Evaluating model", extra={"model": name})
        y_pred = pipe.predict(X_test)
        # Probability for ROC-AUC (binary only)
        auc = None
        if is_binary:
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = None
        # Aggregate metrics
        acc = accuracy_score(y_test, y_pred)
        pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred).tolist()
        report_txt = classification_report(y_test, y_pred, zero_division=0)
        results[name] = {
            "accuracy": acc,
            "precision_weighted": pr_w,
            "recall_weighted": rc_w,
            "f1_weighted": f1_w,
            "precision_macro": pr_m,
            "recall_macro": rc_m,
            "f1_macro": f1_m,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "report": report_txt,
        }
    return results


def save_report(
    results: Dict[str, dict], output_json_path: str, output_text_path: str | None = None
) -> None:
    json_path = Path(output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved JSON report", extra={"path": str(json_path)})

    if output_text_path:
        text_path = Path(output_text_path)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        # Also store a nicely formatted text report for quick reading
        lines = []
        for name, r in results.items():
            lines.append(f"=== {name} ===")
            lines.append(f"accuracy: {r['accuracy']:.4f}")
            lines.append(
                f"f1_weighted: {r['f1_weighted']:.4f} | f1_macro: {r['f1_macro']:.4f}"
            )
            lines.append(f"roc_auc: {r['roc_auc']}")
            lines.append("confusion_matrix:")
            lines.append(str(r["confusion_matrix"]))
            lines.append("classification_report:")
            lines.append(r["report"])
            lines.append("")
        text_path.write_text("\n".join(lines), encoding="utf-8")
        LOGGER.info("Saved text report", extra={"path": str(text_path)})

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluating Models")
    p.add_argument("--input", required=True, help="Path to input data file")
    p.add_argument("--input-format", choices=["dta", "csv", "parquet"], help="Optional explicit format")
    p.add_argument("--models-dict", required=True, help="Path to model artifacts")
    p.add_argument("--output", required=True, help="Directory to save evaluation reports")
    return p

def model_evaluation(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    # Load data
    df = load_data(args.input, fmt=args.input_format, convert_categoricals=False)
    X, y = df.drop(columns=["poor"]), df["poor"]
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Poverty Prediction")
    client = MlflowClient()
    results = {}

    # List all runs in the experiment
    experiment = mlflow.get_experiment_by_name("Poverty Prediction")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"])

    for run in runs:
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            LOGGER.warning(f"Could not load model for run {run_id}: {e}")
            continue

        name = run.data.tags.get("mlflow.runName", run_id)
        LOGGER.info("Evaluating model", extra={"model": name})
        metrics = evaluate_model(name, model, X_test, y_test)
        results[name] = metrics

        # Log metrics to MLflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            # Save confusion matrix as artifact
            cm_path = os.path.join(args.output, f"{name}_confusion_matrix.json")
            with open(cm_path, "w", encoding="utf-8") as f:
                json.dump(metrics["confusion_matrix"], f)
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

    # model_paths = {p.stem: p for p in Path(args.models_dict).glob("*.joblib")}
    # results = {}
    # for name, path in model_paths.items():
    #     LOGGER.info("Loading model", extra={"model": name})
    #     model = joblib.load(path)
    #     results[name] = evaluate_model(name, model, X_test, y_test)


    # # Save reports
    # output_json_path = os.path.join(args.output, "evaluation.json")
    # output_text_path = os.path.join(args.output, "evaluation.txt")
    # save_report(results, output_json_path, output_text_path)


