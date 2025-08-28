import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)

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
