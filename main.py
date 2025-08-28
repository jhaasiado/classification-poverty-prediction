import os
import sys
import time
import logging

# ensure project root is on path when running `python main.py`
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_extraction import basic_clean, load_default_data, get_target_column
from src.evaluation import evaluate_models, save_report
from src.feature_engineering import (
    build_preprocess_pipeline,
    split_features_target,
    train_test_split_stratified,
)
from src.training import fit_models, save_models


class Step:
    def __init__(self, title: str):
        self.title = title
        self.t0 = 0.0

    def __enter__(self):
        print(f"[..] {self.title} ...", flush=True)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type is None:
            print(f"[OK] {self.title} in {dt:.2f}s", flush=True)
        else:
            print(f"[FAIL] {self.title} after {dt:.2f}s: {exc}", flush=True)
        # propagate exception
        return False


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    artifacts_dir = "artifacts"
    reports_dir = "reports"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1) Load + clean
    with Step("Load dataset and basic clean"):
        df = load_default_data()
        df = basic_clean(df)

    # 2) Split features/target
    with Step("Split features and target"):
        target = get_target_column()
        X, y = split_features_target(df, target=target)

    # 3) Preprocess + split
    with Step("Build preprocess pipeline and train/test split"):
        preprocessor = build_preprocess_pipeline(X, scale_numeric=True)
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2)

    # 4) Train
    with Step("Train models"):
        models = fit_models(preprocessor, X_train, y_train)
        # models are kept in memory; saving in the next step

    # 5) Evaluate
    with Step("Evaluate models"):
        results = evaluate_models(models, X_test, y_test)

    with Step("Save models and reports"):
        save_models(models, artifacts_dir)
        save_report(
            results,
            os.path.join(reports_dir, "evaluation.json"),
            os.path.join(reports_dir, "evaluation.txt"),
        )

    # 6) Print summary
    print("\nSummary (also saved to reports/evaluation.*):")
    for name, r in results.items():
        print(f"- {name}: acc={r['accuracy']:.4f}, f1_w={r['f1_weighted']:.4f}, auc={r['roc_auc']}")
    print(f"Models saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
