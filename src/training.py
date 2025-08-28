from typing import Dict

import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Extra models
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

# Note: Intentionally excluding SVM based on performance concern.


LOGGER = logging.getLogger(__name__)


def get_models(random_state: int = 42) -> Dict[str, object]:
    """Return exactly five models, ensuring LightGBM is included.

    If XGBoost is unavailable, fall back to GradientBoostingClassifier.
    """
    # Base three
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, random_state=random_state, n_jobs=-1, class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    # LightGBM: required
    if LGBMClassifier is None:
        raise ImportError(
            "LightGBM is required for training (pip install lightgbm)."
        )
    models["LightGBM"] = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    # XGBoost or fallback
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="auto",
            n_jobs=-1,
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=random_state)

    # Ensure exactly five entries; if we somehow have fewer, duplicate RF with different depth
    if len(models) < 5:
        models["Random Forest (alt)"] = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=random_state, n_jobs=-1, class_weight="balanced"
        )

    LOGGER.info("Prepared models", extra={"count": len(models)})
    return models


def fit_models(preprocessor, X_train, y_train) -> Dict[str, Pipeline]:
    """Wrap each model in a Pipeline with the provided preprocessor and fit."""
    fitted: Dict[str, Pipeline] = {}
    for name, model in get_models().items():
        LOGGER.info("Fitting model", extra={"model": name})
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted


def save_models(models: Dict[str, Pipeline], artifacts_dir: str) -> None:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pipe in models.items():
        path = out_dir / f"{name}.joblib"
        joblib.dump(pipe, path)
        LOGGER.info("Saved model", extra={"model": name, "path": str(path)})
