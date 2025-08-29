"""
Data preprocessing utilities suitable for production and DAG orchestration.

Key features:
- Automatically drops sparse columns based on missingness ratio.
- Imputes numeric columns with median; categorical/object with a sentinel label.
- Optionally drops post-imputation constant columns.
- "Fit/transform" API with JSON serialization for reproducible application on new data.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .data_extraction import load_data, write_data

import mlflow
from mlflow.models.signature import infer_signature

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessReport:
    rows_in: int
    cols_in: int
    rows_out: int
    cols_out: int
    missing_threshold: float
    dropped_missing: List[str] = field(default_factory=list)
    dropped_constant: List[str] = field(default_factory=list)
    numeric_imputed: List[str] = field(default_factory=list)
    categorical_imputed: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class PreprocessorState:
    missing_threshold: float
    categorical_fill_value: str
    drop_constant: bool
    categorical_cols: List[str]
    numeric_impute_values: Dict[str, float]
    dropped_missing_cols: List[str]
    dropped_constant_cols: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "PreprocessorState":
        data = json.loads(text)
        return PreprocessorState(**data)


class DatasetPreprocessor:
    """Fit/transform preprocessor that can be serialized to JSON."""

    def __init__(
        self,
        *,
        missing_threshold: float = 0.9,
        categorical_fill_value: str = "unknown",
        drop_constant: bool = True,
        cast_categorical: bool = True,
    ) -> None:
        if not (0.0 <= missing_threshold <= 1.0):
            raise ValueError("missing_threshold must be within [0, 1]")
        self.missing_threshold = missing_threshold
        self.categorical_fill_value = categorical_fill_value
        self.drop_constant = drop_constant
        self.cast_categorical = cast_categorical

        # Learned during fit
        self._state: Optional[PreprocessorState] = None

    # ------------------------
    # Fit/Transform/Serialize
    # ------------------------
    def fit(self, df: pd.DataFrame) -> "DatasetPreprocessor":
        LOGGER.info(
            "Fitting preprocessor",
            extra={"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        )

        # 1) Drop sparse columns by missingness
        miss_ratio = df.isna().mean()
        dropped_missing_cols = miss_ratio[miss_ratio > self.missing_threshold].index.tolist()
        df_work = df.drop(columns=dropped_missing_cols, errors="ignore")

        # 2) Identify categorical vs numeric after dropping sparse
        categorical_cols = [c for c in df_work.columns if not is_numeric_dtype(df_work[c])]
        numeric_cols = [c for c in df_work.columns if is_numeric_dtype(df_work[c])]

        # 3) Learn numeric impute values (median)
        numeric_impute_values: Dict[str, float] = {}
        for col in numeric_cols:
            series = df_work[col]
            if series.notna().any():
                numeric_impute_values[col] = float(series.median())
            else:
                # Column all-NaN but not dropped by threshold; impute as 0.0
                numeric_impute_values[col] = 0.0

        # 4) Optionally detect constant columns after imputation
        dropped_constant_cols: List[str] = []
        if self.drop_constant:
            # Simulate imputation for const detection
            temp = df_work.copy()
            for col, val in numeric_impute_values.items():
                temp[col] = temp[col].fillna(val)
            for col in categorical_cols:
                temp[col] = temp[col].astype("string").fillna(self.categorical_fill_value)
            dropped_constant_cols = [c for c in temp.columns if temp[c].nunique(dropna=False) <= 1]

        self._state = PreprocessorState(
            missing_threshold=self.missing_threshold,
            categorical_fill_value=self.categorical_fill_value,
            drop_constant=self.drop_constant,
            categorical_cols=categorical_cols,
            numeric_impute_values=numeric_impute_values,
            dropped_missing_cols=dropped_missing_cols,
            dropped_constant_cols=dropped_constant_cols,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._state is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        st = self._state
        # Drop columns determined at fit time
        out = df.drop(columns=st.dropped_missing_cols, errors="ignore").copy()

        # Numeric imputation
        for col, val in st.numeric_impute_values.items():
            if col in out.columns:
                out[col] = out[col].astype("float", errors="ignore").fillna(val)

        # Categorical imputation
        for col in st.categorical_cols:
            if col in out.columns:
                out[col] = out[col].astype("string").fillna(st.categorical_fill_value)
                if self.cast_categorical:
                    out[col] = out[col].astype("category")

        # Drop constant columns if configured
        if st.dropped_constant_cols:
            out = out.drop(columns=[c for c in st.dropped_constant_cols if c in out.columns])

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------
    # Persistence
    # ------------------------
    def save(self, path: str | Path) -> None:
        if self._state is None:
            raise RuntimeError("Cannot save before fitting the preprocessor.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self._state.to_json(), encoding="utf-8")
        LOGGER.info("Saved preprocessor state", extra={"path": str(p)})

    @classmethod
    def load(cls, path: str | Path) -> "DatasetPreprocessor":
        p = Path(path)
        state = PreprocessorState.from_json(p.read_text(encoding="utf-8"))
        obj = cls(
            missing_threshold=state.missing_threshold,
            categorical_fill_value=state.categorical_fill_value,
            drop_constant=state.drop_constant,
        )
        obj._state = state
        LOGGER.info("Loaded preprocessor state", extra={"path": str(p)})
        return obj

    # ------------------------
    # Reporting
    # ------------------------
    def build_report(self, before: pd.DataFrame, after: pd.DataFrame) -> PreprocessReport:
        if self._state is None:
            raise RuntimeError("Preprocessor must be fitted to build report.")
        st = self._state
        return PreprocessReport(
            rows_in=int(before.shape[0]),
            cols_in=int(before.shape[1]),
            rows_out=int(after.shape[0]),
            cols_out=int(after.shape[1]),
            missing_threshold=self.missing_threshold,
            dropped_missing=list(st.dropped_missing_cols),
            dropped_constant=list(st.dropped_constant_cols),
            numeric_imputed=sorted(st.numeric_impute_values.keys()),
            categorical_imputed=sorted(st.categorical_cols),
        )


# ------------------------
# CLI for DAG/orchestration
# ------------------------


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run dataset preprocessing pipeline")
    p.add_argument("--input", required=True, help="Path to input dataset (.dta/.csv/.parquet)")
    p.add_argument("--input-format", choices=["dta", "csv", "parquet"], help="Optional explicit format")
    p.add_argument("--output", required=True, help="Path to write preprocessed dataset (.csv/.parquet/.dta)")
    p.add_argument(
        "--report",
        help="Optional path to write preprocessing report JSON",
    )
    p.add_argument(
        "--artifact",
        help="Optional path to save fitted preprocessor JSON (for reuse)",
    )
    p.add_argument(
        "--missing-threshold",
        type=float,
        default=0.9,
        help="Drop columns with missing ratio strictly greater than this value",
    )
    p.add_argument(
        "--categorical-fill",
        default="unknown",
        help="Fill value used for categorical/object columns",
    )
    p.add_argument(
        "--no-drop-constant",
        action="store_true",
        help="Do not drop post-imputation constant columns",
    )
    p.add_argument(
        "--no-cast-categorical",
        action="store_true",
        help="Do not cast object columns to 'category' dtype after imputation",
    )
    return p


def preprocess_data(argv: Optional[list[str]] = None) -> None:
    # _configure_logging()
    args = _build_argparser().parse_args(argv)

    df = load_data(args.input, fmt=args.input_format, convert_categoricals=False)

    pre = DatasetPreprocessor(
        missing_threshold=args.missing_threshold,
        categorical_fill_value=args.categorical_fill,
        drop_constant=not args.no_drop_constant,
        cast_categorical=not args.no_cast_categorical,
    )
    df_out = pre.fit_transform(df)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Poverty Prediction")
    # MLflow logging
    with mlflow.start_run(run_name="data_preprocessing"):
        # Log parameters
        mlflow.log_param("missing_threshold", args.missing_threshold)
        mlflow.log_param("categorical_fill_value", args.categorical_fill)
        mlflow.log_param("drop_constant", not args.no_drop_constant)
        mlflow.log_param("cast_categorical", not args.no_cast_categorical)
        mlflow.log_param("input_path", args.input)
        mlflow.log_param("output_path", args.output)

        # Log artifacts: preprocessor state and report if available
        if args.artifact:
            pre.save(args.artifact)
            mlflow.log_artifact(args.artifact, artifact_path="preprocessor")

        # Optionally log report
        if args.report:
            report = pre.build_report(df, df_out)
            out_path = Path(args.report)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report.to_json(), encoding="utf-8")
            mlflow.log_artifact(str(out_path), artifact_path="reports")
            LOGGER.info("Wrote preprocessing report", extra={"path": str(out_path)})

        # Log output data sample as artifact (first 100 rows)
        sample_path = Path(args.output).with_suffix(".sample.csv")
        df_out.head(100).to_csv(sample_path, index=False)
        mlflow.log_artifact(str(sample_path), artifact_path="samples")

        # Log schema/signature
        signature = infer_signature(df, df_out)
        mlflow.log_dict(signature.to_dict(), "signature.json")


    write_data(df_out, args.output)


if __name__ == "__main__":
    preprocess_data()
