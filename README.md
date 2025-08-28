# Poverty Classification Pipeline

End-to-end classification pipeline with modular `src` components. The workflow covers data loading, preprocessing, modeling, and evaluation with reproducible artifacts and simple CLIs for orchestration/DAGs.

## Project Structure
- `src/data_extraction.py` – unified data I/O helpers (load/write), default dataset loader, basic cleaning, target accessor, and a small CLI.
- `src/data_preprocessing.py` – production-ready preprocessor: drops sparse columns by missingness, imputes numeric (median) and categorical (sentinel), optional constant-column drop, JSON artifact + CLI.
- `src/feature_engineering.py` – sklearn-style preprocessing pipeline (impute/scale/encode) used by the training pipeline.
- `src/training.py` – model training and serialization (`.joblib`).
- `src/evaluation.py` – metrics (accuracy, weighted/macro precision/recall/F1, ROC‑AUC for binary), reports to JSON/TXT.
- `main.py` – orchestrates the full run and prints progress for each phase.

## Installation
Use a virtual environment and install via the project’s `pyproject.toml`.

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
pip install -e .
```

Optional models:
- XGBoost: `pip install xgboost`
- LightGBM: `pip install lightgbm`

## Run Pipeline (Git Bash)
```bash
cd classification-poverty-prediction
python main.py
```

You’ll see step-by-step progress lines:
```
[..] Load dataset and basic clean ...
[OK] Load dataset and basic clean in 4.21s
[..] Train models ...
[OK] Train models in 68.50s
...
```

Outputs:
- Models: `artifacts/*.joblib`
- Reports: `reports/evaluation.json` and `reports/evaluation.txt`

Quick view:
```bash
cat reports/evaluation.txt
```

## Data I/O CLIs
Unified loader/writer (Stata/CSV/Parquet):
```bash
python -m src.data_extraction --use-default --output data/interim/default.parquet
python -m src.data_extraction --input data/raw/MWI_2010_individual.dta --format dta \
  --columns ind_age wta_hh poor --output data/interim/subset.parquet
```

Automatic preprocessing as a standalone step:
```bash
python -m src.data_preprocessing \
  --input data/interim/default.parquet \
  --output data/processed/preprocessed.parquet \
  --report reports/preprocess_report.json \
  --artifact artifacts/preprocessor.json \
  --missing-threshold 0.9
```

## Notes
- Default dataset expected at `data/raw/MWI_2010_individual.dta`; target column is `poor`.
- `main.py` currently uses the sklearn `feature_engineering` pipeline for modeling. The production `data_preprocessing.py` module is provided for DAGs and batch preprocessing workflows.
- XGBoost/LightGBM are optional; if unavailable, they are skipped.
