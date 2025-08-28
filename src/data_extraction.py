import os
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# ---- logging ----
LOGGER = logging.getLogger(__name__)

# ---- fixed defaults ----
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
DEFAULT_FILE = "MWI_2010_individual.dta"
DEFAULT_TARGET = "poor"


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix.startswith("."):
        suffix = suffix[1:]
    return suffix


def load_data(
    input_path: str | Path,
    *,
    fmt: Optional[str] = None,
    convert_categoricals: bool = False,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """General-purpose loader for .dta/.csv/.parquet with optional column subsetting."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if fmt is None:
        fmt = _infer_format(path)

    fmt = fmt.lower()
    LOGGER.info("Loading data", extra={"path": str(path), "format": fmt})

    if fmt == "dta":
        df = pd.read_stata(path, convert_categoricals=convert_categoricals)
    elif fmt == "csv":
        df = pd.read_csv(path, usecols=columns)
    elif fmt in {"parquet", "pq"}:
        df = pd.read_parquet(path, columns=list(columns) if columns else None)
    else:
        raise ValueError(
            f"Unsupported format '{fmt}'. Supported: dta, csv, parquet"
        )

    if columns is not None and fmt != "csv":
        keep = [c for c in columns if c in df.columns]
        if keep:
            df = df[keep]
        else:
            LOGGER.warning("None of the requested columns found; returning all columns")

    LOGGER.info(
        "Loaded frame",
        extra={"rows": int(df.shape[0]), "cols": int(df.shape[1])},
    )
    return df


def write_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Write a DataFrame using extension to pick format (.csv/.parquet/.dta)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _infer_format(path)

    LOGGER.info(
        "Writing data", extra={"path": str(path), "format": fmt, "rows": int(df.shape[0])}
    )

    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt in {"parquet", "pq"}:
        df.to_parquet(path, index=False)
    elif fmt == "dta":
        df.to_stata(path, write_index=False)
    else:
        raise ValueError(
            f"Unsupported output format '{fmt}'. Supported: csv, parquet, dta"
        )

def load_default_data(**read_kwargs) -> pd.DataFrame:
    """Load data/raw/MWI_2010_individual.dta using the general loader."""
    source_path = os.path.join(DATA_DIR, DEFAULT_FILE)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Dataset not found: {source_path}")
    df = load_data(source_path, fmt="dta", convert_categoricals=False)
    print(f"Loaded dataset: {source_path} with shape {df.shape}")
    return df

def get_target_column() -> str:
    """Return the default target column name ('poor')."""
    return DEFAULT_TARGET

def basic_clean(df: pd.DataFrame, drop_dupes: bool = True, strip_cols: bool = True) -> pd.DataFrame:
    """Basic cleanup: optional duplicate removal and column name stripping."""
    if drop_dupes:
        df = df.drop_duplicates().reset_index(drop=True)
    if strip_cols:
        df.columns = [c.strip() for c in df.columns]
    return df

def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(description="Data IO helper (load/write)")
    p.add_argument("--input", help="Path to input file (.dta/.csv/.parquet)")
    p.add_argument("--output", help="Optional path to write the loaded data")
    p.add_argument(
        "--format",
        choices=["dta", "csv", "parquet"],
        help="Explicit input format; inferred from extension if omitted",
    )
    p.add_argument("--columns", nargs="*", help="Optional list of columns to keep")
    p.add_argument(
        "--convert-categoricals",
        action="store_true",
        help="For Stata .dta, convert categorical encodings to labels",
    )
    p.add_argument(
        "--use-default",
        action="store_true",
        help="Ignore --input and load the project's default dataset",
    )
    return p


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def main(argv: Optional[list[str]] = None) -> None:
    _configure_logging()
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if args.use_default:
        df = load_default_data()
    else:
        if not args.input:
            parser.error("--input is required unless --use-default is provided")
        df = load_data(
            args.input,
            fmt=args.format,
            convert_categoricals=args.convert_categoricals,
            columns=args.columns,
        )

    if args.output:
        write_data(df, args.output)


if __name__ == "__main__":
    main()
