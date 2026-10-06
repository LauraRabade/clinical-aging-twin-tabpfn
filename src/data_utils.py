from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    AGE_EXACT_COLS,
    AGE_LEAKAGE_PATTERNS,
    CORE_EXTRA_FEATURES,
    DESIGN_COLS,
    MORTALITY_COLS,
)


def _read_csv_flexible(path: Path, *, sep: str | None = None) -> pd.DataFrame:
    """Read a CSV/semicolon-delimited CSV with a robust encoding fallback."""
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    last_error = None
    for encoding in encodings:
        try:
            if sep is not None:
                return pd.read_csv(path, sep=sep, encoding=encoding)
            # NHANES combined data are expected to be comma-separated.
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
    raise last_error


def load_variable_dictionary(path: Path) -> pd.DataFrame:
    """Load the user's variable dictionary; its delimiter is ';'."""
    df = _read_csv_flexible(path, sep=";")
    df["Var"] = df["Var"].astype(str).str.strip()
    df["Human"] = df["Human"].fillna("").astype(str).str.strip()
    df["Demo/Exam/Quest/Lab/Mort"] = (
        df["Demo/Exam/Quest/Lab/Mort"].fillna("").astype(str).str.strip()
    )
    df["ForceInc"] = pd.to_numeric(df["ForceInc"], errors="coerce").fillna(0).astype(int)
    return df


def load_nhanes(path: Path) -> pd.DataFrame:
    df = _read_csv_flexible(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def numeric_coercion(df: pd.DataFrame, min_numeric_fraction: float = 0.90) -> pd.DataFrame:
    """Coerce mostly-numeric NHANES columns while preserving clearly non-numeric columns."""
    out = df.copy()
    for col in out.columns:
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.notna().mean() >= min_numeric_fraction:
            out[col] = converted
    return out

def load_categorical_variable_names(path: Path) -> list[str]:
    """Read the supplied NHANES categorical-variable dictionary."""

    names = []

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()

            if not line or line.startswith("Categóricas por tipo:"):
                continue

            name = line.split(maxsplit=1)[0]

            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                names.append(name)

    return list(dict.fromkeys(names))


def apply_categorical_dtypes(
    df: pd.DataFrame,
    categorical_cols: Iterable[str],
) -> pd.DataFrame:
    """Convert selected columns to pandas category dtype."""

    out = df.copy()

    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out

def get_age_leakage_cols(variable_dict: pd.DataFrame, available_cols: Iterable[str] | None = None) -> list[str]:
    available = set(available_cols) if available_cols is not None else None
    mask = variable_dict["Var"].isin(AGE_EXACT_COLS)
    human = variable_dict["Human"].fillna("")
    pattern_mask = pd.Series(False, index=variable_dict.index)
    for pattern in AGE_LEAKAGE_PATTERNS:
        pattern_mask |= human.str.contains(pattern, regex=True, case=False, na=False)
    cols = variable_dict.loc[mask | pattern_mask, "Var"].tolist()
    if available is not None:
        cols = [c for c in cols if c in available]
    return list(dict.fromkeys(cols))


def get_lab_cols(variable_dict: pd.DataFrame) -> list[str]:
    return variable_dict.loc[
        variable_dict["Demo/Exam/Quest/Lab/Mort"].eq("LAB"), "Var"
    ].tolist()


def get_forceinc_cols(variable_dict: pd.DataFrame) -> list[str]:
    return variable_dict.loc[variable_dict["ForceInc"].eq(1), "Var"].tolist()


def build_feature_sets(variable_dict: pd.DataFrame, available_cols: Iterable[str]) -> dict[str, list[str]]:
    available = set(available_cols)
    age_leak = set(get_age_leakage_cols(variable_dict, available))
    design = set(DESIGN_COLS)
    mortality = set(MORTALITY_COLS)

    forceinc = get_forceinc_cols(variable_dict)
    core_age = [
        c for c in forceinc + CORE_EXTRA_FEATURES
        if c in available and c not in age_leak and c not in design and c not in mortality
    ]
    core_age = list(dict.fromkeys(core_age))

    # A wider, optional set for experimentation. It remains capped by the user's own dictionary.
    tabular_categories = {"DEMO", "Q", "E", "LAB"}
    wide_age = variable_dict.loc[
        variable_dict["Demo/Exam/Quest/Lab/Mort"].isin(tabular_categories), "Var"
    ].tolist()
    wide_age = [
        c for c in wide_age
        if c in available and c not in age_leak and c not in design and c not in mortality
    ]

    mortality_core = [
        c for c in core_age + ["RIDAGEYR"]
        if c in available and c not in mortality and c not in design
    ]
    mortality_core = list(dict.fromkeys(mortality_core))

    return {
        "age_core": core_age,
        "age_wide": wide_age,
        "mortality_core": mortality_core,
        "age_leakage": sorted(age_leak),
        "lab": get_lab_cols(variable_dict),
        "mortality_columns": [c for c in MORTALITY_COLS if c in available],
    }


def make_model_frame(df: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in feature_cols + [target] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
