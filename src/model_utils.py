from __future__ import annotations

from pathlib import Path

from tabpfn_client import TabPFNClassifier, TabPFNRegressor


def make_age_model(thinking_effort: str = "high"):
    # v3.5_default is explicitly supported by tabpfn-client 0.5.2+.
    return TabPFNRegressor(
        model_path="v3.5_default",
        thinking_effort=thinking_effort,
        random_state=42,
    )


def make_mortality_model(thinking_effort: str = "high"):
    return TabPFNClassifier(
        model_path="v3.5_default",
        thinking_effort=thinking_effort,
        random_state=42,
    )


def save_model(model, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    return model.save_model(str(path))
