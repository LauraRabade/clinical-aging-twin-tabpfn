from pathlib import Path

from src.data_utils import build_feature_sets, load_variable_dictionary
from src.config import DESIGN_COLS, MORTALITY_COLS


def test_feature_sets_exclude_design_and_targets():
    dictionary = load_variable_dictionary(Path("data/raw/variables_explained.csv"))
    available = dictionary["Var"].tolist()
    fs = build_feature_sets(dictionary, available)

    assert "SEQN" not in fs["age_core"]
    assert "RIDAGEYR" not in fs["age_core"]
    assert "MORTSTAT" not in fs["age_core"]
    assert "MORTSTAT" not in fs["mortality_core"]
    assert "RIDAGEYR" in fs["mortality_core"]
    assert not (set(DESIGN_COLS) & set(fs["age_core"]))
    assert not (set(MORTALITY_COLS) & set(fs["age_core"]))
