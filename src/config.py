from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

NHANES_DATA_PATH = RAW_DIR / "nhanes.csv"
VARIABLE_DICT_PATH = RAW_DIR / "variables_explained.csv"
CATEGORICAL_VARIABLES_PATH = RAW_DIR / "categorical_variables.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20
ADULT_MIN_AGE = 20
TARGET_AGE = "RIDAGEYR"
TARGET_MORTALITY = "MORTSTAT"

# Survey/design fields or identifiers that should not be used as predictors.
DESIGN_COLS = [
    "SEQN",
    "SDDSRVYR",
    "RIDSTATR",
    "RIDEXMON",
    "WTINT2YR",
    "WTINT4YR",
    "WTMEC2YR",
    "WTMEC4YR",
    "SDMVPSU",
    "SDMVSTRA",
]

# These are the only explicit extra variables added to the curated ForceInc set.
# They keep the final app clinically meaningful without making the UI enormous.
CORE_EXTRA_FEATURES = [
    "RIAGENDR",
    "RIDRETH1",
    "DMDEDUC2",
    "DMDMARTL",
    "INDFMPIR",
    "SMQ040",
    "PAQ180",
    "ALQ130",
]

# Exact mortality-linkage fields; all are excluded from X and MORTSTAT becomes y.
MORTALITY_COLS = [
    "ELIGSTAT",
    "MORTSTAT",
    "UCOD_LEADING",
    "DIABETES",
    "HYPERTEN",
    "PERMTH_INT",
    "PERMTH_EXM",
]

AGE_EXACT_COLS = [
    "RIDAGEYR",
    "RIDAGEMN",
    "RIDAGEEX",
    "DMDHRAGE",
]

# Regex fragments used against the human-readable dictionary. These catch variables
# that explicitly encode the age at an earlier event (diagnosis, fracture, smoking, etc.).
AGE_LEAKAGE_PATTERNS = [
    r"\bAge first\b",
    r"\bAge when\b",
    r"\bAge at\b",
    r"\bAge started\b",
    r"\bAge last\b",
    r"\bMothers age\b",
    r"\bHH reference person age\b",
    r"\bNumber of years of age\b",
]

# Key variables exposed as controls in the Streamlit app when present in the model.
APP_EDITABLE_FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "BMXWAIST",
    "BPXPLS",
    "BPXSAR",
    "BPXDAR",
    "LBXGH",
    "LBDSGLSI",
    "LBDTCSI",
    "LBDHDLSI",
    "LBDSTRSI",
    "LBXCRP",
    "LBDSCRSI",
    "LBDSALSI",
    "SMQ040",
    "PAQ180",
    "ALQ130",
]
