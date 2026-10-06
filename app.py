from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tabpfn_client import TabPFNClassifier, TabPFNRegressor

from src.data_utils import apply_categorical_dtypes

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
DEMO_DIR = ROOT / "data" / "demo"
MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="Clinical Aging Twin",
    page_icon="🧬",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Human-readable labels for the variables exposed in the interactive UI.
# The model continues to use the original NHANES variable codes internally.
# -----------------------------------------------------------------------------
FEATURE_UI = {
    "RIAGENDR": {
        "label": "Sex",
        "help": "Participant sex recorded in NHANES.",
        "type": "sex",
    },
    "RIDAGEYR": {
        "label": "Chronological age",
        "help": "Age in years at the NHANES screening interview.",
        "type": "age",
    },
    "BMXBMI": {
        "label": "Body Mass Index (BMI)",
        "help": "Weight relative to height. Higher values indicate greater body mass relative to height.",
        "unit": "kg/m²",
    },
    "BMXWAIST": {
        "label": "Waist circumference",
        "help": "Circumference around the waist, used as a measure of central body fat.",
        "unit": "cm",
    },
    "BPXPLS": {
        "label": "Resting heart rate",
        "help": "Pulse measured during the physical examination.",
        "unit": "beats/min",
    },
    "BPXSAR": {
        "label": "Systolic blood pressure",
        "help": "Average systolic blood pressure measured during the physical examination.",
        "unit": "mmHg",
    },
    "BPXDAR": {
        "label": "Diastolic blood pressure",
        "help": "Average diastolic blood pressure measured during the physical examination.",
        "unit": "mmHg",
    },
    "LBXGH": {
        "label": "HbA1c (glycated haemoglobin)",
        "help": "Blood marker reflecting average blood glucose over the previous few months.",
        "unit": "%",
    },
    "LBDSGLSI": {
        "label": "Blood glucose",
        "help": "Glucose concentration measured in blood.",
        "unit": "mmol/L",
    },
    "LBDTCSI": {
        "label": "Total cholesterol",
        "help": "Total cholesterol concentration in blood.",
        "unit": "mmol/L",
    },
    "LBDHDLSI": {
        "label": "HDL cholesterol",
        "help": "HDL cholesterol concentration in blood.",
        "unit": "mmol/L",
    },
    "LBDSTRSI": {
        "label": "Triglycerides",
        "help": "Triglyceride concentration in blood.",
        "unit": "mmol/L",
    },
    "LBXCRP": {
        "label": "C-reactive protein (CRP)",
        "help": "A blood marker commonly used as an indicator of systemic inflammation.",
        "unit": "mg/dL",
    },
    "LBDSCRSI": {
        "label": "Creatinine",
        "help": "A blood marker commonly used to assess kidney function.",
        "unit": "µmol/L",
    },
    "LBDSALSI": {
        "label": "Albumin",
        "help": "A major blood protein that can reflect nutritional and physiological status.",
        "unit": "g/L",
    },
    "PAQ180": {
        "label": "Daily physical activity level",
        "help": (
            "Self-reported usual daily activity. NHANES codes 1–4 are ordinal categories: "
            "1 = mainly sitting, 2 = walking/standing a lot, 3 = light loads or frequent stairs/hills, "
            "4 = heavy work or heavy loads. This variable is not measured in hours/day."
        ),
        "type": "physical_activity",
    },
    "ALQ130": {
        "label": "Average alcoholic drinks per day",
        "help": "Self-reported average number of alcoholic drinks consumed per day during the previous 12 months.",
        "unit": "drinks/day",
    },
    "SMQ040": {
        "label": "Current cigarette smoking",
        "help": "Current smoking status reported by the participant.",
        "type": "smoking",
    },
}

SMOKING_LABELS = {
    1: "Every day",
    2: "Some days",
    7: "Refused / not reported",
    9: "Don't know / not reported",
}

SEX_LABELS = {
    1: "Male",
    2: "Female",
}

PAQ180_LABELS = {
    1: "1 — Mainly sitting",
    2: "2 — Standing/walking a lot",
    3: "3 — Light loads or frequent stairs/hills",
    4: "4 — Heavy work or heavy loads",
    7: "Refused / not reported",
    9: "Don't know / not reported",
}

EDITABLE_NUMERIC = [
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
    "PAQ180",
    "ALQ130",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_or_train_models():
    """Load full fitted models; fall back to small demo fits when unavailable."""
    used_demo_fallback = False

    try:
        age_model = TabPFNRegressor.load_model(str(MODELS_DIR / "age_model.json"))
        mortality_model = TabPFNClassifier.load_model(str(MODELS_DIR / "mortality_model.json"))
        return age_model, mortality_model, used_demo_fallback
    except Exception:
        used_demo_fallback = True

    age_demo = pd.read_csv(DEMO_DIR / "train_age_demo.csv")
    mort_demo = pd.read_csv(DEMO_DIR / "train_mortality_demo.csv")
    metadata = load_json(PROCESSED_DIR / "metadata.json")
    age_features = metadata["age_features"]
    mortality_features = metadata["mortality_features"]
    categorical_age_features = metadata.get("categorical_age_features", [])
    categorical_mortality_features = metadata.get("categorical_mortality_features", [])

    age_demo = apply_categorical_dtypes(age_demo, categorical_age_features)
    mort_demo = apply_categorical_dtypes(mort_demo, categorical_mortality_features)

    age_model = TabPFNRegressor(
        model_path="v3.5_default",
        thinking_effort="medium",
        random_state=42,
    )
    age_model.fit(age_demo[age_features], age_demo["RIDAGEYR"])

    mortality_model = TabPFNClassifier(
        model_path="v3.5_default",
        thinking_effort="medium",
        random_state=42,
    )
    mortality_model.fit(mort_demo[mortality_features], mort_demo["MORTSTAT"])

    return age_model, mortality_model, used_demo_fallback


metadata = load_json(PROCESSED_DIR / "metadata.json")
age_features = metadata["age_features"]
mortality_features = metadata["mortality_features"]
categorical_age_features = metadata.get("categorical_age_features", [])
categorical_mortality_features = metadata.get("categorical_mortality_features", [])
demo_profiles = pd.read_csv(DEMO_DIR / "demo_profiles.csv")

age_model, mortality_model, used_demo_fallback = load_or_train_models()

st.title("🧬 Clinical Aging Twin")
st.caption("Clinical age and mortality modelling powered by TabPFN-3.5")

if used_demo_fallback:
    st.info(
        "The saved full-fit model records were not available to this account, "
        "so this session is using a smaller reproducible TabPFN-3.5 demo fit."
    )

st.markdown(
    """
    **Explore how a clinical profile maps to an estimated clinical age.**  
    The age gap is a descriptive proxy for accelerated clinical aging; it is not a causal or clinical diagnosis.
    """
)


def human_label(feature: str) -> str:
    """Return the public-facing label while keeping NHANES codes internal."""
    if feature in FEATURE_UI:
        return FEATURE_UI[feature]["label"]
    return feature.replace("_", " ").title()


def format_value(feature: str, value) -> str:
    """Format a patient value for display in the comparison table."""
    if pd.isna(value):
        return "Not available"

    if feature == "RIDAGEYR":
        return f"{float(value):.0f} years"
    if feature == "RIAGENDR":
        return SEX_LABELS.get(int(round(float(value))), "Not reported")
    if feature == "SMQ040":
        return SMOKING_LABELS.get(int(round(float(value))), "Not reported")
    if feature == "PAQ180":
        return PAQ180_LABELS.get(int(round(float(value))), "Not reported")

    meta = FEATURE_UI.get(feature, {})
    unit = meta.get("unit")
    numeric = float(value)
    if feature in {"BMXBMI", "BMXWAIST", "LBDSGLSI", "LBDTCSI", "LBDHDLSI", "LBDSTRSI", "LBXCRP", "LBDSCRSI", "LBDSALSI", "ALQ130"}:
        formatted = f"{numeric:.1f}"
    else:
        formatted = f"{numeric:.0f}"
    return f"{formatted} {unit}" if unit else formatted


def widget_label(feature: str) -> str:
    meta = FEATURE_UI.get(feature, {})
    label = meta.get("label", human_label(feature))
    unit = meta.get("unit")
    return f"{label} ({unit})" if unit else label


def widget_help(feature: str) -> str | None:
    return FEATURE_UI.get(feature, {}).get("help")


def safe_float(value, default=0.0):
    value = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(value) else float(value)


def predict_profile(row: pd.Series):
    age_x = pd.DataFrame([row.reindex(age_features)])
    mort_x = pd.DataFrame([row.reindex(mortality_features)])

    age_x = apply_categorical_dtypes(age_x, categorical_age_features)
    mort_x = apply_categorical_dtypes(mort_x, categorical_mortality_features)

    predicted_age = float(age_model.predict(age_x)[0])
    mortality_probability = float(mortality_model.predict_proba(mort_x)[0, 1])
    chronological_age = float(row["RIDAGEYR"])
    acceleration = predicted_age - chronological_age
    return predicted_age, mortality_probability, acceleration


def render_metrics(row: pd.Series):
    predicted_age, mortality_probability, acceleration = predict_profile(row)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chronological age", f"{float(row['RIDAGEYR']):.0f}")
    c2.metric("Clinical age", f"{predicted_age:.1f}")
    c3.metric("Age acceleration", f"{acceleration:+.1f} years")
    c4.metric("Mortality model score", f"{mortality_probability:.1%}")
    return predicted_age, mortality_probability, acceleration


def comparison_table(row_a: pd.Series, row_b: pd.Series, name_a: str, name_b: str):
    """Render profile comparison as one readable table rather than side-by-side cards."""
    age_a, mort_a, acc_a = predict_profile(row_a)
    age_b, mort_b, acc_b = predict_profile(row_b)

    rows = [
        {
            "Measure": "Chronological age",
            name_a: f"{float(row_a['RIDAGEYR']):.0f} years",
            name_b: f"{float(row_b['RIDAGEYR']):.0f} years",
            "Difference (B − A)": f"{float(row_b['RIDAGEYR']) - float(row_a['RIDAGEYR']):+.0f} years",
        },
        {
            "Measure": "Estimated clinical age",
            name_a: f"{age_a:.1f} years",
            name_b: f"{age_b:.1f} years",
            "Difference (B − A)": f"{age_b - age_a:+.1f} years",
        },
        {
            "Measure": "Clinical age acceleration",
            name_a: f"{acc_a:+.1f} years",
            name_b: f"{acc_b:+.1f} years",
            "Difference (B − A)": f"{acc_b - acc_a:+.1f} years",
        },
        {
            "Measure": "Mortality model score",
            name_a: f"{mort_a:.1%}",
            name_b: f"{mort_b:.1%}",
            "Difference (B − A)": f"{mort_b - mort_a:+.1%}",
        },
    ]

    comparison = pd.DataFrame(rows)
    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Measure": st.column_config.TextColumn("Measure", width="medium"),
            name_a: st.column_config.TextColumn(name_a, width="medium"),
            name_b: st.column_config.TextColumn(name_b, width="medium"),
            "Difference (B − A)": st.column_config.TextColumn("Difference (B − A)", width="medium"),
        },
    )


# -------------------------
# Single-profile explorer
# -------------------------
tab1, tab2 = st.tabs(["Explore one profile", "Compare two profiles"])

with tab1:
    profile_name = st.selectbox(
        "Starting profile",
        demo_profiles["profile_id"].tolist(),
        key="single_profile",
    )
    base = demo_profiles.loc[demo_profiles["profile_id"] == profile_name].iloc[0].copy()

    st.subheader("Clinical profile")
    st.caption("The names below are written for a general audience; NHANES variable codes remain internal to the model.")

    current_row = base.copy()

    # Demographics
    with st.expander("Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age_value = st.number_input(
                widget_label("RIDAGEYR"),
                min_value=20,
                max_value=84,
                value=int(round(safe_float(base["RIDAGEYR"], 60))),
                step=1,
                help=widget_help("RIDAGEYR"),
            )
            current_row["RIDAGEYR"] = age_value

        with col2:
            if "RIAGENDR" in current_row.index:
                base_sex = int(round(safe_float(base["RIAGENDR"], 1)))
                sex_options = [1, 2]
                current_row["RIAGENDR"] = st.selectbox(
                    widget_label("RIAGENDR"),
                    options=sex_options,
                    index=sex_options.index(base_sex) if base_sex in sex_options else 0,
                    format_func=lambda x: SEX_LABELS[x],
                    help=widget_help("RIAGENDR"),
                )

    # Body and vital signs
    with st.expander("Body measurements and vital signs", expanded=True):
        left, right = st.columns(2)
        for i, feature in enumerate(EDITABLE_NUMERIC[:5]):
            if feature not in current_row.index:
                continue
            value = safe_float(base[feature])
            container = left if i < 3 else right
            with container:
                unit = FEATURE_UI[feature].get("unit")
                step = 0.1 if feature in {"BMXBMI", "BMXWAIST"} else 1.0
                current_row[feature] = st.number_input(
                    widget_label(feature),
                    value=value,
                    step=step,
                    key=f"single_{feature}",
                    help=widget_help(feature),
                    format="%.1f" if step < 1 else "%.1f",
                )

    # Blood tests
    with st.expander("Blood tests", expanded=True):
        left, right = st.columns(2)
        lab_features = EDITABLE_NUMERIC[5:13]
        for i, feature in enumerate(lab_features):
            if feature not in current_row.index:
                continue
            value = safe_float(base[feature])
            container = left if i < 4 else right
            with container:
                current_row[feature] = st.number_input(
                    widget_label(feature),
                    value=value,
                    step=0.1,
                    key=f"single_{feature}",
                    help=widget_help(feature),
                    format="%.2f",
                )

    # Lifestyle
    with st.expander("Lifestyle", expanded=True):
        left, right = st.columns(2)

        with left:
            if "SMQ040" in current_row.index:
                base_smoking = int(round(safe_float(base["SMQ040"], 9)))
                smoking_options = list(SMOKING_LABELS.keys())
                if base_smoking not in smoking_options:
                    smoking_options.insert(0, base_smoking)
                current_row["SMQ040"] = st.selectbox(
                    widget_label("SMQ040"),
                    options=smoking_options,
                    index=smoking_options.index(base_smoking),
                    format_func=lambda x: SMOKING_LABELS.get(x, "Not reported"),
                    help=widget_help("SMQ040"),
                )

            if "PAQ180" in current_row.index:
                raw_value = int(round(safe_float(base["PAQ180"], 2)))
                # PAQ180 is an ordinal 1–4 NHANES category, not a measure of hours.
                # Codes 7/9 are refusal / don't know and are not exposed as editable choices.
                if raw_value not in (1, 2, 3, 4):
                    raw_value = 2
                current_row["PAQ180"] = st.selectbox(
                    widget_label("PAQ180"),
                    options=[1, 2, 3, 4],
                    index=[1, 2, 3, 4].index(raw_value),
                    format_func=lambda x: PAQ180_LABELS[x],
                    key="single_PAQ180",
                    help=widget_help("PAQ180"),
                )

        with right:
            if "ALQ130" in current_row.index:
                value = safe_float(base["ALQ130"])
                current_row["ALQ130"] = st.number_input(
                    widget_label("ALQ130"),
                    value=value,
                    step=0.1,
                    key="single_ALQ130",
                    help=widget_help("ALQ130"),
                    format="%.1f",
                )

    st.divider()
    render_metrics(current_row)

    st.caption(
        "What-if mode changes only the displayed variables. All other model inputs stay fixed to the selected demo profile; "
        "therefore changes are scenario-based and should not be interpreted causally."
    )

# -------------------------
# Compare two profiles
# -------------------------
with tab2:
    if len(demo_profiles) >= 2:
        st.subheader("Compare two clinical aging profiles")
        st.caption("Both profiles are summarised in one table so differences can be read directly.")

        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox(
                "Profile A",
                demo_profiles["profile_id"].tolist(),
                index=0,
                key="compare_a",
            )
        with col2:
            p2 = st.selectbox(
                "Profile B",
                demo_profiles["profile_id"].tolist(),
                index=1,
                key="compare_b",
            )

        row_a = demo_profiles.loc[demo_profiles["profile_id"] == p1].iloc[0]
        row_b = demo_profiles.loc[demo_profiles["profile_id"] == p2].iloc[0]

        comparison_table(row_a, row_b, p1, p2)

        # Clinical measurements that are most intuitive to a general audience.
        comparison_features = [
            "RIAGENDR",
            "RIDAGEYR",
            "BMXBMI",
            "BMXWAIST",
            "BPXSAR",
            "BPXDAR",
            "BPXPLS",
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

        clinical_rows = []
        for feature in comparison_features:
            if feature not in row_a.index or feature not in row_b.index:
                continue
            clinical_rows.append(
                {
                    "Clinical measure": human_label(feature),
                    p1: format_value(feature, row_a[feature]),
                    p2: format_value(feature, row_b[feature]),
                }
            )

        st.subheader("Clinical measurements")
        st.dataframe(
            pd.DataFrame(clinical_rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Clinical measure": st.column_config.TextColumn("Clinical measure", width="large"),
                p1: st.column_config.TextColumn(p1, width="medium"),
                p2: st.column_config.TextColumn(p2, width="medium"),
            },
        )

        st.caption(
            "The mortality value is a model score, not a clinical risk assessment. Clinical age acceleration is a descriptive model-derived measure."
        )

st.divider()
st.caption(
    "Research demonstration only — not a medical diagnostic tool or individual clinical risk assessment."
)
