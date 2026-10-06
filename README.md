# Clinical Aging Twin — TabPFN-3.5

Interactive clinical-aging explorer powered by **TabPFN-3.5** and trained on NHANES data.

## Hackathon idea

The project has three layers:

1. **Clinical age estimation** — TabPFN-3.5 regressor predicts chronological age from routine clinical/biochemical features.
2. **Clinical age acceleration** — predicted age minus chronological age is used as a descriptive proxy for accelerated clinical aging.
3. **Downstream mortality prediction** — a TabPFN-3.5 classifier predicts mortality status from the clinical profile.

The Streamlit app turns the models into an interactive demo where users can compare two profiles and perturb selected clinical variables. The scenario view is **not causal** and is not a medical decision-support tool.

## Why TabPFN-3.5 is central

The competition prototype uses TabPFN-3.5 for both regression and classification, with `thinking_effort="high"`. The repository also evaluates conventional baselines (dummy, linear/logistic, random forest / gradient boosting) so the contribution of TabPFN-3.5 is visible.

`tabpfn-client` supports an explicit `v3.5_default` model path and Thinking mode; fitted models can be saved as small JSON records, but those records are account-specific. See the official [tabpfn-client changelog](https://github.com/PriorLabs/tabpfn-client/blob/main/CHANGELOG.md) and [README](https://github.com/PriorLabs/tabpfn-client/blob/main/README.md).

## 1. Environment

```bash
conda create -n clinical-aging python=3.11 -y
conda activate clinical-aging
pip install -r requirements.txt
```

Set your Prior Labs token:

```bash
export TABPFN_TOKEN="YOUR_TOKEN"
```

On Windows PowerShell:

```powershell
$env:TABPFN_TOKEN="YOUR_TOKEN"
```

## 2. Data

Place:

```text
data/raw/nhanes.csv
data/raw/variables_explained.csv
```

The latter is the supplied variable dictionary.

## 3. Run the notebooks

Run in order:

```text
01_data_preparation.ipynb
02_train_age.ipynb
03_train_mortality.ipynb
04_age_acceleration.ipynb
```

The notebooks create processed data, model metadata, evaluation tables, plots and (for the local account) fitted-model JSON references.

## 4. Run the app

```bash
streamlit run app.py
```

The app first tries to load the saved fitted-model records. If those records are unavailable to the current account, it can fall back to training a small demo model from `data/demo/` using the current account's token.

## Important reproducibility note

The Prior Labs client is a cloud-based service. Your NHANES rows are sent to the service when inference is executed, so only use data you are permitted to share. Fitted-model JSON references can only be loaded by the account that created the fit. See the official [client README](https://github.com/PriorLabs/tabpfn-client/blob/main/README.md).

## Scientific caveats

- `RIDAGEYR` is chronological age; this is **clinical/phenotypic age estimation**, not a directly measured biological age.
- The difference `predicted age - chronological age` is a descriptive proxy for accelerated clinical aging and should not be interpreted causally.
- NHANES age is top-coded at 85 in the source variable, so the age model should not be interpreted as estimating exact ages above that threshold.
- Mortality linkage variables are excluded from the mortality predictors to prevent leakage.
- The app is a research demonstration, not a medical diagnostic or risk-assessment tool.
