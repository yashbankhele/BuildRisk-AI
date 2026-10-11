# BuildRisk AI — Construction Delay & Cost Overrun Prediction

An end-to-end portfolio project for construction/EPC project controls. It combines a synthetic project dataset, machine-learning models, an interactive Streamlit dashboard, risk indicators, and reproducible evaluation.

> **Important:** the included data is synthetic and is not from any real contractor. The application is a portfolio prototype, not a validated construction forecasting tool.

## What it does
- Classifies delay risk as Low / Medium / High.
- Estimates additional delay days.
- Estimates the probability of a final cost overrun of at least 10%.
- Displays portfolio KPIs and project-level charts.
- Offers a project input form and a rule-based review checklist.
- Exports filtered portfolio data to CSV.
- Includes a train/test evaluation report and basic tests.

## Tech stack
Python · Pandas · NumPy · scikit-learn · Streamlit · Plotly · joblib · pytest

## Dataset options
1. **Start here: generated synthetic dataset.** Reproducible and safe for a portfolio demo. Use `python -m src.data_generator` to generate `data/generated_projects.csv`.
2. **Indian infrastructure project data:** investigate MoSPI's monthly *Flash Report on Central Sector Projects* for cost and time overruns. A public example project exploring that approach is [InfraForecast](https://github.com/rivcoder/InfraForecast). Confirm source permissions, definitions and data extraction quality before using.
3. **US public construction contract dataset:** the [DoD construction contract dataset paper and data description](https://pmc.ncbi.nlm.nih.gov/articles/PMC7452689/) describes more than 132,000 projects with contract attributes and cost/schedule information. Check availability, fields, licensing and fit before modeling.
4. **NYC Open Data project-phase records:** explore records for schedule and cost overrun analysis; verify the current table, schema and usage terms before importing.

Do not combine sources until you have aligned definitions, currency, project phase, reporting date, and outcome labels. A project-level dataset without time-stamped snapshots may be suitable for retrospective analysis but not for an early-warning model.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Generate data and train models
```powershell
python -m src.data_generator
python -m src.train
```
The generator writes `data/generated_projects.csv`; training writes three joblib model files and `models/metrics.json`.

## Launch dashboard
```powershell
streamlit run app.py
```
On first launch, the app trains models automatically if the model files are missing. For a clean and reproducible workflow, run the training command explicitly first.

## Run tests
```powershell
pytest -q
```

## Model design
- Delay risk: Random Forest classifier, evaluated using accuracy and macro-F1.
- Delay days: Random Forest regressor, evaluated using mean absolute error in days.
- Cost overrun: Random Forest classifier, evaluated using accuracy and ROC-AUC where defined.
- Numeric missing values are median-imputed; categorical values are most-frequent-imputed and one-hot encoded.
- A fixed 80/20 holdout split is used for this starter.

## Features used at the reporting snapshot
Project type, region, approved budget, planned duration, elapsed days, physical progress, budget spent, labor productivity, material/equipment availability, weather risk, change orders, RFI backlog, safety incidents, contractor experience, and complexity.

Final outcomes (`actual_delay_days`, `actual_cost_overrun_pct`, `delay_risk`, `cost_overrun`) are never passed as model inputs.

## Before claiming this works in a real EPC setting
1. Replace synthetic data with licensed historical project records.
2. Ensure each record is a dated project snapshot and labels describe outcomes after that snapshot.
3. Split evaluation by project and time to prevent leakage.
4. Compare against baseline models; report precision/recall, calibration, MAE and confidence intervals.
5. Have project controls experts review features, thresholds and suggested actions.
6. Monitor missing data, drift, performance by project type and human overrides.

## Suggested GitHub description
**BuildRisk AI: An ML-powered construction project dashboard for delay-risk classification, delay estimation, and cost-overrun early warning.**

## Suggested resume bullet (after you have run and verified it)
Built a Streamlit-based construction risk analytics prototype using Python and scikit-learn to classify delay risk, estimate delay days, and assess cost-overrun probability, with model evaluation and interactive project KPIs. Dataset is synthetic.
