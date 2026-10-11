# 🏗️ BuildRisk AI — Construction Delay & Cost Overrun Prediction

**An ML-powered construction project dashboard for delay-risk classification, delay-days estimation, and cost-overrun early warning.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Data](https://img.shields.io/badge/data-synthetic-lightgrey)

> ⚠️ **Important:** The included dataset is **synthetic** and is not from any real contractor. This application is a **portfolio prototype**, not a validated construction forecasting tool. Do not use its outputs for real project decisions.

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Problem Statement](#-problem-statement)
3. [Key Features](#-key-features)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [System Workflow](#-system-workflow)
7. [Dataset](#-dataset)
8. [Feature Dictionary](#-feature-dictionary)
9. [Prediction Targets](#-prediction-targets)
10. [Model Design](#-model-design)
11. [Evaluation](#-evaluation)
12. [Installation & Setup](#-installation--setup)
13. [Usage](#-usage)
14. [Dashboard Walkthrough](#-dashboard-walkthrough)
15. [Testing](#-testing)
16. [Limitations](#-limitations)
17. [Before Claiming This Works in a Real EPC Setting](#-before-claiming-this-works-in-a-real-epc-setting)
18. [Roadmap](#-roadmap)
19. [Contributing](#-contributing)
20. [Author](#-author)

---

## 🔍 Overview

**BuildRisk AI** is an end-to-end portfolio project built for the **construction / EPC (Engineering, Procurement & Construction) project-controls** domain. It combines:

- a reproducible **synthetic project dataset**,
- **three machine-learning models** (one classifier for delay risk, one regressor for delay days, one classifier for cost overrun),
- an **interactive Streamlit dashboard** with portfolio KPIs and project-level charts,
- a **rule-based review checklist** that turns risk signals into suggested actions,
- a **train/test evaluation report**, and
- **basic automated tests** (pytest).

The goal is to demonstrate the full data-science workflow: data generation → feature engineering → model training → evaluation → deployment as a usable app.

---

## 🎯 Problem Statement

Construction projects frequently suffer from **schedule delays** and **cost overruns**. By the time a project manager notices the problem in a monthly report, the options to recover are limited.

Project-controls teams need an **early-warning signal** based on information already available mid-project, such as:

- How much physical progress has been achieved vs. time elapsed?
- How much budget has been spent vs. progress made?
- Are change orders, RFIs (Requests for Information) and safety incidents piling up?
- Is labor productivity, material/equipment availability or weather putting the schedule at risk?

BuildRisk AI explores how a machine-learning model could convert these snapshot indicators into:

1. a **delay-risk category** (Low / Medium / High),
2. an **estimate of additional delay days**, and
3. a **probability of a final cost overrun ≥ 10%**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Delay-risk classification** | Predicts Low / Medium / High delay risk for a project. |
| **Delay-days estimation** | Regression model estimating additional delay in days. |
| **Cost-overrun probability** | Probability that the final cost overrun will be ≥ 10%. |
| **Portfolio KPIs** | Summary metrics across all projects in the dataset. |
| **Project-level charts** | Interactive Plotly visualisations for exploring individual projects and segments. |
| **Project input form** | Enter a project's snapshot values manually and get predictions. |
| **Rule-based review checklist** | Simple, transparent rules that flag issues for a project controls review. |
| **CSV export** | Download the filtered portfolio data. |
| **Evaluation report** | Metrics saved to `models/metrics.json` after training. |
| **Basic tests** | pytest suite for sanity checks. |
| **Auto-training on first launch** | If model files are missing, the app trains them automatically. |

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Machine learning | scikit-learn (Random Forest, pipelines, imputers, one-hot encoding) |
| Model persistence | joblib |
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Testing | pytest |

---

## 📂 Project Structure

> The layout below follows the commands used in this README (`src.data_generator`, `src.train`, `app.py`). Adjust the tree to match your actual repository.

```text
BuildRisk-AI/
├── app.py                      # Streamlit dashboard entry point
├── requirements.txt            # Python dependencies
├── README.md
├── data/
│   └── generated_projects.csv  # Synthetic dataset (created by the generator)
├── models/
│   ├── *.joblib                # Three trained model files (created by training)
│   └── metrics.json            # Evaluation metrics (created by training)
├── src/
│   ├── data_generator.py       # Synthetic project data generator
│   └── train.py                # Model training and evaluation script
└── tests/                      # pytest tests
```

---

## 🔄 System Workflow

```text
┌────────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│ Synthetic data     │ ──▶ │ Preprocessing      │ ──▶ │ Model training       │
│ generator          │     │ (impute + encode)  │     │ (3 Random Forests)   │
└────────────────────┘     └────────────────────┘     └──────────┬───────────┘
                                                                 │
                                                                 ▼
┌────────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│ Dashboard: KPIs,   │ ◀── │ Predictions +      │ ◀── │ Saved models +       │
│ charts, form,      │     │ rule-based review  │     │ metrics.json         │
│ CSV export         │     │ checklist          │     │ (joblib)             │
└────────────────────┘     └────────────────────┘     └──────────────────────┘
```

1. **Generate** a reproducible synthetic dataset of construction projects.
2. **Preprocess** features: median imputation for numeric columns; most-frequent imputation + one-hot encoding for categorical columns.
3. **Train** three Random Forest models with a fixed 80/20 holdout split.
4. **Evaluate** and write metrics to `models/metrics.json`.
5. **Serve** predictions through the Streamlit dashboard.

---

## 🗃️ Dataset

### Dataset options

1. **Start here — generated synthetic dataset.** Reproducible and safe for a portfolio demo. Generate it with:
   ```bash
   python -m src.data_generator
   ```
   This writes `data/generated_projects.csv`.

2. **Indian infrastructure project data.** Investigate MoSPI's monthly *Flash Report on Central Sector Projects* for cost and time overruns. A public example project exploring this approach is [InfraForecast](https://github.com/rivcoder/InfraForecast). Confirm source permissions, definitions and data-extraction quality before using.

3. **US public construction contract dataset.** The [DoD construction contract dataset paper and data description](https://pmc.ncbi.nlm.nih.gov/articles/PMC7452689/) describes more than 132,000 projects with contract attributes and cost/schedule information. Check availability, fields, licensing and fit before modeling.

4. **NYC Open Data project-phase records.** Explore records for schedule and cost overrun analysis; verify the current table, schema and usage terms before importing.

### ⚠️ Data caution

Do **not** combine sources until you have aligned:

- definitions,
- currency,
- project phase,
- reporting date, and
- outcome labels.

A project-level dataset **without time-stamped snapshots** may be suitable for *retrospective* analysis but **not** for an *early-warning* model.

---

## 📖 Feature Dictionary

These are the features available **at the reporting snapshot** and used as model inputs:

| Feature | Type | Meaning |
|---|---|---|
| Project type | Categorical | Category of construction project (e.g., building, road, industrial). |
| Region | Categorical | Geographic region of the project. |
| Approved budget | Numeric | Sanctioned project budget. |
| Planned duration | Numeric | Planned project length in days. |
| Elapsed days | Numeric | Days passed since project start at the snapshot. |
| Physical progress | Numeric | Percentage of physical work completed. |
| Budget spent | Numeric | Cost incurred so far. |
| Labor productivity | Numeric | Productivity indicator for the workforce. |
| Material / equipment availability | Numeric | Indicator of resource availability on site. |
| Weather risk | Numeric | Weather-related risk indicator. |
| Change orders | Numeric | Number of change orders raised. |
| RFI backlog | Numeric | Open Requests for Information. |
| Safety incidents | Numeric | Count of safety incidents. |
| Contractor experience | Numeric | Contractor's experience indicator. |
| Complexity | Numeric/Categorical | Project complexity indicator. |

> Exact column names, units and value ranges are defined in `src/data_generator.py`.

**Leakage protection:** Final outcomes (`actual_delay_days`, `actual_cost_overrun_pct`, `delay_risk`, `cost_overrun`) are **never** passed as model inputs.

---

## 🎯 Prediction Targets

| Target | Task | Description |
|---|---|---|
| `delay_risk` | Multi-class classification | Low / Medium / High delay risk. |
| `actual_delay_days` | Regression | Additional delay in days. |
| `cost_overrun` | Binary classification | Whether the final cost overrun is ≥ 10% (`actual_cost_overrun_pct`). |

---

## 🧠 Model Design

| Model | Algorithm | Metrics |
|---|---|---|
| Delay risk | Random Forest Classifier | Accuracy, Macro-F1 |
| Delay days | Random Forest Regressor | Mean Absolute Error (days) |
| Cost overrun | Random Forest Classifier | Accuracy, ROC-AUC (where defined) |

**Preprocessing**
- Numeric missing values → **median imputation**
- Categorical missing values → **most-frequent imputation**
- Categorical encoding → **one-hot encoding**

**Validation**
- Fixed **80/20 holdout split** (starter setup).

**Why Random Forest?**
Robust on tabular data, handles non-linear interactions, needs little tuning, and gives feature importances that are easy to explain to non-technical stakeholders.

---

## 📊 Evaluation

After running `python -m src.train`, metrics are saved to `models/metrics.json`.

Fill in your own results after running the training:

| Model | Metric | Result |
|---|---|---|
| Delay risk | Accuracy | _fill after training_ |
| Delay risk | Macro-F1 | _fill after training_ |
| Delay days | MAE (days) | _fill after training_ |
| Cost overrun | Accuracy | _fill after training_ |
| Cost overrun | ROC-AUC | _fill after training_ |

> Because the data is synthetic and generated by known rules, high scores here **do not** imply real-world performance.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or newer
- pip
- Git

### Windows (PowerShell)

```powershell
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Generate the dataset
```bash
python -m src.data_generator
```
Creates `data/generated_projects.csv`.

### 2. Train the models
```bash
python -m src.train
```
Creates three joblib model files and `models/metrics.json`.

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Then open the URL shown in the terminal (usually `http://localhost:8501`).

> On first launch, the app trains models automatically if the model files are missing. For a clean and reproducible workflow, run the training command explicitly first.

---

## 🖥️ Dashboard Walkthrough

1. **Portfolio overview** — KPI cards and charts summarising all projects (risk distribution, delay and cost trends by project type/region).
2. **Filters** — slice the portfolio by project type, region or risk level.
3. **Project explorer** — drill into individual projects with interactive Plotly charts.
4. **Prediction form** — enter snapshot values for a project and get:
   - delay-risk category,
   - estimated delay days,
   - cost-overrun probability.
5. **Review checklist** — rule-based flags (e.g., high RFI backlog, low productivity vs. progress) to guide a project-controls review.
6. **Export** — download the filtered data as CSV.

> 📸 *Add screenshots here:* `docs/screenshots/overview.png`, `docs/screenshots/prediction_form.png`

---

## 🧪 Testing

```bash
pytest -q
```

The test suite provides basic checks for the data and model pipeline. Extend it with tests for data schema, leakage prevention and prediction ranges as the project grows.

---

## 🚧 Limitations

- **Synthetic data only** — patterns are generated by rules, so the model learns those rules, not real construction behaviour.
- **Single holdout split** — no cross-validation, no project-level or time-based split.
- **No calibration analysis** — predicted probabilities may not match real-world frequencies.
- **No baseline comparison** — results are not benchmarked against simple heuristics or other algorithms.
- **No confidence intervals** on metrics.
- **No domain validation** — features, thresholds and suggested actions have not been reviewed by project-controls experts.

---

## ✅ Before Claiming This Works in a Real EPC Setting

1. Replace synthetic data with **licensed historical project records**.
2. Ensure each record is a **dated project snapshot** and labels describe outcomes **after** that snapshot.
3. **Split evaluation by project and by time** to prevent leakage.
4. **Compare against baseline models**; report precision/recall, calibration, MAE and confidence intervals.
5. Have **project-controls experts** review features, thresholds and suggested actions.
6. **Monitor** missing data, drift, performance by project type and human overrides.

---

## 🗺️ Roadmap

- [ ] Add real/public dataset ingestion (MoSPI, DoD, NYC Open Data)
- [ ] Time-based and project-grouped cross-validation
- [ ] Baselines: logistic regression, gradient boosting (XGBoost / LightGBM)
- [ ] Probability calibration (Platt / isotonic)
- [ ] SHAP-based explanations in the dashboard
- [ ] Multi-snapshot (time-series) early-warning model
- [ ] Dockerfile and cloud deployment (Streamlit Community Cloud)
- [ ] CI pipeline with GitHub Actions running pytest
- [ ] Model monitoring and drift detection

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

Add a license of your choice (e.g., MIT) and place it in a `LICENSE` file.

---

## 👤 Author

**Yash Bankhele**
- GitHub: [@yashbankhele](https://github.com/yashbankhele)

---

### Suggested GitHub description
**BuildRisk AI: An ML-powered construction project dashboard for delay-risk classification, delay estimation, and cost-overrun early warning.**

### Suggested resume bullet (after you have run and verified it)
Built a Streamlit-based construction risk analytics prototype using Python and scikit-learn to classify delay risk, estimate delay days, and assess cost-overrun probability, with model evaluation and interactive project KPIs. Dataset is synthetic.
