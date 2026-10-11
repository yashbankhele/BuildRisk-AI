"""Train delay-risk, delay-days, and cost-overrun models."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_generator import generate_projects

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

CATEGORICAL = ["project_type", "region"]
NUMERIC = [
    "budget_inr", "planned_duration_days", "elapsed_days", "progress_pct",
    "cost_spent_pct", "labor_productivity_index", "material_availability_pct",
    "equipment_availability_pct", "weather_risk_0_10", "change_orders_count",
    "rfi_backlog_count", "safety_incidents_count", "contractor_experience_years",
    "complexity_1_5"
]
FEATURES = CATEGORICAL + NUMERIC

def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    data_path = DATA_DIR / "generated_projects.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_projects()
        df.to_csv(data_path, index=False)

    X = df[FEATURES]
    y_delay_risk = df["delay_risk"]
    y_delay_days = df["actual_delay_days"]
    y_cost_overrun = df["cost_overrun"]

    X_train, X_test, yd_train, yd_test, days_train, days_test, yc_train, yc_test = train_test_split(
        X, y_delay_risk, y_delay_days, y_cost_overrun,
        test_size=0.2, random_state=42, stratify=y_delay_risk
    )
    delay_model = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=220, min_samples_leaf=3,
                                         class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    days_model = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=220, min_samples_leaf=3,
                                        random_state=42, n_jobs=-1))
    ])
    cost_model = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=220, min_samples_leaf=3,
                                         class_weight="balanced", random_state=43, n_jobs=-1))
    ])

    delay_model.fit(X_train, yd_train)
    days_model.fit(X_train, days_train)
    cost_model.fit(X_train, yc_train)

    delay_pred = delay_model.predict(X_test)
    days_pred = days_model.predict(X_test)
    cost_pred = cost_model.predict(X_test)
    try:
        cost_auc = roc_auc_score(yc_test, cost_model.predict_proba(X_test)[:, list(cost_model.classes_).index(1)])
    except Exception:
        cost_auc = None

    metrics = {
        "synthetic_data": True,
        "rows": int(len(df)),
        "test_rows": int(len(X_test)),
        "delay_risk_accuracy": round(float(accuracy_score(yd_test, delay_pred)), 4),
        "delay_risk_macro_f1": round(float(f1_score(yd_test, delay_pred, average="macro")), 4),
        "delay_days_mae": round(float(mean_absolute_error(days_test, days_pred)), 2),
        "cost_overrun_accuracy": round(float(accuracy_score(yc_test, cost_pred)), 4),
        "cost_overrun_roc_auc": round(float(cost_auc), 4) if cost_auc is not None else None,
        "warning": "Metrics measure performance on synthetic holdout data only; they are not evidence of real-world EPC accuracy."
    }
    joblib.dump({"model": delay_model, "features": FEATURES, "classes": list(delay_model.classes_)},
                MODEL_DIR / "delay_risk.joblib")
    joblib.dump({"model": days_model, "features": FEATURES}, MODEL_DIR / "delay_days.joblib")
    joblib.dump({"model": cost_model, "features": FEATURES, "classes": list(cost_model.classes_)},
                MODEL_DIR / "cost_overrun.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nDelay risk report:\n", classification_report(yd_test, delay_pred, zero_division=0))
    print(f"\nSaved trained models to {MODEL_DIR}")

if __name__ == "__main__":
    main()
