from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_generator import generate_projects
from src.train import FEATURES, CATEGORICAL, NUMERIC, main as train_main

st.set_page_config(page_title="BuildRisk AI", page_icon="🏗️", layout="wide")
st.title("🏗️ BuildRisk AI")
st.caption("Construction Delay & Cost Overrun Early-Warning Dashboard")
st.info("Portfolio prototype using synthetic demonstration data. Predictions are not validated for real construction decisions.")

def ensure_models():
    needed = [ROOT / "models" / "delay_risk.joblib", ROOT / "models" / "delay_days.joblib",
              ROOT / "models" / "cost_overrun.joblib", ROOT / "models" / "metrics.json"]
    if not all(p.exists() for p in needed):
        with st.spinner("Training demonstration models for first launch..."):
            train_main()

ensure_models()
delay_bundle = joblib.load(ROOT / "models" / "delay_risk.joblib")
days_bundle = joblib.load(ROOT / "models" / "delay_days.joblib")
cost_bundle = joblib.load(ROOT / "models" / "cost_overrun.joblib")
metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))

@st.cache_data
def load_data():
    p = ROOT / "data" / "generated_projects.csv"
    return pd.read_csv(p) if p.exists() else generate_projects()

df = load_data()
with st.sidebar:
    st.header("Portfolio controls")
    project_type_filter = st.multiselect("Project type", sorted(df["project_type"].unique()),
                                         default=sorted(df["project_type"].unique()))
    region_filter = st.multiselect("Region", sorted(df["region"].unique()),
                                   default=sorted(df["region"].unique()))
    st.caption("Filters apply to the demonstration portfolio.")
filtered = df[df["project_type"].isin(project_type_filter) & df["region"].isin(region_filter)]
if filtered.empty:
    st.warning("Select at least one project type and region.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projects in view", f"{len(filtered):,}")
c2.metric("High delay risk", f"{(filtered.delay_risk == 'High').mean():.1%}")
c3.metric("Cost overrun ≥ 10%", f"{filtered.cost_overrun.mean():.1%}")
c4.metric("Median budget", f"₹{filtered.budget_inr.median()/1e7:.2f} Cr")

tab1, tab2, tab3, tab4 = st.tabs(["Portfolio overview", "Predict a project", "Model evaluation", "Data & limitations"])
with tab1:
    a, b = st.columns(2)
    with a:
        risk_counts = filtered["delay_risk"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("Risk").reset_index(name="Projects")
        st.plotly_chart(px.bar(risk_counts, x="Risk", y="Projects", title="Delay risk distribution"), use_container_width=True)
    with b:
        st.plotly_chart(px.scatter(filtered, x="progress_pct", y="cost_spent_pct", color="delay_risk",
                                   hover_data=["project_type", "region", "budget_inr"],
                                   title="Progress vs. cost spent"), use_container_width=True)
    st.subheader("Project register")
    display_cols = ["project_type", "region", "budget_inr", "progress_pct", "cost_spent_pct",
                    "actual_delay_days", "actual_cost_overrun_pct", "delay_risk"]
    st.dataframe(filtered[display_cols].head(1000), use_container_width=True, hide_index=True)
    st.download_button("Download filtered projects CSV", filtered.to_csv(index=False).encode("utf-8"),
                       "buildrisk_filtered_projects.csv", "text/csv")

with tab2:
    st.write("Enter a project's current status snapshot. Use figures available on the reporting date.")
    with st.form("project_input"):
        col1, col2, col3 = st.columns(3)
        with col1:
            project_type = st.selectbox("Project type", ["Road", "Commercial Building", "Industrial", "Water Infrastructure", "Residential"])
            region = st.selectbox("Region", ["West", "North", "South", "East", "Central"])
            budget = st.number_input("Approved budget (₹)", min_value=10_000_000, max_value=10_000_000_000,
                                     value=80_000_000, step=5_000_000)
            planned = st.number_input("Planned duration (days)", 30, 2000, 365)
            elapsed = st.number_input("Elapsed days", 0, 2000, 150)
        with col2:
            progress = st.slider("Physical progress (%)", 0.0, 100.0, 42.0)
            cost_spent = st.slider("Budget spent (%)", 0.0, 150.0, 45.0)
            productivity = st.slider("Labour productivity index (1.0 = plan)", 0.3, 1.5, 0.9, 0.05)
            materials = st.slider("Material availability (%)", 0.0, 100.0, 85.0)
            equipment = st.slider("Equipment availability (%)", 0.0, 100.0, 90.0)
        with col3:
            weather = st.slider("Weather risk (0–10)", 0, 10, 3)
            changes = st.number_input("Change orders", 0, 50, 2)
            rfis = st.number_input("Open RFIs", 0, 100, 5)
            incidents = st.number_input("Safety incidents to date", 0, 50, 0)
            experience = st.number_input("Contractor experience (years)", 0, 60, 8)
            complexity = st.slider("Project complexity (1–5)", 1, 5, 3)
        submitted = st.form_submit_button("Assess project risk", type="primary")
    if submitted:
        row = pd.DataFrame([{
            "project_type": project_type, "region": region, "budget_inr": budget,
            "planned_duration_days": planned, "elapsed_days": elapsed,
            "progress_pct": progress, "cost_spent_pct": cost_spent,
            "labor_productivity_index": productivity, "material_availability_pct": materials,
            "equipment_availability_pct": equipment, "weather_risk_0_10": weather,
            "change_orders_count": changes, "rfi_backlog_count": rfis,
            "safety_incidents_count": incidents, "contractor_experience_years": experience,
            "complexity_1_5": complexity
        }])[FEATURES]
        risk_model = delay_bundle["model"]
        risk = risk_model.predict(row)[0]
        probabilities = risk_model.predict_proba(row)[0]
        class_probs = dict(zip(risk_model.classes_, probabilities))
        delay_days = max(0, float(days_bundle["model"].predict(row)[0]))
        cost_model = cost_bundle["model"]
        cost_probs = dict(zip(cost_model.classes_, cost_model.predict_proba(row)[0]))
        overrun_probability = float(cost_probs.get(1, 0))
        st.subheader("Assessment result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Delay risk", str(risk))
        r2.metric("Estimated additional delay", f"{delay_days:.0f} days")
        r3.metric("Cost overrun probability", f"{overrun_probability:.1%}")
        st.write("Delay risk class probabilities")
        st.plotly_chart(px.bar(pd.DataFrame({"Risk": list(class_probs.keys()), "Probability": list(class_probs.values())}),
                               x="Risk", y="Probability", range_y=[0, 1]), use_container_width=True)
        st.markdown("**Suggested review checklist**")
        checks = []
        if progress < (elapsed / max(planned, 1) * 100) - 5: checks.append("Physical progress is behind elapsed-time pace; review the critical path and recovery plan.")
        if cost_spent > progress + 10: checks.append("Cost spent is materially ahead of physical progress; review earned value and remaining-cost estimate.")
        if materials < 75: checks.append("Material availability is low; confirm procurement and supplier commitments.")
        if productivity < 0.8: checks.append("Labour productivity is below the chosen planning baseline; investigate constraints.")
        if weather >= 7: checks.append("Weather risk is elevated; check schedule buffers and site-specific mitigation.")
        if rfis >= 10: checks.append("RFI backlog is high; review response owners and aging.")
        if changes >= 5: checks.append("Several change orders are present; review scope, approvals and cost/schedule impacts.")
        if not checks: checks.append("No configured rule was triggered. Continue routine monitoring and validate inputs.")
        for item in checks: st.write(f"- {item}")
        st.caption("Probabilities and estimates come from a model trained on synthetic data. Do not use for contractual commitments or safety-critical decisions.")

with tab3:
    st.subheader("Holdout metrics")
    st.json(metrics)
    st.caption("These scores are calculated on a held-out split of simulated data. They are not a benchmark of real project performance.")
    st.write("A production pilot should evaluate by project/time period, compare with simple baselines, calibrate probabilities and test on independently collected real project records.")

with tab4:
    st.subheader("Data dictionary")
    st.dataframe(pd.DataFrame({
        "Field": FEATURES,
        "Meaning": [
            "Construction category", "Broad region", "Approved budget in INR", "Baseline schedule duration",
            "Elapsed time at snapshot", "Physical progress complete", "Budget percentage spent",
            "Actual productivity divided by planned productivity", "Materials available as percentage of required",
            "Equipment available as percentage of required", "Weather disruption risk score", "Approved/pending change orders count",
            "Open requests for information", "Recorded safety incidents to date", "Contractor experience in years",
            "Complexity score from 1 to 5"
        ]
    }), use_container_width=True, hide_index=True)
    st.markdown("""
    **Data provenance:** generated synthetic data only. No real Ellora EPC records are included.

    **Target definitions:** delay-risk bands are Low (0–14 days), Medium (15–45 days), and High (over 45 days). Cost overrun is defined for this demonstration as simulated final overrun of at least 10%.

    **Known limitation:** because the targets are simulated from related input variables, good test scores mainly show that the model learned the simulator's rules. Real-world accuracy requires validated historical project snapshots and final outcomes. Use as decision support, not as a substitute for a project controls professional.
    """)
