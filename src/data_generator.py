"""Generate clearly labelled synthetic construction-project snapshots for demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_TYPES = ["Road", "Commercial Building", "Industrial", "Water Infrastructure", "Residential"]
REGIONS = ["West", "North", "South", "East", "Central"]

def generate_projects(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    project_type = rng.choice(PROJECT_TYPES, n)
    region = rng.choice(REGIONS, n)
    budget = np.exp(rng.normal(np.log(8_00_00_000), 0.85, n)).clip(2_00_00_000, 80_00_00_000)
    planned_days = rng.integers(120, 1100, n)
    elapsed_days = (planned_days * rng.uniform(0.15, 0.85, n)).astype(int)
    progress = (elapsed_days / planned_days * rng.normal(1.0, 0.16, n) * 100).clip(2, 98)
    cost_spent_pct = (progress + rng.normal(4, 10, n)).clip(1, 115)
    labor_productivity = rng.normal(0.90, 0.18, n).clip(0.35, 1.35)
    material_availability = rng.uniform(45, 100, n)
    equipment_availability = rng.uniform(50, 100, n)
    weather_risk = rng.integers(0, 11, n)
    change_orders = rng.poisson(3, n).clip(0, 15)
    rfi_backlog = rng.poisson(7, n).clip(0, 35)
    safety_incidents = rng.poisson(1.2, n).clip(0, 8)
    contractor_experience = rng.integers(1, 31, n)
    complexity = rng.integers(1, 6, n)

    # Latent relationship creates plausible correlations for a demo, not real-world ground truth.
    schedule_pressure = (
        (elapsed_days / planned_days - progress / 100) * 100
        + (1 - labor_productivity) * 28
        + (100 - material_availability) * 0.16
        + (100 - equipment_availability) * 0.10
        + weather_risk * 1.15
        + change_orders * 1.35
        + rfi_backlog * 0.35
        + safety_incidents * 1.8
        + complexity * 1.4
        - contractor_experience * 0.12
        + rng.normal(0, 5, n)
    )
    delay_days = np.maximum(0, (schedule_pressure - 8) * 1.8 + rng.normal(0, 18, n)).round().astype(int)
    overrun_pct = (
        (cost_spent_pct - progress) * 0.28
        + (1 - labor_productivity) * 18
        + (100 - material_availability) * 0.10
        + change_orders * 0.75
        + weather_risk * 0.35
        + complexity * 0.9
        + rng.normal(0, 3.5, n)
    ).clip(-8, 55).round(2)

    df = pd.DataFrame({
        "project_type": project_type, "region": region,
        "budget_inr": budget.round(0).astype(int),
        "planned_duration_days": planned_days, "elapsed_days": elapsed_days,
        "progress_pct": progress.round(1), "cost_spent_pct": cost_spent_pct.round(1),
        "labor_productivity_index": labor_productivity.round(2),
        "material_availability_pct": material_availability.round(1),
        "equipment_availability_pct": equipment_availability.round(1),
        "weather_risk_0_10": weather_risk, "change_orders_count": change_orders,
        "rfi_backlog_count": rfi_backlog, "safety_incidents_count": safety_incidents,
        "contractor_experience_years": contractor_experience, "complexity_1_5": complexity,
        "actual_delay_days": delay_days, "actual_cost_overrun_pct": overrun_pct,
    })
    df["delay_risk"] = pd.cut(df["actual_delay_days"], bins=[-1, 14, 45, np.inf],
                              labels=["Low", "Medium", "High"]).astype(str)
    df["cost_overrun"] = (df["actual_cost_overrun_pct"] >= 10).astype(int)
    return df

def main():
    out = Path("data/generated_projects.csv")
    out.parent.mkdir(exist_ok=True)
    df = generate_projects()
    df.to_csv(out, index=False)
    print(f"Created {len(df):,} synthetic project snapshots: {out}")
    print("Reminder: this is simulated data, not real EPC project history.")

if __name__ == "__main__":
    main()
