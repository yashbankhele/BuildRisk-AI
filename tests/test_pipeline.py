from src.data_generator import generate_projects
from src.train import FEATURES

def test_generator_is_reproducible():
    a = generate_projects(100, seed=7)
    b = generate_projects(100, seed=7)
    assert a.equals(b)

def test_schema_and_ranges():
    df = generate_projects(200, seed=9)
    assert set(FEATURES).issubset(df.columns)
    assert df["progress_pct"].between(0, 100).all()
    assert df["actual_delay_days"].ge(0).all()
    assert set(df["delay_risk"].unique()).issubset({"Low", "Medium", "High"})
    assert set(df["cost_overrun"].unique()).issubset({0, 1})
