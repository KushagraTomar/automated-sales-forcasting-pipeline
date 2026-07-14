from pathlib import Path
from typing import Optional

import pandas as pd


def load_sales_data(path: Optional[str] = None):
    data_path = Path(path or "data/raw/train.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df


def simulate_weekly_slice(df: pd.DataFrame, current_date: str):
    threshold = pd.Timestamp(current_date)
    return df[df["date"] <= threshold].copy()

