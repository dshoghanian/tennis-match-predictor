# src/data.py
"""Load, clean, and chronologically order ATP match files."""
from pathlib import Path
import pandas as pd

# Lower number = earlier in the tournament.
ROUND_ORDER = {
    "R128": 0, "R64": 1, "R32": 2, "R16": 3,
    "QF": 4, "SF": 5, "F": 6, "RR": 3, "BR": 5,
}


def load_matches(raw_dir, start_year=1991, end_year=2026):
    """Read and concatenate atp_matches_<year>.csv files that exist."""
    raw_dir = Path(raw_dir)
    frames = []
    for year in range(start_year, end_year + 1):
        path = raw_dir / f"atp_matches_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No match files found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def clean_matches(df):
    """Parse dates, drop unusable rows, sort chronologically."""
    df = df.copy()
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype("Int64").astype(str), format="%Y%m%d"
    )
    df = df.dropna(subset=["surface", "winner_id", "loser_id"])
    df["_round_order"] = df["round"].map(ROUND_ORDER).fillna(3).astype(int)
    df = df.dropna(subset=["tourney_date"])
    df = df.sort_values(["tourney_date", "_round_order"]).reset_index(drop=True)
    return df.drop(columns="_round_order")
