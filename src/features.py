# src/features.py
"""Build a leakage-safe, symmetric training table.

Each match becomes one row from a randomized "player A vs player B" view.
target a_won = 1 if A is the actual winner. All numeric features are differences
(A minus B) of PRE-match quantities, so column order carries no signal.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "height_diff",
]


def build_features(df, seed=42):
    """Return a DataFrame with FEATURE_COLUMNS + 'a_won' + 'tourney_date'.

    df must already carry the Elo `*_before` snapshot columns (see elo.py).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    # a_is_winner True => player A is the actual winner.
    a_is_winner = rng.integers(0, 2, size=n).astype(bool)

    def pick(win_col, lose_col):
        """Return (A_values, B_values) honoring the random A/B assignment."""
        w = df[win_col].to_numpy(dtype=float)
        l = df[lose_col].to_numpy(dtype=float)
        a = np.where(a_is_winner, w, l)
        b = np.where(a_is_winner, l, w)
        return a, b

    out = pd.DataFrame({"tourney_date": df["tourney_date"].values})

    a_elo, b_elo = pick("winner_elo_before", "loser_elo_before")
    out["elo_diff"] = a_elo - b_elo

    a_se, b_se = pick("winner_surface_elo_before", "loser_surface_elo_before")
    out["surface_elo_diff"] = a_se - b_se

    a_rk, b_rk = pick("winner_rank", "loser_rank")
    out["rank_diff"] = a_rk - b_rk  # negative => A is higher-ranked

    a_rp, b_rp = pick("winner_rank_points", "loser_rank_points")
    out["rank_points_diff"] = a_rp - b_rp

    a_age, b_age = pick("winner_age", "loser_age")
    out["age_diff"] = a_age - b_age

    a_ht, b_ht = pick("winner_ht", "loser_ht")
    out["height_diff"] = a_ht - b_ht

    out["a_won"] = a_is_winner.astype(int)
    return out
