# src/features.py
"""Build a leakage-safe, symmetric training table.

Each match becomes one row from a randomized "player A vs player B" view.
target a_won = 1 if A is the actual winner. All numeric features are differences
(A minus B) of PRE-match quantities, so column order carries no signal.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "height_diff",
    "recent_winpct_diff",
    "h2h_diff",
    "days_since_last_diff",
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

    if "winner_recent_winpct" in df.columns:
        a_f, b_f = pick("winner_recent_winpct", "loser_recent_winpct")
        out["recent_winpct_diff"] = a_f - b_f
        a_h, b_h = pick("winner_h2h_wins", "loser_h2h_wins")
        out["h2h_diff"] = a_h - b_h
        a_d, b_d = pick("winner_days_since_last", "loser_days_since_last")
        out["days_since_last_diff"] = a_d - b_d
    else:
        for c in ("recent_winpct_diff", "h2h_diff", "days_since_last_diff"):
            out[c] = 0.0

    out["a_won"] = a_is_winner.astype(int)
    return out


def add_history_features(df, window=10):
    """Annotate each match with PRE-match rolling history for both players.

    Adds winner/loser columns: recent_winpct, h2h_wins, days_since_last.
    All values reflect only matches BEFORE the current row (chronological).
    """
    last_results = defaultdict(lambda: deque(maxlen=window))  # 1=win,0=loss
    h2h = defaultdict(int)        # (player, opponent) -> wins by player
    last_date = {}                # player -> last match date

    new = {c: [] for c in (
        "winner_recent_winpct", "loser_recent_winpct",
        "winner_h2h_wins", "loser_h2h_wins",
        "winner_days_since_last", "loser_days_since_last",
    )}

    def winpct(pid):
        d = last_results[pid]
        return sum(d) / len(d) if d else 0.5

    def days_since(pid, date):
        return (date - last_date[pid]).days if pid in last_date else 365

    for row in df.itertuples(index=False):
        w, l, date = row.winner_id, row.loser_id, row.tourney_date
        new["winner_recent_winpct"].append(winpct(w))
        new["loser_recent_winpct"].append(winpct(l))
        new["winner_h2h_wins"].append(h2h[(w, l)])
        new["loser_h2h_wins"].append(h2h[(l, w)])
        new["winner_days_since_last"].append(days_since(w, date))
        new["loser_days_since_last"].append(days_since(l, date))

        # Update AFTER snapshotting.
        last_results[w].append(1)
        last_results[l].append(0)
        h2h[(w, l)] += 1
        last_date[w] = date
        last_date[l] = date

    out = df.copy()
    for c, vals in new.items():
        out[c] = vals
    return out
