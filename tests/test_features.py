# tests/test_features.py
import numpy as np
import pandas as pd
from src.features import build_features, FEATURE_COLUMNS


def _elo_annotated_df():
    # Two matches; provide the columns build_features consumes.
    return pd.DataFrame({
        "tourney_date": pd.to_datetime(["2020-01-01", "2020-01-08"]),
        "surface": ["Hard", "Clay"],
        "winner_id": [1, 1], "loser_id": [2, 3],
        "winner_elo_before": [1600.0, 1620.0],
        "loser_elo_before": [1500.0, 1510.0],
        "winner_surface_elo_before": [1550.0, 1500.0],
        "loser_surface_elo_before": [1500.0, 1505.0],
        "winner_rank": [10, 9], "loser_rank": [20, 30],
        "winner_rank_points": [3000, 3200], "loser_rank_points": [1500, 900],
        "winner_age": [25.0, 25.1], "loser_age": [28.0, 30.0],
        "winner_ht": [185.0, 185.0], "loser_ht": [190.0, 188.0],
    })


def test_target_is_balanced_label():
    feats = build_features(_elo_annotated_df(), seed=0)
    assert set(feats["a_won"].unique()).issubset({0, 1})


def test_elo_diff_sign_matches_winner_perspective():
    feats = build_features(_elo_annotated_df(), seed=0)
    # When a_won==1, player A is the winner, so a's elo_before - b's should be +100.
    won = feats[feats["a_won"] == 1].iloc[0]
    assert won["elo_diff"] == 100.0
    lost = feats[feats["a_won"] == 0].iloc[0]
    assert lost["elo_diff"] == -100.0


def test_no_outcome_columns_leak_into_features():
    feats = build_features(_elo_annotated_df(), seed=0)
    banned = {"winner_id", "loser_id", "w_ace", "l_ace", "score"}
    assert banned.isdisjoint(set(FEATURE_COLUMNS))
