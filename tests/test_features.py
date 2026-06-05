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
        "loser_elo_before": [1500.0, 1520.0],
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
    feats = build_features(_elo_annotated_df(), seed=42)
    # When a_won==1, player A is the winner, so a's elo_before - b's should be +100.
    won = feats[feats["a_won"] == 1].iloc[0]
    assert won["elo_diff"] == 100.0
    lost = feats[feats["a_won"] == 0].iloc[0]
    assert lost["elo_diff"] == -100.0


def test_no_outcome_columns_leak_into_features():
    feats = build_features(_elo_annotated_df(), seed=0)
    banned = {"winner_id", "loser_id", "w_ace", "l_ace", "score"}
    assert banned.isdisjoint(set(FEATURE_COLUMNS))


# Rolling history feature tests
from src.features import add_history_features


def test_recent_winpct_uses_only_past_matches():
    # Player 1 wins match0, then plays match1. At match1, player 1's pre-match
    # recent win% must be 1.0 (from match0) and must NOT include match1 itself.
    df = pd.DataFrame({
        "tourney_date": pd.to_datetime(["2020-01-01", "2020-01-08"]),
        "surface": ["Hard", "Hard"],
        "winner_id": [1, 1], "loser_id": [2, 3],
    })
    out = add_history_features(df, window=5)
    # match0 is player 1's first ever match => no history => 0.5 prior
    assert out.loc[0, "winner_recent_winpct"] == 0.5
    # match1 => player 1 has 1 prior win => 1.0
    assert out.loc[1, "winner_recent_winpct"] == 1.0


def test_h2h_counts_only_prior_meetings():
    df = pd.DataFrame({
        "tourney_date": pd.to_datetime(["2020-01-01", "2020-06-01"]),
        "surface": ["Hard", "Hard"],
        "winner_id": [1, 1], "loser_id": [2, 2],  # same pairing twice
    })
    out = add_history_features(df, window=5)
    assert out.loc[0, "winner_h2h_wins"] == 0  # first meeting
    assert out.loc[1, "winner_h2h_wins"] == 1  # one prior win vs this foe


def test_history_requires_sorted_input():
    import pytest
    df = pd.DataFrame({
        "tourney_date": pd.to_datetime(["2020-06-01", "2020-01-01"]),  # descending
        "surface": ["Hard", "Hard"],
        "winner_id": [1, 1], "loser_id": [2, 3],
    })
    with pytest.raises(ValueError):
        add_history_features(df)
