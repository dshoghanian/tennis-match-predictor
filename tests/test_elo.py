# tests/test_elo.py
import pandas as pd
import pytest
from src.elo import expected_score, k_factor, compute_elo


def test_expected_score_equal_ratings():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_higher_rating_favored():
    assert expected_score(1700, 1500) > 0.5


def test_k_factor_decreases_with_matches():
    assert k_factor(0) > k_factor(100)


def test_elo_is_zero_sum_after_one_match():
    # Winner gains exactly what loser loses (both start at 1500).
    df = pd.DataFrame({
        "tourney_date": pd.to_datetime(["2020-01-01"]),
        "surface": ["Hard"],
        "winner_id": [1], "loser_id": [2],
        "winner_name": ["A"], "loser_name": ["B"],
    })
    out, ratings, _ = compute_elo(df)
    gain = ratings[1] - 1500
    loss = 1500 - ratings[2]
    assert gain == pytest.approx(loss)
    assert gain > 0


def test_elo_columns_are_pre_match_snapshots():
    # First match for both players => their *before* Elo must be the 1500 seed,
    # not the post-update value. This is the leakage guard.
    df = pd.DataFrame({
        "tourney_date": pd.to_datetime(["2020-01-01"]),
        "surface": ["Hard"],
        "winner_id": [1], "loser_id": [2],
        "winner_name": ["A"], "loser_name": ["B"],
    })
    out, _, _ = compute_elo(df)
    assert out.loc[0, "winner_elo_before"] == 1500
    assert out.loc[0, "loser_elo_before"] == 1500
