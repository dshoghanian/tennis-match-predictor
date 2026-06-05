# tests/test_data.py
import pandas as pd
from src.data import clean_matches, ROUND_ORDER


def _raw_df():
    return pd.DataFrame({
        "tourney_date": [20230102, 20230102, 20230109],
        "round": ["R32", "F", "R16"],
        "surface": ["Hard", "Hard", None],   # last row has missing surface
        "winner_id": [1, 1, 2],
        "loser_id": [2, 3, 3],
    })


def test_clean_parses_dates():
    out = clean_matches(_raw_df())
    assert pd.api.types.is_datetime64_any_dtype(out["tourney_date"])


def test_clean_drops_missing_surface():
    out = clean_matches(_raw_df())
    assert out["surface"].isna().sum() == 0
    assert len(out) == 2


def test_clean_sorts_chronologically_then_by_round():
    # Same date: R32 (earlier round) must come before F (final)
    out = clean_matches(_raw_df()).reset_index(drop=True)
    assert ROUND_ORDER[out.loc[0, "round"]] <= ROUND_ORDER[out.loc[1, "round"]]


def test_clean_sorts_across_dates():
    raw = pd.DataFrame({
        "tourney_date": [20230109, 20230102],
        "round": ["F", "F"],
        "surface": ["Hard", "Hard"],
        "winner_id": [1, 2],
        "loser_id": [3, 4],
    })
    out = clean_matches(raw).reset_index(drop=True)
    assert out.loc[0, "tourney_date"] < out.loc[1, "tourney_date"]


def test_clean_drops_helper_column():
    out = clean_matches(_raw_df())
    assert "_round_order" not in out.columns


# --- load_matches tour-subfolder tests ---
from src.data import load_matches


def test_load_matches_reads_tour_subdir(tmp_path):
    d = tmp_path / "wta"
    d.mkdir()
    pd.DataFrame({
        "tourney_date": [20200101], "round": ["F"], "surface": ["Hard"],
        "winner_id": [1], "loser_id": [2],
    }).to_csv(d / "wta_matches_2020.csv", index=False)
    df = load_matches(tmp_path, tour="wta", start_year=2020, end_year=2020)
    assert len(df) == 1
    assert df.loc[0, "winner_id"] == 1


def test_load_matches_missing_tour_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_matches(tmp_path, tour="wta", start_year=2020, end_year=2020)
