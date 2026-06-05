# tests/test_predict.py
import numpy as np
import pytest
from src.predict import (
    save_artifacts,
    load_artifacts,
    predict_proba,
    predict_match,
    _resolve_surface,
    _resolve_player,
)


class _DummyModel:
    """Predicts A wins iff elo_diff (first feature) > 0."""
    def predict_proba(self, X):
        p_a = (np.asarray(X)[:, 0] > 0).astype(float)
        return np.column_stack([1 - p_a, p_a])


def test_artifacts_round_trip(tmp_path):
    state = {
        "ratings": {1: 1700.0, 2: 1500.0},
        "surface_ratings": {(1, "Clay"): 1650.0, (2, "Clay"): 1500.0},
        "names": {1: "Carlos Alcaraz", 2: "Jannik Sinner"},
        "feature_columns": ["elo_diff", "surface_elo_diff"],
    }
    save_artifacts(tmp_path, _DummyModel(), state)
    model, loaded = load_artifacts(tmp_path)
    assert loaded["ratings"][1] == 1700.0


# helper: wrap a single state as a one-tour predictors dict
def _atp(state):
    return {"atp": (_DummyModel(), state)}


def test_predict_proba_favors_higher_elo(tmp_path):
    state = {
        "ratings": {1: 1700.0, 2: 1500.0},
        "surface_ratings": {(1, "Clay"): 1650.0, (2, "Clay"): 1500.0},
        "names": {1: "Carlos Alcaraz", 2: "Jannik Sinner"},
        "feature_columns": ["elo_diff", "surface_elo_diff"],
    }
    p = predict_proba(_atp(state), "Alcaraz", "Sinner", "Clay", tour="atp")
    assert p > 0.5


def test_unknown_name_raises():
    state = {
        "ratings": {1: 1700.0, 2: 1500.0},
        "surface_ratings": {(1, "Clay"): 1650.0, (2, "Clay"): 1500.0},
        "names": {1: "Carlos Alcaraz", 2: "Jannik Sinner"},
        "feature_columns": ["elo_diff", "surface_elo_diff"],
    }
    with pytest.raises(ValueError):
        predict_proba(_atp(state), "zzzxqwq", "Sinner", "Clay", tour="atp")


def _multi_surface_state():
    return {
        "ratings": {1: 1600.0, 2: 1500.0},
        "surface_ratings": {
            (1, "Clay"): 1800.0, (2, "Clay"): 1500.0,   # player 1 strong on clay
            (1, "Grass"): 1500.0, (2, "Grass"): 1800.0,  # player 2 strong on grass
        },
        "names": {1: "Carlos Alcaraz", 2: "Jannik Sinner"},
        "feature_columns": ["surface_elo_diff", "elo_diff"],  # dummy reads col 0
    }


def test_resolve_surface_normalizes_case():
    sr = {(1, "Clay"): 1.0, (1, "Hard"): 1.0}
    assert _resolve_surface("clay", sr) == "Clay"
    assert _resolve_surface("HARD", sr) == "Hard"
    assert _resolve_surface(" Clay ", sr) == "Clay"


def test_unknown_surface_raises():
    sr = {(1, "Clay"): 1.0, (1, "Hard"): 1.0}
    with pytest.raises(ValueError):
        _resolve_surface("dirt", sr)


def test_name_disambiguated_by_elo():
    # A bare surname shared by two players must resolve to the higher-Elo one,
    # not whichever the raw fuzzy score happens to rank first.
    names = {1: "Jannik Sinner", 2: "Martin Sinner"}
    ratings = {1: 2100.0, 2: 1400.0}
    _id, matched = _resolve_player("Sinner", names, ratings)
    assert matched == "Jannik Sinner"


def test_name_typo_resolves_to_prominent_player():
    names = {1: "Carlos Alcaraz", 2: "Emilio Benfele Alvarez"}
    ratings = {1: 2200.0, 2: 1500.0}
    _id, matched = _resolve_player("Alcarez", names, ratings)
    assert matched == "Carlos Alcaraz"


def test_exact_full_name_beats_elo_tiebreak():
    # Typing the full weaker name must still select that player; the Elo
    # tiebreak only applies among near-tied fuzzy scores.
    names = {1: "Jannik Sinner", 2: "Martin Sinner"}
    ratings = {1: 2100.0, 2: 1400.0}
    _id, matched = _resolve_player("Martin Sinner", names, ratings)
    assert matched == "Martin Sinner"


def test_predict_match_reports_resolved_names():
    state = {
        "ratings": {1: 2100.0, 2: 1400.0},
        "surface_ratings": {(1, "Clay"): 2100.0, (2, "Clay"): 1400.0},
        "names": {1: "Jannik Sinner", 2: "Martin Sinner"},
        "feature_columns": ["elo_diff", "surface_elo_diff"],
    }
    out = predict_match(_atp(state), "Sinner", "Martin Sinner", "clay", tour="atp")
    assert out["player_a"] == "Jannik Sinner"
    assert out["player_b"] == "Martin Sinner"
    assert out["surface"] == "Clay"
    assert out["tour"] == "atp"
    assert 0.0 <= out["prob"] <= 1.0


def test_surface_changes_prediction():
    state = _multi_surface_state()
    preds = {"atp": (_DummyModel(), state)}
    p_clay = predict_proba(preds, "Alcaraz", "Sinner", "clay", tour="atp")
    p_grass = predict_proba(preds, "Alcaraz", "Sinner", "grass", tour="atp")
    assert p_clay != p_grass
    assert p_clay == predict_proba(preds, "Alcaraz", "Sinner", "Clay", tour="atp")


def test_unknown_tour_raises():
    state = {
        "ratings": {1: 1700.0, 2: 1500.0},
        "surface_ratings": {(1, "Clay"): 1650.0, (2, "Clay"): 1500.0},
        "names": {1: "Carlos Alcaraz", 2: "Jannik Sinner"},
        "feature_columns": ["elo_diff", "surface_elo_diff"],
    }
    with pytest.raises(ValueError):
        predict_match(_atp(state), "Alcaraz", "Sinner", "Clay", tour="wta")


def test_load_predictors_loads_available_tours(tmp_path):
    from src.predict import load_predictors
    state = {
        "ratings": {1: 1700.0}, "surface_ratings": {(1, "Clay"): 1700.0},
        "names": {1: "Carlos Alcaraz"}, "feature_columns": ["elo_diff", "surface_elo_diff"],
    }
    save_artifacts(tmp_path / "atp", _DummyModel(), state)
    save_artifacts(tmp_path / "wta", _DummyModel(), state)
    preds = load_predictors(tmp_path)
    assert set(preds.keys()) == {"atp", "wta"}


def test_load_predictors_skips_missing_tour(tmp_path):
    from src.predict import load_predictors
    state = {
        "ratings": {1: 1700.0}, "surface_ratings": {(1, "Clay"): 1700.0},
        "names": {1: "Carlos Alcaraz"}, "feature_columns": ["elo_diff", "surface_elo_diff"],
    }
    save_artifacts(tmp_path / "atp", _DummyModel(), state)  # only atp present
    preds = load_predictors(tmp_path)
    assert set(preds.keys()) == {"atp"}


def test_load_predictors_raises_when_no_tours(tmp_path):
    from src.predict import load_predictors
    with pytest.raises(FileNotFoundError):
        load_predictors(tmp_path)
