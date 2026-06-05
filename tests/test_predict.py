# tests/test_predict.py
import numpy as np
import pytest
from src.predict import (
    save_artifacts,
    load_artifacts,
    predict_proba,
    _resolve_surface,
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


def test_predict_proba_favors_higher_elo(tmp_path):
    state = {
        "ratings": {1: 1700.0, 2: 1500.0},
        "surface_ratings": {(1, "Clay"): 1650.0, (2, "Clay"): 1500.0},
        "names": {1: "Carlos Alcaraz", 2: "Jannik Sinner"},
        "feature_columns": ["elo_diff", "surface_elo_diff"],
    }
    save_artifacts(tmp_path, _DummyModel(), state)
    model, loaded = load_artifacts(tmp_path)
    p = predict_proba(model, loaded, "Alcaraz", "Sinner", "Clay")
    assert p > 0.5  # higher-Elo player A favored


def test_unknown_name_raises(tmp_path):
    state = {
        "ratings": {1: 1700.0, 2: 1500.0},
        "surface_ratings": {(1, "Clay"): 1650.0, (2, "Clay"): 1500.0},
        "names": {1: "Carlos Alcaraz", 2: "Jannik Sinner"},
        "feature_columns": ["elo_diff", "surface_elo_diff"],
    }
    save_artifacts(tmp_path, _DummyModel(), state)
    model, loaded = load_artifacts(tmp_path)
    with pytest.raises(ValueError):
        predict_proba(model, loaded, "zzzxqwq", "Sinner", "Clay")


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


def test_surface_changes_prediction(tmp_path):
    # The bug: lowercase surface silently produced the same number every time.
    # With normalization, clay vs grass must give different results, and a
    # lowercase surface must match its capitalized form exactly.
    state = _multi_surface_state()
    save_artifacts(tmp_path, _DummyModel(), state)
    model, loaded = load_artifacts(tmp_path)
    p_clay = predict_proba(model, loaded, "Alcaraz", "Sinner", "clay")
    p_grass = predict_proba(model, loaded, "Alcaraz", "Sinner", "grass")
    assert p_clay != p_grass  # surface actually matters now
    # case-insensitive: lowercase equals capitalized
    assert p_clay == predict_proba(model, loaded, "Alcaraz", "Sinner", "Clay")
