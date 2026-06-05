# tests/test_predict.py
import numpy as np
from src.predict import save_artifacts, load_artifacts, predict_proba


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
    import pytest
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
