# src/predict.py
"""Persist trained artifacts and run on-demand matchup predictions."""
import pickle
from pathlib import Path

import numpy as np
from thefuzz import process as fuzz_process

from src.elo import INITIAL_ELO

MODEL_FILE = "model.pkl"
STATE_FILE = "state.pkl"
FUZZY_MATCH_THRESHOLD = 60


def save_artifacts(out_dir, model, state):
    """Persist the trained model and the latest Elo/name state."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(out_dir / STATE_FILE, "wb") as f:
        pickle.dump(state, f)


def load_artifacts(in_dir):
    """Return (model, state)."""
    in_dir = Path(in_dir)
    with open(in_dir / MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(in_dir / STATE_FILE, "rb") as f:
        state = pickle.load(f)
    return model, state


def _resolve_player(name, names):
    """Fuzzy-match a typed name to a known player_id.

    Raises ValueError if no candidate scores above FUZZY_MATCH_THRESHOLD,
    so an unrecognized name fails loudly instead of resolving to a wrong player.
    """
    name_to_id = {v: k for k, v in names.items()}
    match, score = fuzz_process.extractOne(name, list(name_to_id.keys()))
    if score < FUZZY_MATCH_THRESHOLD:
        raise ValueError(
            f"No confident match for {name!r} (best guess {match!r}, score {score})."
        )
    return name_to_id[match], match


def _feature_vector(state, id_a, id_b, surface):
    """Build the model input row for A vs B, matching FEATURE_COLUMNS order.

    Only Elo-based features are populated for live prediction; history-based
    diffs default to 0 (neutral) since we score a hypothetical future match.
    """
    ratings = state["ratings"]
    sr = state["surface_ratings"]
    elo_diff = ratings.get(id_a, INITIAL_ELO) - ratings.get(id_b, INITIAL_ELO)
    se_diff = (sr.get((id_a, surface), INITIAL_ELO)
               - sr.get((id_b, surface), INITIAL_ELO))
    values = {"elo_diff": elo_diff, "surface_elo_diff": se_diff}
    return np.array([[values.get(c, 0.0) for c in state["feature_columns"]]])


def predict_proba(model, state, name_a, name_b, surface):
    """Return P(player A beats player B) on the given surface."""
    id_a, _ = _resolve_player(name_a, state["names"])
    id_b, _ = _resolve_player(name_b, state["names"])
    X = _feature_vector(state, id_a, id_b, surface)
    return float(model.predict_proba(X)[0, 1])
