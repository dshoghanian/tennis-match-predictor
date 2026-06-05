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
# When several names score within this many points of the best fuzzy match,
# treat them as a tie and disambiguate by Elo (prefer the stronger player).
# This prevents a bare surname like "Sinner" from resolving to an obscure
# namesake (Martin Sinner) instead of the player you mean (Jannik Sinner).
FUZZY_MATCH_MARGIN = 15


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


def _resolve_surface(surface, surface_ratings):
    """Normalize a surface string and validate it against known surfaces.

    Accepts any case ("clay", "CLAY" -> "Clay"). Raises ValueError for an
    unknown surface so a typo fails loudly instead of silently falling back to
    a neutral rating (which would make every surface return the same number).
    """
    known = {s for (_pid, s) in surface_ratings}
    normalized = str(surface).strip().title()
    if normalized not in known:
        raise ValueError(
            f"Unknown surface {surface!r}. Expected one of {sorted(known)}."
        )
    return normalized


def _resolve_player(name, names, ratings):
    """Fuzzy-match a typed name to a known player_id, disambiguating by Elo.

    Names are matched case-insensitively. Among all candidates whose fuzzy score
    is within FUZZY_MATCH_MARGIN of the best, the one with the highest Elo is
    chosen, so an ambiguous surname resolves to the player you most likely mean
    (e.g. "Sinner" -> Jannik Sinner, not Martin Sinner). Raises ValueError if no
    candidate scores above FUZZY_MATCH_THRESHOLD, so an unrecognized name fails
    loudly instead of resolving to a wrong player.
    """
    name_to_id = {v: k for k, v in names.items()}
    candidates = fuzz_process.extract(name, list(name_to_id), limit=10)
    best_score = candidates[0][1]
    if best_score < FUZZY_MATCH_THRESHOLD:
        raise ValueError(
            f"No confident match for {name!r} "
            f"(best guess {candidates[0][0]!r}, score {best_score})."
        )
    # Among near-tied name matches, prefer the highest-Elo (most prominent) player.
    close = [c for c in candidates if c[1] >= best_score - FUZZY_MATCH_MARGIN]
    best_name = max(
        close, key=lambda c: ratings.get(name_to_id[c[0]], INITIAL_ELO)
    )[0]
    return name_to_id[best_name], best_name


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


def predict_match(model, state, name_a, name_b, surface):
    """Resolve names + surface and predict, returning the resolved labels too.

    Returns a dict: {prob, player_a, player_b, surface} where prob is
    P(player_a beats player_b). Exposing the resolved full names lets callers
    confirm the fuzzy matcher picked the players they meant.
    """
    surface = _resolve_surface(surface, state["surface_ratings"])
    id_a, resolved_a = _resolve_player(name_a, state["names"], state["ratings"])
    id_b, resolved_b = _resolve_player(name_b, state["names"], state["ratings"])
    X = _feature_vector(state, id_a, id_b, surface)
    prob = float(model.predict_proba(X)[0, 1])
    return {"prob": prob, "player_a": resolved_a,
            "player_b": resolved_b, "surface": surface}


def predict_proba(model, state, name_a, name_b, surface):
    """Return P(player A beats player B) on the given surface.

    `surface` is case-insensitive ("clay" == "Clay"); an unknown surface raises
    ValueError. Names are fuzzy-matched and disambiguated by Elo (see
    _resolve_player). Use predict_match to also see the resolved player names.
    """
    return predict_match(model, state, name_a, name_b, surface)["prob"]
