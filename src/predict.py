# src/predict.py
"""Persist trained artifacts and run on-demand matchup predictions."""
import pickle
from pathlib import Path

import numpy as np
from thefuzz import fuzz
from thefuzz import process as fuzz_process

from src.elo import INITIAL_ELO

MODEL_FILE = "model.pkl"
STATE_FILE = "state.pkl"
FUZZY_MATCH_THRESHOLD = 60
# Every word in a typed name must appear (this well) in the matched player's
# name. A low minimum means a typed word is absent from the match — i.e. the
# player is probably not in the dataset, so we refuse rather than silently
# substitute someone else (e.g. "Maria Oliver Sanchez" -> "Maria Sakkari").
# 85 keeps surname lookups and single-char typos while rejecting absent players.
NAME_COVERAGE_THRESHOLD = 85


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


def _name_coverage(query, candidate):
    """Min over query words of the best per-word similarity to the candidate name.

    Low coverage means a typed word does not appear in the matched name, which
    signals the intended player is not in the dataset (rather than a typo).
    """
    q_tokens = query.lower().split()
    c_tokens = candidate.lower().split()
    if not q_tokens or not c_tokens:
        return 0
    return min(max(fuzz.ratio(q, c) for c in c_tokens) for q in q_tokens)


def _resolve_player(name, names, ratings):
    """Fuzzy-match a typed name to a known player_id, disambiguating by Elo.

    Names are matched case-insensitively. Candidates are first restricted to
    those where every typed word sufficiently appears (NAME_COVERAGE_THRESHOLD),
    then the highest-Elo (most prominent) of those is chosen — so an ambiguous
    surname resolves to the player you most likely mean ("Sinner" -> Jannik
    Sinner) while a coincidental higher-Elo partial namesake cannot hijack an
    exact match ("Maja Chwalinska" stays Maja Chwalinska). Raises ValueError if
    no candidate scores above FUZZY_MATCH_THRESHOLD, or if none clears the
    coverage bar — so a player not in the dataset fails loudly instead of being
    silently swapped for an unrelated namesake.
    """
    name_to_id = {v: k for k, v in names.items()}
    candidates = fuzz_process.extract(name, list(name_to_id), limit=20)
    best_score = candidates[0][1]
    if best_score < FUZZY_MATCH_THRESHOLD:
        raise ValueError(
            f"No confident match for {name!r} "
            f"(best guess {candidates[0][0]!r}, score {best_score})."
        )
    # Keep only candidates where every typed word actually appears in the name,
    # THEN prefer the highest-Elo player among them. Filtering by coverage before
    # the Elo tiebreak stops a higher-Elo partial namesake from displacing an
    # exact match and then being rejected (which would refuse a real player).
    eligible = [c for c in candidates
                if _name_coverage(name, c[0]) >= NAME_COVERAGE_THRESHOLD]
    if not eligible:
        raise ValueError(
            f"Couldn't confidently find a player named {name!r} "
            f"(closest was {candidates[0][0]!r}). They may not be in the dataset, "
            f"which covers tour-level ATP/WTA main-draw players only."
        )
    best_name = max(
        eligible, key=lambda c: ratings.get(name_to_id[c[0]], INITIAL_ELO)
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


def load_predictors(models_dir, tours=("atp", "wta")):
    """Load each available tour's (model, state) from models_dir/<tour>/.

    Returns {tour: (model, state)} for every tour whose artifacts exist. Tours
    with no saved model are skipped, so an ATP-only setup still loads.
    """
    models_dir = Path(models_dir)
    predictors = {}
    for tour in tours:
        tour_dir = models_dir / tour
        if (tour_dir / MODEL_FILE).exists() and (tour_dir / STATE_FILE).exists():
            predictors[tour] = load_artifacts(tour_dir)
    if not predictors:
        raise FileNotFoundError(
            f"No tour models found under {models_dir} (looked for {list(tours)})."
        )
    return predictors


def _predict_one(model, state, name_a, name_b, surface):
    """Score a single matchup against one tour's model/state.

    Returns {prob, player_a, player_b, surface} where prob is P(player_a beats
    player_b). `surface` is case-insensitive; names are fuzzy-matched and
    disambiguated by Elo (see _resolve_player).
    """
    surface = _resolve_surface(surface, state["surface_ratings"])
    id_a, resolved_a = _resolve_player(name_a, state["names"], state["ratings"])
    id_b, resolved_b = _resolve_player(name_b, state["names"], state["ratings"])
    X = _feature_vector(state, id_a, id_b, surface)
    prob = float(model.predict_proba(X)[0, 1])
    return {"prob": prob, "player_a": resolved_a,
            "player_b": resolved_b, "surface": surface}


def predict_match(predictors, name_a, name_b, surface, tour):
    """Predict a matchup on a given tour.

    `predictors` is the dict from load_predictors. Raises ValueError if `tour`
    is not among the loaded tours. Returns the _predict_one dict plus `tour`.
    """
    if tour not in predictors:
        raise ValueError(
            f"Unknown tour {tour!r}. Available: {sorted(predictors)}."
        )
    model, state = predictors[tour]
    result = _predict_one(model, state, name_a, name_b, surface)
    result["tour"] = tour
    return result


def predict_proba(predictors, name_a, name_b, surface, tour):
    """Return P(player_a beats player_b) on the given tour and surface."""
    return predict_match(predictors, name_a, name_b, surface, tour)["prob"]
