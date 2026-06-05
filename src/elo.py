# src/elo.py
"""Chronological Elo engine: overall + surface-specific ratings.

Each match row is annotated with the players' PRE-match Elo before the rating
is updated, so downstream features never see post-match information.
"""
from collections import defaultdict

INITIAL_ELO = 1500.0


def expected_score(rating_a, rating_b):
    """Probability that A beats B under the logistic Elo model."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def k_factor(matches_played):
    """Decaying K: volatile for new players, stable for veterans."""
    return 250.0 / ((matches_played + 5) ** 0.4)


def compute_elo(df):
    """Replay matches in order, annotating pre-match Elo.

    Returns:
        out: df copy with winner/loser elo_before + surface_elo_before columns.
        ratings: dict[player_id] -> final overall Elo.
        surface_ratings: dict[(player_id, surface)] -> final surface Elo.
    """
    ratings = defaultdict(lambda: INITIAL_ELO)
    surface_ratings = defaultdict(lambda: INITIAL_ELO)
    n_played = defaultdict(int)
    n_played_surface = defaultdict(int)

    cols = {c: [] for c in (
        "winner_elo_before", "loser_elo_before",
        "winner_surface_elo_before", "loser_surface_elo_before",
    )}

    for row in df.itertuples(index=False):
        w, l, s = row.winner_id, row.loser_id, row.surface
        wr, lr = ratings[w], ratings[l]
        wsr, lsr = surface_ratings[(w, s)], surface_ratings[(l, s)]

        # Snapshot BEFORE updating.
        cols["winner_elo_before"].append(wr)
        cols["loser_elo_before"].append(lr)
        cols["winner_surface_elo_before"].append(wsr)
        cols["loser_surface_elo_before"].append(lsr)

        # Overall update (zero-sum).
        exp_w = expected_score(wr, lr)
        k = k_factor(n_played[w])
        k2 = k_factor(n_played[l])
        delta = ((k + k2) / 2) * (1 - exp_w)
        ratings[w] = wr + delta
        ratings[l] = lr - delta
        n_played[w] += 1
        n_played[l] += 1

        # Surface update.
        exp_ws = expected_score(wsr, lsr)
        ks = k_factor(n_played_surface[(w, s)])
        ks2 = k_factor(n_played_surface[(l, s)])
        deltas = ((ks + ks2) / 2) * (1 - exp_ws)
        surface_ratings[(w, s)] = wsr + deltas
        surface_ratings[(l, s)] = lsr - deltas
        n_played_surface[(w, s)] += 1
        n_played_surface[(l, s)] += 1

    out = df.copy()
    for c, vals in cols.items():
        out[c] = vals
    return out, dict(ratings), dict(surface_ratings)
