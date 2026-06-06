# ATP & WTA Tennis Match Predictor

Predicts the winner of a professional singles match on the men's (ATP) or women's (WTA) tour, using Elo ratings (overall + surface-specific) and an XGBoost model trained on Jeff Sackmann's match data (1991–present). Each tour is a separate model. Includes an on-demand matchup predictor.

## Results (held-out test set, 2022–present)

| Tour | Higher-rank | Elo | XGBoost | AUC |
|------|-------------|-----|---------|-----|
| ATP (men) | 0.639 | 0.641 | **0.653** | 0.716 |
| WTA (women) | 0.625 | 0.650 | **0.655** | 0.713 |

XGBoost beats the rank and Elo baselines on both tours; ~0.65 is in the expected
range for pre-match tennis prediction (realistic ceiling ~0.67–0.70).

## Setup

```bash
conda env create -f environment.yml
conda activate tennis_env
bash scripts/download_data.sh        # fetch ATP + WTA CSVs into data/raw/<tour>/
jupyter lab                            # open notebooks/tennis_prediction.ipynb
```

## Tests

```bash
pytest -v        # or: conda run -n tennis_env pytest -q
```

## Project layout

- `src/data.py` — load, clean, chronologically sort match files
- `src/elo.py` — chronological Elo engine (overall + surface-specific)
- `src/features.py` — leakage-safe symmetric feature builder
- `src/predict.py` — artifact persistence + tour-routed on-demand predictor (`load_predictors`, `predict_match`)
- `notebooks/tennis_prediction.ipynb` — EDA → Elo → features → models → evaluation → live prediction
- `scripts/download_data.sh` — data fetcher
- `tests/` — unit tests incl. leakage guards

## Avoiding data leakage

Every feature is computed strictly from information available **before** a match:
the Elo engine snapshots each player's rating before updating it, and the rolling
history features (recent form, head-to-head, rest days) use only prior matches.
Matches are split by time (train on the past, test on the future), never shuffled.

## On-demand prediction

Each tour is a separate model; `tour` selects which. Run the notebook's per-tour
training section first to generate `models/atp/` and `models/wta/`.

```python
from src.predict import load_predictors, predict_match
preds = load_predictors("models")
predict_match(preds, "Sinner", "Medvedev", "Hard", tour="atp")
predict_match(preds, "Swiatek", "Sabalenka", "Clay", tour="wta")
```

Player names are fuzzy-matched and disambiguated by Elo within the chosen tour
(so "Sinner" → Jannik Sinner, not the journeyman Martin Sinner). Surfaces are
case-insensitive. An unknown tour, surface, or unrecognizable name raises a clear
error. Use `predict_match` to see the resolved full names; `predict_proba(...)`
returns just the probability.

**Scope:** the data is tour-level (ATP/WTA *main draw*), so lower-tier players
(ITF / futures / qualifying) are not included. Naming one raises a "not in the
dataset" error rather than silently substituting an unrelated namesake — the
predictor refuses to guess about players it has never seen.

## Data

[JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) (men) and
[JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta) (women),
both licensed CC BY-NC-SA 4.0 (non-commercial use only).
