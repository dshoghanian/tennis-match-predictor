# ATP Tennis Match Predictor

Predicts the winner of an ATP singles match using Elo ratings (overall +
surface-specific) and an XGBoost model trained on Jeff Sackmann's match data
(1991–present). Includes an on-demand matchup predictor.

## Results (held-out test set, 2022–present)

| Model | Accuracy | Log-loss | AUC |
|-------|----------|----------|-----|
| Higher-rank baseline | 0.639 | — | — |
| Elo baseline | 0.641 | 0.636 | 0.700 |
| **XGBoost** | **0.653** | **0.616** | **0.716** |

(~0.65 is in the expected range for pre-match ATP prediction; the realistic
ceiling is roughly 0.67–0.70.)

## Setup

```bash
conda env create -f environment.yml
conda activate tennis_env
bash scripts/download_data.sh          # fetch CSVs into data/raw/
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
- `src/predict.py` — artifact persistence + on-demand `predict_proba`
- `notebooks/tennis_prediction.ipynb` — EDA → Elo → features → models → evaluation → live prediction
- `scripts/download_data.sh` — data fetcher
- `tests/` — unit tests incl. leakage guards

## Avoiding data leakage

Every feature is computed strictly from information available **before** a match:
the Elo engine snapshots each player's rating before updating it, and the rolling
history features (recent form, head-to-head, rest days) use only prior matches.
Matches are split by time (train on the past, test on the future), never shuffled.

## On-demand prediction

```python
from src.predict import load_artifacts, predict_proba
model, state = load_artifacts("models")
predict_proba(model, state, "Alcaraz", "Sinner", "Clay")  # -> win probability
```

Note: on-demand predictions for a hypothetical future match use the players'
latest Elo ratings; the rolling history features are neutralized (set to 0), so
these probabilities are Elo-driven and can be more extreme than the model's
test-set distribution.

## Data

[JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp),
licensed CC BY-NC-SA 4.0 (non-commercial use only).
