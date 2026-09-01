# Fantasy Football Success Predictor

Predicts a player's fantasy points for their **next week** using their
recent performance trend (rolling averages of the past 3 weeks), compared
across a simple baseline model (Linear Regression) and a more flexible
model (Random Forest).

This is Project 1 of a sports-prediction portfolio (next up: NCAA
tournament / "March Madness" win predictor).

## How it's structured

```
fantasy-football-predictor/
├── main.py              # run this — ties everything together
├── requirements.txt
├── src/
│   ├── data_loader.py   # loads real NFL data (or synthetic demo data)
│   ├── opponent.py      # computes opponent defensive strength by position
│   ├── features.py      # builds rolling-average + opponent-strength features
│   └── model.py          # trains + evaluates the models
```

## Quick start (works right now, no setup)

The project ships with a synthetic data generator so you can run the
whole pipeline immediately, with no data download required:

```bash
pip install -r requirements.txt
python main.py
```

You should see model results printed (MAE and R² for each model, plus
which features mattered most).

## Using REAL NFL data instead of synthetic data

Right now `data_loader.py` automatically falls back to fake data because
`nfl_data_py` isn't installed. To use real historical NFL stats:

```bash
pip install nfl_data_py
python main.py
```

That's it — `load_weekly_data()` will detect the package and pull real
weekly stats automatically instead of generating synthetic data.

`nfl_data_py` pulls from the public [nflverse/nflfastR](https://github.com/nflverse)
data releases, so no API key is needed.

## What the model is actually doing

1. **Data**: one row per player per week (points scored, targets, carries, etc.)
2. **Player features**: for each player-week, the average of their own stats
   over the *previous* 3 weeks — never looking at the target week itself
   (avoiding "data leakage," a common mistake in sports prediction)
3. **Opponent-strength feature**: for the upcoming opponent, how many
   fantasy points that defense has allowed to this position over its
   *previous* 3 games — this accounts for matchup difficulty, not just
   the player's own recent form (`src/opponent.py`)
4. **Target**: that same player's fantasy points the *following* week
5. **Models**: Linear Regression (interpretable baseline) vs. Random
   Forest (captures nonlinear patterns, e.g. workload thresholds)
6. **Evaluation**: trained on 80% of the data, tested on the held-out 20%
   the model never saw — MAE (average error in points) and R² (how much
   of the variance the model explains)

## Ideas to extend this further (pick 1, don't do everything)

- **Add injury/snap-count data**: `nfl_data_py` has snap count data;
  players with rising snap shares often break out
- **Try classification instead of regression**: predict "boom" (top 25%
  of points for their position that week) vs. not, using logistic
  regression — a cleaner story for a write-up than raw point prediction
- **Position-specific models**: train separate models per position
  (QB/RB/WR/TE) instead of one combined model — likely to improve accuracy
  since the stat profiles are so different
- **Backtest a full season**: simulate "if I'd started my highest-predicted
  player every week," how would that have done vs. always starting last
  week's top scorer?

## Honest note on the current results

With synthetic data, R² is low (~0.13–0.16) — meaning the model explains
some but not most of the variance, which is realistic! Fantasy football
outcomes are genuinely noisy (game script, injuries, matchups all matter).
With real data and a couple of the extensions above (especially opponent
strength), expect meaningfully better performance — and that improvement
story ("I added X and MAE dropped by Y") is exactly the kind of concrete,
specific detail worth including in a project write-up.
