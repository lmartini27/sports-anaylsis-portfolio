# Fantasy Football Success Predictor — Project Report

## Objective

Predict an NFL player's fantasy points (PPR scoring) for their **next
week**, using their own recent performance trend and their upcoming
opponent's defensive strength at their position. The goal was to build a
genuinely applied statistical model — not just a stat tracker — using
real historical NFL data.

## Data

- **Source**: `nfl_data_py`, which provides weekly player statistics
  pulled from the public nflverse/nflfastR data releases
- **Scope**: weekly player-level stats across the 2021–2023 NFL seasons
- **Volume**: 16,982 player-week rows loaded; 15,171 rows retained after
  feature engineering (rows are dropped when a player doesn't yet have
  enough game history, or when there's no "next week" to predict, e.g.
  a player's final game of a season)

## Methodology

### Feature Engineering

Two categories of features were built, both carefully avoiding data
leakage (never using information the model wouldn't actually have
*before* a game is played):

1. **Player form** — rolling 3-week averages of the player's own stats
   (fantasy points, targets, carries, receiving/rushing/passing yards,
   receptions), computed using only *prior* weeks.
2. **Opponent strength** — for the upcoming opponent, a rolling 3-week
   average of how many fantasy points that defense has allowed to the
   player's position, computed the same leakage-safe way. This captures
   matchup difficulty (facing a strong vs. weak defense) independent of
   how well the player themselves has been playing.

Position (QB/RB/WR/TE) was included as a one-hot encoded feature.

In total, **26 features** were used per prediction.

### Modeling

Two models were trained and compared:
- **Linear Regression** — an interpretable baseline
- **Random Forest** (200 trees, max depth 6) — able to capture nonlinear
  relationships between features

Both were trained on 80% of the data and evaluated on a held-out 20% test
set the models never saw during training, to get an honest measure of
predictive accuracy rather than a measure of memorization.

## Results

| Model | MAE (points) | R² |
|---|---|---|
| Linear Regression | 5.04 | 0.313 |
| Random Forest | 5.07 | 0.305 |

**Interpretation**: the typical prediction is off by about 5 fantasy
points, and the model explains roughly 31% of the week-to-week variance
in a player's scoring. Linear Regression performed marginally better
than Random Forest, suggesting the relationships captured by these
features are close to linear — the added complexity of a Random Forest
isn't buying meaningful accuracy here.

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| Recent fantasy points (avg, last 3 wks) | 0.792 |
| Recent targets (avg, last 3 wks) | 0.050 |
| Recent passing yards (avg, last 3 wks) | 0.036 |
| Recent carries (avg, last 3 wks) | 0.035 |
| Recent rushing yards (avg, last 3 wks) | 0.030 |
| **Opponent defensive strength (avg, last 3 wks)** | **0.023** |
| Recent receiving yards (avg, last 3 wks) | 0.019 |
| Recent receptions (avg, last 3 wks) | 0.013 |
| Position: QB | 0.001 |
| Position: TE | 0.001 |

## Key Findings

1. **Recency dominates.** A player's own recent scoring average accounts
   for the vast majority of the model's predictive power (0.792 of total
   importance). This is intuitive but also a limitation — the model is
   mostly learning "hot players stay hot," not a deeper pattern.
2. **Opponent strength contributes, modestly.** Adding the defensive
   matchup feature improved R² from 0.297–0.306 (baseline, without
   opponent data) to 0.305–0.313 (with it) — a real but small gain. The
   feature ranked 6th in importance, likely because it summarizes an
   entire position's performance against a defense over only 3 games,
   which is a fairly coarse and noisy signal.
3. **Random Forest added no advantage over Linear Regression** in this
   configuration, suggesting the underlying relationships are close to
   linear given the current feature set — added model complexity isn't
   the bottleneck right now.

## Limitations

- Fantasy scoring is inherently noisy (injuries, game script, weather,
  coaching decisions), so an R² around 0.3 is a realistic ceiling for a
  model using only performance-based features — it is not a sign the
  model is broken.
- The opponent-strength feature is coarse: it aggregates all players at
  a position against a defense, rather than accounting for more specific
  matchup factors (e.g. a specific cornerback vs. a specific receiver).
- The 3-week rolling window is a simplifying choice; it hasn't been
  tuned or compared against other window lengths.

## Next Steps

Planned extensions include training separate models per position (since
QB, RB, WR, and TE have very different scoring profiles), reframing part
of the problem as a "boom/bust" classification task rather than pure
point prediction, and adding statistical rigor (cross-validation,
regularization) to the evaluation process.
