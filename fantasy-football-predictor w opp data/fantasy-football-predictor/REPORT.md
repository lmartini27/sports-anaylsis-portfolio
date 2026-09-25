# Fantasy Football Success Predictor: Project Report

## Objective

Predict an NFL player's PPR fantasy points in their **next game**, using their recent form and the next opponent's recent defense against their position. The model is compared against a naive baseline so its value is measurable, not just its accuracy.

## Data

- **Source:** `nfl_data_py` (weekly player stats from the public nflverse/nflfastR releases)
- **Scope:** 2021–2023 regular seasons; QB, RB, WR, TE only
- **Volume:** 16,248 player-week rows loaded; 12,997 used for modeling

A row is kept only if the player's next game is in the **same season** and at most 2 weeks later (allowing for a bye week). This excludes predictions across an offseason and after injury absences.

## Methodology

### Timing and leakage

Each row is a player's game in week *t*. Features use only information available after week *t* and before the next game. The target is the player's points in that next game.

### Features (12 total)

1. **Player form:** 3-game rolling averages, up to and including week *t*, of fantasy points, targets, carries, receptions, and receiving, rushing, and passing yards. Averages reset each season.
2. **Next-opponent strength:** the average fantasy points the *next* opponent allowed to the player's position over its previous 3 games that season, excluding the game being predicted.
3. **Position:** one-hot QB/RB/WR/TE.

### Evaluation

- **Walk-forward split by season:** train on 2021 and test on 2022; then train on 2021–22 and test on 2023. Models never train on the future. Reported scores average the two test seasons.
- **Naive baseline:** predict each player's 3-game points average. Any useful model must beat this.
- **Ablation:** every model is run with and without the opponent feature to isolate its contribution.

Models: Linear Regression, and Random Forest (200 trees, max depth 6).

## Results

### Accuracy (average of 2022 and 2023 test seasons)

| Model | MAE (pts) | R² | R² 2022 | R² 2023 |
|---|---|---|---|---|
| Naive baseline (3-game average) | 5.34 | 0.182 | 0.156 | 0.208 |
| Linear Regression | **5.06** | **0.309** | 0.295 | 0.322 |
| Random Forest | 5.10 | 0.298 | 0.280 | 0.315 |

### Opponent-feature ablation

| Model | R² without opponent | R² with opponent | Change |
|---|---|---|---|
| Linear Regression | 0.308 | 0.309 | +0.001 |
| Random Forest | 0.299 | 0.298 | −0.001 |

### Feature importance (Random Forest, 2023 fold)

| Feature | Importance |
|---|---|
| Fantasy points (3-game avg) | 0.752 |
| Targets (3-game avg) | 0.051 |
| Passing yards (3-game avg) | 0.047 |
| Next opponent's points allowed to position | 0.037 |
| Carries (3-game avg) | 0.034 |
| Rushing yards (3-game avg) | 0.033 |
| Receiving yards (3-game avg) | 0.030 |
| Receptions (3-game avg) | 0.012 |
| Position indicators | ≤ 0.001 each |

## Key Findings

1. **The model clearly beats the naive baseline.** Linear Regression explains about 31% of the variance in next-game scoring vs. 18% for the recent average, and cuts the typical error from 5.34 to 5.06 points. It wins in both test seasons, so the gain isn't a lucky split.

2. **The model's main job is regression to the mean.** Recent fantasy points is by far the top feature, yet the model beats simply *using* that average by a wide margin. Raw recent averages overreact to hot and cold streaks; the model learns to pull predictions back toward typical levels, with volume stats (targets, carries) helping judge whether a streak is backed by real opportunity.

3. **The matchup feature adds essentially nothing.** Correctly aligned to the next game, it changes R² by about ±0.001. A defense's total points allowed to a position over three games appears too noisy to be useful: it depends on how many players at that position saw the field, on game script, and on just three games.

4. **Linear Regression matches or beats Random Forest.** The added flexibility buys nothing here, suggesting the relationships in these features are close to linear.

## Corrections from the Previous Version

An earlier version of this project reported R² ≈ 0.31 with a small opponent-feature gain. A review found three problems, all fixed here:

- The opponent feature described the opponent of the **current** game rather than the game being predicted.
- Predictions crossed season boundaries (a player's last game of one season was used to predict week 1 of the next).
- A random train/test split let the same player's neighbouring weeks appear in both training and test data, and there was no naive baseline to judge the R² against.

The corrected model reaches a similar R², but now against a harder, time-based test, with a baseline showing that the model genuinely adds value. The opponent-feature gain from the original version did not hold up.

## Limitations

- Fantasy scoring is inherently noisy (injuries, game script, weather, play-calling), so an R² near 0.3 is a realistic range for performance-based features.
- The opponent feature is coarse: it sums points allowed to a whole position group over three games, not per-player or per-snap.
- Only two test seasons are available; more seasons would tighten the estimates.
- The 3-game window was not tuned.

## Next Steps

- **Better matchup features:** season-to-date points allowed per player at the position, or defense-vs-league-average adjustments, may extract a signal the 3-game version couldn't.
- **Per-position models,** since QB, RB, WR, and TE scoring behave differently.
- **Tuning the rolling window,** comparing 3, 5, and season-to-date averages.
