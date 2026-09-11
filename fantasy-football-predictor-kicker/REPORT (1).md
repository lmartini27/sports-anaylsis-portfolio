# Kicker Fantasy Predictor — Full Investigation Report

## Objective

Most fantasy football tools ignore kickers entirely — they're the most
underserved position in fantasy analytics. This project asks a direct
question: **is a kicker's weekly fantasy output actually predictable,
and if not, why not?**

## The Investigation (five hypotheses, tested in order)

### 1. Recent form + opponent field-goal pressure
Rolling 3-week averages of the kicker's own stats, plus how many FG
attempts the upcoming opponent's defense has recently allowed.

**Result:** MAE 3.64 pts, R² ≈ -0.007 (linear) / -0.006 (RF) — no better
than guessing the average every week.

### 2. Team-level red zone tendency
Added the kicker's own team's red zone trip volume, and the opponent's
red zone touchdown rate allowed (a more precise version of "how stingy
is this defense near the goal line").

**Result:** MAE 3.65 pts, R² ≈ -0.008 / -0.011 — still no real
improvement. But `opp_redzone_td_rate_allowed` became the single most
important feature (0.188), ahead of the kicker's own recent scoring —
confirming the *reasoning* was right even though it didn't move overall
accuracy.

### 3. Vegas-implied team totals
Added the betting market's implied point total for the kicker's own
team — the single most information-dense predictor available in sports
analytics (it prices in weather, injuries, coaching tendencies, and
more, all at once).

**Result:** MAE 3.62 pts, R² ≈ -0.007 / -0.020 — again, no real
improvement, and again the new feature (`implied_team_total`) ranked
#1 in importance (0.186). Three different hypotheses, three times the
"most informative" feature still isn't informative *enough*.

### 4. Decomposition: separating accuracy from opportunity volume
Instead of predicting the combined weekly point total directly (which
compounds several separate random processes — how many chances a
kicker gets, and whether he converts each one), this splits the problem
into two parts:

**Accuracy model** (individual field goal attempts, make/miss, using
exact distance + weather/dome):
- ROC-AUC: **0.781** — real, meaningful discriminative power
- Coefficients all physically sensible: `kick_distance: -0.103` (longer
  is harder), `is_dome: +0.092` (domes help), `wind: -0.010` (wind
  hurts), `temp: +0.007` (warmer helps slightly)
- This is genuine signal — confirms the pipeline can find real
  relationships when they exist

**Volume models** (attempts per distance bucket + PATs per week):
| Target | MAE | R² |
|---|---|---|
| 0-39 yard attempts | 0.70 | 0.007 |
| 40-49 yard attempts | 0.67 | -0.023 |
| 50+ yard attempts | 0.55 | -0.013 |
| PAT attempts | 1.18 | 0.036 |

All essentially zero. Combined with the accuracy model into one
expected-points calculation, backtested predictions clustered tightly
(7.8–8.7) while real outcomes ranged from 4 to 14 — the model wasn't
wrong so much as barely responsive to real week-to-week variation.

## The Conclusion

**A kicker's fantasy output isn't unpredictable because kicking itself
is a random skill.** The accuracy model proves the opposite — makes and
misses follow real, physically sensible patterns tied to distance and
conditions. The unpredictability comes specifically from **opportunity
volume**: how many scoring chances a kicker's offense generates in any
given game is close to random relative to any pre-game information
tested here — recent form, opponent quality (raw and red-zone-specific),
and market expectations all failed to move the needle, while the same
methodology found strong signal the moment the target was accuracy
instead of volume.

This is a precise, falsifiable finding, not a shrug: the ceiling on
kicker fantasy prediction is set by game-to-game variance in scoring
opportunity — not by unmeasurable kicker skill, and not by a flawed
model (five separate model types across two outcome variables all
behaved consistently, which is itself evidence the pipeline works
correctly).

## Methodology Notes

- All rolling features use only strictly prior weeks (no data leakage)
- A target/feature alignment issue was identified and fixed mid-project:
  earlier versions predicted a game one week further out than the
  features actually supported, silently discarding each player's most
  recent game
- Every aggregation script was validated against synthetic data with a
  **known** ground-truth relationship before being trusted on real data
- Vegas lines and weather/dome status are used as direct per-game
  values (not rolling averages), since they're legitimately known
  before a specific game — unlike box-score stats, which are only
  known after

## Data Sources
- `nfl_data_py` (nflverse/nflfastR) — weekly stats, play-by-play,
  schedules/betting lines
- Field goal, red zone, and Vegas data all derived from play-by-play
  and schedule data via custom aggregation scripts in this repo
