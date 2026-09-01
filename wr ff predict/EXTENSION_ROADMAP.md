# Fantasy Football Predictor — Extension Roadmap

This treats Project 1 as an ongoing, evolving project rather than a
one-and-done. Come back to it over several weeks alongside your other
work — depth here matters more than speed.

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 1 | Per-position models | **Current focus** |
| 2 | Boom/bust classification | **Current focus** |
| 3 | Statistical rigor (cross-validation, regularization, residual diagnostics) | Next |
| 4 (optional) | Live demo (Streamlit dashboard) | Stretch goal |

---

## Phase 1: Per-Position Models

**The problem with your current model:** it's trained on QBs, RBs, WRs,
and TEs all mixed together. But these positions score fantasy points in
totally different ways — a QB's baseline is ~18 pts/game mostly from
passing yards/TDs, a TE's baseline is ~8 pts/game mostly from receptions.
A single combined model has to compromise across all of these very
different scoring patterns, which is part of why one feature (recent
scoring average) dominates everything else — it's the one thing that
"works okay" across all positions, even if it's not the best signal for
any single one.

**The fix:** train four separate models — one each for QB, RB, WR, TE —
each using only that position's data. This is a real statistical
technique called *stratification* (splitting a population into
meaningfully different subgroups before modeling, instead of treating it
as one homogeneous group).

**What to figure out as you build this** (see `position_models.py`):
- How do you filter your dataset down to just one position's rows?
- Once every row is the same position, does keeping the one-hot position
  columns (`pos_QB`, `pos_RB`, etc.) as features still make sense? Why or
  why not?
- Does splitting by position help every position equally? Or does it help
  some positions (e.g. RB, where workload/touches matter a lot) more than
  others (e.g. QB, which might already be well-predicted)?

---

## Phase 2: Boom/Bust Classification

**The idea:** instead of predicting an exact point total (regression),
predict a yes/no question: *"will this player have a big week?"* This is
actually a more decision-relevant question for someone setting a fantasy
lineup — you don't need to know if someone will score exactly 14.3
points, you need to know if they're a smart start.

**Defining "boom":** a common approach is: for each position, a player
"booms" if their points that week are in the top 25% for that position.
This means the *threshold* for "boom" is different for QBs than for TEs,
since their scoring distributions are different — which is why you
compute this per-position, not as one number across everyone.

**The key statistical concept to understand — class imbalance:** if
"boom" is defined as the top 25%, then 75% of your data is "not boom."
A lazy model could just always predict "not boom" and be right 75% of
the time — that's a *useless* model with a deceptively good-looking
accuracy score. This is why classification problems like this need to be
evaluated with **precision, recall, and ROC-AUC**, not just accuracy:
- **Precision**: of the players you predicted would boom, how many
  actually did?
- **Recall**: of the players who actually boomed, how many did you catch?
- **ROC-AUC**: how well does the model rank players by boom-likelihood,
  independent of where you set the cutoff?

**What to figure out as you build this** (see `boom_classifier.py`):
- How do you compute a per-position percentile threshold?
- Why might computing that threshold using the *entire* dataset (rather
  than just the training set) be a subtle form of data leakage? (Think
  about what your model would "know" about the test set's distribution
  that it shouldn't.)
- Once you have real results, which metric matters most for *your* use
  case — would you rather have high precision (confident when you say
  "boom") or high recall (catch every real boom, even with some false
  alarms)?

---

## Phase 3 (Next): Statistical Rigor

Once Phases 1–2 are working, the next layer is about evaluation quality
itself, not new features:
- **K-fold cross-validation** instead of a single train/test split — more
  robust, and the *spread* of results across folds tells you something
  important about how stable your model actually is
- **Ridge/Lasso regression** — introduces regularization, a core
  bias-variance tradeoff concept, and will connect directly to what
  you're learning in Linear Algebra (regularization has a clean matrix
  formulation)
- **Residual analysis** — plotting predicted vs. actual and checking
  whether errors are randomly scattered (good) or show a pattern (a sign
  your model is missing something systematic)

We'll get into this once Phases 1–2 are solid.
