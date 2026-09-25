"""
05_train_model.py

Three analyses of Champions League results vs. playing style:

1. STYLE MATRIX -- home win / draw / away win % for every (home style,
   away style) pair, WITH the number of matches behind each cell.
   Descriptive only: with ~100 matches over 16 cells, most cells are thin.

2. STYLE-GAP ANALYSIS (headline) -- the four clusters form a ladder
   (Low-Block < Balanced < Structured Progressive < Possession Control),
   so each match gets a "style gap" = home rank - away rank (-3..+3).
   Uses every match at once. Includes a linear regression of home goal
   difference on style gap (slope, R^2, p-value).

3. PREDICTIVE CHECK -- repeated stratified cross-validation (every match
   gets used as test data in rotation) comparing:
     - baseline: predicts the overall H/D/A frequencies every time
     - style-pair model: logistic regression on both teams' style labels
     - style-gap model: logistic regression on the style gap alone
   Reported as mean accuracy and log loss (lower = better) across folds.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

STYLE_ORDER = ["Low-Block / Reactive", "Balanced / Mid-Block",
               "Structured Progressive", "Possession Control"]
OUT = config.PROCESSED_DATA_DIR


def style_matrix(df):
    rows = []
    for (h, a), g in df.groupby(["home_style", "away_style"]):
        rows.append({"home_style": h, "away_style": a, "n": len(g),
                     "home_win_pct": (g["result"] == "H").mean() * 100,
                     "draw_pct": (g["result"] == "D").mean() * 100,
                     "away_win_pct": (g["result"] == "A").mean() * 100})
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(OUT, "style_matchup_matrix.csv"), index=False)

    order = [s for s in STYLE_ORDER if s in set(df["home_style"]) | set(df["away_style"])]
    win = long.pivot(index="home_style", columns="away_style", values="home_win_pct").reindex(index=order, columns=order)
    n = long.pivot(index="home_style", columns="away_style", values="n").reindex(index=order, columns=order)
    win.to_csv(os.path.join(OUT, "home_win_rate_matrix.csv"))
    n.to_csv(os.path.join(OUT, "matchup_counts_matrix.csv"))

    print("1) HOME WIN % BY STYLE MATCHUP  (n = matches in that cell)")
    display = win.round(0).astype("Int64").astype(str) + "% (n=" + n.fillna(0).astype(int).astype(str) + ")"
    display = display.where(n.notna(), "-")
    print(display.to_string())
    print(f"   Cells with fewer than 5 matches: {(n < 5).sum().sum()} of {n.notna().sum().sum()} "
          f"-- treat those percentages as anecdotes, not findings.\n")


def gap_analysis(df):
    rank = {s: i for i, s in enumerate(STYLE_ORDER)}
    df["style_gap"] = df["home_style"].map(rank) - df["away_style"].map(rank)
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    tbl = df.groupby("style_gap").agg(
        n=("result", "size"),
        home_win_pct=("result", lambda r: (r == "H").mean() * 100),
        draw_pct=("result", lambda r: (r == "D").mean() * 100),
        away_win_pct=("result", lambda r: (r == "A").mean() * 100),
        avg_goal_diff=("goal_diff", "mean"),
    ).round(1)
    tbl.to_csv(os.path.join(OUT, "style_gap_results.csv"))

    print("2) RESULTS BY STYLE GAP  (home style rank minus away style rank;")
    print("   positive = home team is higher up the possession/control ladder)")
    print(tbl.to_string())

    reg = stats.linregress(df["style_gap"], df["goal_diff"])
    print(f"\n   Linear regression: home goal difference ~ style gap  (n={len(df)})")
    print(f"     slope     = {reg.slope:+.3f} goals per step up the style ladder")
    print(f"     intercept = {reg.intercept:+.3f} (home advantage when styles are equal)")
    print(f"     R^2       = {reg.rvalue ** 2:.3f}")
    print(f"     p-value   = {reg.pvalue:.4f}  ({'significant' if reg.pvalue < 0.05 else 'NOT significant'} at 0.05)\n")
    pd.DataFrame([{"n": len(df), "slope": reg.slope, "intercept": reg.intercept,
                   "r_squared": reg.rvalue ** 2, "p_value": reg.pvalue}]).to_csv(
        os.path.join(OUT, "style_gap_regression.csv"), index=False)
    return df


def cross_validate(df):
    y = df["result"].values
    X_pair = OneHotEncoder(handle_unknown="ignore").fit_transform(df[["home_style", "away_style"]])
    X_gap = df[["style_gap"]].values

    models = {
        "Baseline (overall H/D/A rates)": (DummyClassifier(strategy="prior"), X_gap),
        "Style-pair model": (LogisticRegression(max_iter=1000), X_pair),
        "Style-gap model": (LogisticRegression(max_iter=1000), X_gap),
    }
    min_class = pd.Series(y).value_counts().min()
    folds = max(2, min(5, min_class))
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=20, random_state=config.RANDOM_STATE)

    rows = []
    for name, (model, X) in models.items():
        accs, losses = [], []
        for tr, te in cv.split(X_gap, y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X[tr], y[tr])
                proba = model.predict_proba(X[te])
            accs.append(accuracy_score(y[te], model.classes_[proba.argmax(axis=1)]))
            losses.append(log_loss(y[te], proba, labels=model.classes_))
        rows.append({"model": name, "accuracy_mean": np.mean(accs), "accuracy_sd": np.std(accs),
                     "log_loss_mean": np.mean(losses), "log_loss_sd": np.std(losses)})

    res = pd.DataFrame(rows).set_index("model").round(3)
    res.to_csv(os.path.join(OUT, "model_comparison.csv"))
    print(f"3) PREDICTIVE CHECK  ({folds}-fold cross-validation x 20 repeats; log loss: lower = better)")
    print(res.to_string())
    print("\n   Most-predicted outcome for the baseline is always 'home win', so its accuracy")
    print("   equals the home-win rate. Log loss is the fairer comparison: it rewards")
    print("   well-calibrated probabilities, not just picking the most common result.")


def main():
    pd.set_option("display.width", 200)
    df = pd.read_csv(os.path.join(OUT, "cl_matchup_dataset.csv"))
    print(f"Loaded {len(df)} matches.  Overall: "
          f"{(df['result'] == 'H').mean():.0%} home wins, {(df['result'] == 'D').mean():.0%} draws, "
          f"{(df['result'] == 'A').mean():.0%} away wins\n")
    style_matrix(df)
    df = gap_analysis(df)
    cross_validate(df)


if __name__ == "__main__":
    main()
