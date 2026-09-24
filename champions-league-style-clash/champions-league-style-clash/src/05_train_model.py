"""
05_train_model.py

Two things happen here:

1. THE HEADLINE RESULT: a style-vs-style win-rate matrix -- for every
   pairing of (home style, away style), what share of matches did the
   home side win/draw/lose? This is the direct, descriptive answer to
   "which styles struggle against which."

2. A predictive check: does knowing both teams' style archetypes (plus
   home advantage) predict the match result better than just guessing
   the most common outcome? This is deliberately a modest model -- no
   external strength/Elo rating is folded in here (see REPORT.md
   limitations), so treat any lift over baseline as a lower bound on
   how much style matters, not the full picture.
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import OneHotEncoder

sys.path.append("..")
import config


def main():
    df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/cl_matchup_dataset.csv")
    print(f"Loaded {df.shape[0]} matches with style labels on both sides.\n")

    # --- 1. style-vs-style win rate matrix ------------------------------------
    result_map = {"H": "Home win", "D": "Draw", "A": "Away win"}
    df["result_label"] = df["result"].map(result_map)

    matrix = (
        pd.crosstab(
            [df["home_style"]], [df["away_style"], df["result_label"]],
            normalize="index",
        ) * 100
    ).round(1)
    print("Style vs. style outcome matrix (% of matches, home side's perspective):")
    print(matrix)
    matrix.to_csv(f"{config.PROCESSED_DATA_DIR}/style_matchup_matrix.csv")

    # simpler version: home win rate only, easy to eyeball / heatmap in step 06
    home_win_rate = (
        df.assign(home_win=(df["result"] == "H").astype(int))
        .pivot_table(index="home_style", columns="away_style", values="home_win", aggfunc="mean")
        * 100
    ).round(1)
    home_win_rate.to_csv(f"{config.PROCESSED_DATA_DIR}/home_win_rate_matrix.csv")
    print("\nHome win rate (%) by style matchup, saved to home_win_rate_matrix.csv:")
    print(home_win_rate)

    # --- 2. predictive check ---------------------------------------------------
    features = df[["home_style", "away_style"]]
    target = df["result"]

    encoder = OneHotEncoder(handle_unknown="ignore")
    X = encoder.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.25, random_state=config.RANDOM_STATE, stratify=target
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

    model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model.fit(X_train, y_train)
    model_acc = accuracy_score(y_test, model.predict(X_test))
    model_logloss = log_loss(y_test, model.predict_proba(X_test), labels=model.classes_)

    print(f"\nBaseline (always predict most common result) accuracy: {baseline_acc:.3f}")
    print(f"Style-matchup model accuracy: {model_acc:.3f}")
    print(f"Style-matchup model log loss: {model_logloss:.3f}")
    print("\nFull classification report (style-matchup model):")
    print(classification_report(y_test, model.predict(X_test)))

    print(
        "\nInterpretation: if model_acc is only marginally above baseline_acc, "
        "style alone (without a strength/quality signal) isn't doing much "
        "predictive work on its own -- expected, and worth saying plainly in "
        "REPORT.md. The win-rate matrix above is the more honest headline "
        "result either way: it shows real matchup effects even where the "
        "classifier's raw accuracy lift is small, because a few percentage "
        "points of extra win probability in one direction is a real edge, "
        "just not enough to flip most individual predictions."
    )


if __name__ == "__main__":
    main()
