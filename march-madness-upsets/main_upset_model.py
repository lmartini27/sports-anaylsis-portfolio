"""
main_upset_model.py — the core test: does roster continuity predict
tournament upsets BEYOND what seed difference already captures?

Merges tournament_games.csv with roster_continuity.csv, then compares
THREE models using a TIME-BASED train/test split (train on earlier
seasons, test on the most recent ones — not a random split, per the
lesson from the kicker project):

  A) Baseline: seed difference only
  B) Seed + continuity_diff (underdog's continuity minus favorite's)
  C) Seed + each team's continuity separately (not combined with B,
     to avoid redundant/collinear features in the same model)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

TOURNAMENT_CSV = "tournament_games.csv"
CONTINUITY_CSV = "roster_continuity.csv"

TEST_SEASONS = [2024, 2025]


def evaluate_model(train, test, features, label):
    model = LogisticRegression()
    model.fit(train[features], train["upset"])
    probs = model.predict_proba(test[features])[:, 1]
    auc = roc_auc_score(test["upset"], probs)
    acc = accuracy_score(test["upset"], model.predict(test[features]))
    print(f"--- {label} ---")
    print(f"ROC-AUC: {auc:.3f}   Accuracy: {acc:.3f}")
    for name, coef in zip(features, model.coef_[0]):
        print(f"  {name}: {coef:+.4f}")
    print()
    return auc


def main():
    games = pd.read_csv(TOURNAMENT_CSV)
    continuity = pd.read_csv(CONTINUITY_CSV)
    print(f"Loaded {len(games)} tournament games, {len(continuity)} team-season continuity rows.\n")

    games = games.merge(
        continuity[["season", "team", "continuity_pct"]].rename(
            columns={"team": "favorite_team", "continuity_pct": "favorite_continuity"}
        ),
        on=["season", "favorite_team"], how="left",
    )
    games = games.merge(
        continuity[["season", "team", "continuity_pct"]].rename(
            columns={"team": "underdog_team", "continuity_pct": "underdog_continuity"}
        ),
        on=["season", "underdog_team"], how="left",
    )

    games["seed_diff"] = games["underdog_seed"] - games["favorite_seed"]
    games["continuity_diff"] = games["underdog_continuity"] - games["favorite_continuity"]

    before_drop = len(games)
    games = games.dropna(subset=["favorite_continuity", "underdog_continuity"])
    print(f"{len(games)} of {before_drop} games have continuity data for both teams "
          f"(some drop is expected in early seasons with no prior-year data).\n")

    train = games[~games["season"].isin(TEST_SEASONS)]
    test = games[games["season"].isin(TEST_SEASONS)]
    print(f"Train: {len(train)} games (before {min(TEST_SEASONS)})")
    print(f"Test:  {len(test)} games ({TEST_SEASONS})\n")

    if len(test) == 0 or train["upset"].nunique() < 2:
        print("Not enough data to train/test properly — check season ranges.")
        return

    auc_a = evaluate_model(train, test, ["seed_diff"], "Model A: Baseline (seed only)")
    auc_b = evaluate_model(train, test, ["seed_diff", "continuity_diff"], "Model B: Seed + continuity_diff")
    auc_c = evaluate_model(train, test, ["seed_diff", "favorite_continuity", "underdog_continuity"],
                            "Model C: Seed + both teams' continuity")

    print("--- The key question ---")
    print(f"Model B improvement over baseline: {auc_b - auc_a:+.3f}")
    print(f"Model C improvement over baseline: {auc_c - auc_a:+.3f}")


if __name__ == "__main__":
    main()
