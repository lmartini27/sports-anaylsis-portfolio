"""
Model training and evaluation, using a WALK-FORWARD time split:
each test season is predicted by models trained only on earlier seasons
(e.g. train 2021 -> test 2022, then train 2021-22 -> test 2023). This
mimics real use: you never get to train on the future.

Every model is compared against a NAIVE BASELINE that simply predicts a
player's recent average -- the bar any real model has to clear.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def _score(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}


def train_and_evaluate(df_model, feature_cols, baseline_col,
                       target_col="target_next_week_points", random_state=42):
    """
    Returns (results, last_rf):
      results -- {model_name: {"mae", "r2", "folds": [{season, mae, r2, n}]}}
                 where mae/r2 are averages over the test seasons
      last_rf -- Random Forest from the final fold (for feature importance)
    """
    seasons = sorted(df_model["season"].unique())
    if len(seasons) < 2:
        raise ValueError("Need at least 2 seasons for a walk-forward split.")

    folds = {"naive_baseline": [], "linear_regression": [], "random_forest": []}
    last_rf = None

    for test_season in seasons[1:]:
        train = df_model[df_model["season"] < test_season]
        test = df_model[df_model["season"] == test_season]
        X_tr, y_tr = train[feature_cols], train[target_col]
        X_te, y_te = test[feature_cols], test[target_col]

        preds = {"naive_baseline": test[baseline_col].values}

        lr = LinearRegression().fit(X_tr, y_tr)
        preds["linear_regression"] = lr.predict(X_te)

        rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                   random_state=random_state, n_jobs=-1).fit(X_tr, y_tr)
        preds["random_forest"] = rf.predict(X_te)
        last_rf = rf

        for name, p in preds.items():
            folds[name].append({"season": test_season, "n": len(test), **_score(y_te, p)})

    results = {
        name: {"mae": np.mean([f["mae"] for f in fs]),
               "r2": np.mean([f["r2"] for f in fs]),
               "folds": fs}
        for name, fs in folds.items()
    }
    return results, last_rf


def print_results(results, title):
    print(f"\n--- {title} ---")
    for name, r in results.items():
        per_fold = ", ".join(f"{f['season']}: R^2={f['r2']:.3f}" for f in r["folds"])
        print(f"{name:18s} MAE = {r['mae']:.2f} pts   R^2 = {r['r2']:.3f}   ({per_fold})")


def print_feature_importance(rf_model, feature_cols, top_n=12):
    importances = sorted(zip(feature_cols, rf_model.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} most predictive features (Random Forest, final fold):")
    for name, imp in importances[:top_n]:
        print(f"  {name}: {imp:.3f}")
