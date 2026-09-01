"""
Model training and evaluation for the Fantasy Football Success Predictor.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_and_evaluate(df_model, feature_cols, target_col="target_next_week_points",
                        test_size=0.2, random_state=42):
    """
    Trains a baseline Linear Regression and a Random Forest, then compares
    their performance on a held-out test set (data the model never saw
    during training — this is what makes the evaluation meaningful).
    """
    X = df_model[feature_cols]
    y = df_model[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = {}

    # Baseline model — simple and interpretable
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    results["linear_regression"] = {
        "model": lr,
        "mae": mean_absolute_error(y_test, lr_preds),
        "r2": r2_score(y_test, lr_preds),
    }

    # Improved model — can capture nonlinear patterns (e.g. workload thresholds)
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=random_state)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["random_forest"] = {
        "model": rf,
        "mae": mean_absolute_error(y_test, rf_preds),
        "r2": r2_score(y_test, rf_preds),
    }

    return results, (X_train, X_test, y_train, y_test)


def print_feature_importance(rf_model, feature_cols, top_n=10):
    """Shows which features the Random Forest relied on most — good for your write-up."""
    importances = sorted(
        zip(feature_cols, rf_model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print(f"\nTop {top_n} most predictive features:")
    for name, importance in importances[:top_n]:
        print(f"  {name}: {importance:.3f}")
