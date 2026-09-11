"""
accuracy_model.py — predicts whether an INDIVIDUAL field goal attempt is
made, using distance and weather/venue conditions. This tests whether FG
accuracy specifically (as opposed to total weekly points) is predictable.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def load_fg_attempts_data(csv_path="fg_attempts_data.csv"):
    return pd.read_csv(csv_path)


def prepare_accuracy_features(df):
    """
    Builds model-ready features from raw attempt data:
      - kick_distance (the single strongest predictor, definitionally)
      - is_dome (domes eliminate weather as a factor entirely)
      - wind and temp, with sensible fill values for dome games (which
        often have no wind/temp recorded at all)
    """
    df = df.copy()

    if "roof" in df.columns:
        df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    else:
        df["is_dome"] = 0

    if "wind" in df.columns:
        df["wind_filled"] = df["wind"].fillna(0)
        df.loc[df["is_dome"] == 1, "wind_filled"] = 0
    else:
        df["wind_filled"] = 0

    if "temp" in df.columns:
        df["temp_filled"] = df["temp"].fillna(70)
        df.loc[df["is_dome"] == 1, "temp_filled"] = 70
    else:
        df["temp_filled"] = 70

    feature_cols = ["kick_distance", "is_dome", "wind_filled", "temp_filled"]
    df_model = df.dropna(subset=["kick_distance", "made"])

    return df_model, feature_cols


def train_accuracy_model(df_model, feature_cols, target_col="made",
                          test_size=0.2, random_state=42):
    """
    Trains a Logistic Regression to predict make/miss, and reports it
    against a NAIVE baseline (always guessing the majority class) — the
    real test of whether this model is doing anything useful.
    """
    X = df_model[feature_cols]
    y = df_model[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    naive_baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    results = {
        "model": model,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "naive_baseline_accuracy": naive_baseline_acc,
    }
    return results, (X_train, X_test, y_train, y_test)


def print_accuracy_results(results, feature_cols):
    print(f"Naive baseline accuracy (always guess majority class): {results['naive_baseline_accuracy']:.3f}")
    print(f"Model accuracy:                                        {results['accuracy']:.3f}")
    print(f"Model precision:                                       {results['precision']:.3f}")
    print(f"Model recall:                                          {results['recall']:.3f}")
    print(f"Model ROC-AUC:                                         {results['roc_auc']:.3f}")
    print("\nCoefficients (positive = increases make probability):")
    for name, coef in zip(feature_cols, results["model"].coef_[0]):
        print(f"  {name}: {coef:.4f}")
