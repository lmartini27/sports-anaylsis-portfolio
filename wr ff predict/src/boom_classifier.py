"""
Boom/bust classification: instead of predicting an exact point total,
predict whether a player will "boom" (score in the top percentile for
their position that week) — a more decision-relevant question than an
exact number for someone setting a fantasy lineup.

See EXTENSION_ROADMAP.md for the full explanation, including why class
imbalance matters here. Fill in the TODOs below.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def add_boom_label(df_model, position_prefix="pos_", threshold_percentile=75):
    """
    Creates a binary `boom` column: 1 if target_next_week_points is at or
    above the Nth percentile FOR THAT PLAYER'S POSITION, else 0.

    Start simple: compute the threshold using the whole df_model to get
    this working first. Once it works, come back and think about why
    computing this threshold using your TEST set's data (not just
    training data) could be a subtle form of data leakage — and how
    you'd restructure this function to avoid it.
    """
    df = df_model.copy()
    position_cols = [c for c in df.columns if c.startswith(position_prefix)]

    df["boom"] = 0  # default, overwritten below

    for pos_col in position_cols:
        # TODO 1: get the subset of rows for this position
        # Hint: df[df[pos_col] == 1]
        position_rows = None

        # TODO 2: compute the threshold_percentile of
        # target_next_week_points within this subset.
        # Hint: pandas Series has a .quantile() method — quantile(0.75)
        # is the same thing as the 75th percentile.
        threshold = None

        # TODO 3: for rows in this position, set boom = 1 where
        # target_next_week_points >= threshold.
        # Hint: you'll need to update df.loc[...] using position_rows'
        # index and the condition on target_next_week_points.

    return df


def train_boom_classifier(df_model, feature_cols, target_col="boom",
                           test_size=0.2, random_state=42):
    """
    Trains a Logistic Regression to classify boom vs. not-boom.

    TODO 1: split df_model into train/test using train_test_split,
    same pattern as train_and_evaluate() in model.py.

    TODO 2: fit a LogisticRegression on the training data.

    TODO 3: evaluate using FOUR metrics on the test set:
        - accuracy_score
        - precision_score
        - recall_score
        - roc_auc_score (needs predicted PROBABILITIES, not just the
          predicted class — look up `.predict_proba()` on your fitted
          model, and use the probability of class 1)

    Print all four. Before you run it: what accuracy would a model get by
    just always predicting "not boom"? Compare that number to what your
    real model achieves — that comparison is the honest way to judge
    whether this model is actually doing anything useful.
    """
    pass
