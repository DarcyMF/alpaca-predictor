"""Walk-forward training: only ever predict a day using a model trained on days before it.

A normal train/test split on shuffled rows would let the model learn from the future,
which makes almost any stock model look amazing. Instead we train on everything up to a
month, predict that month, then move forward a month and retrain.
"""

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

# about 3 years of trading days before the first prediction
MIN_TRAIN_DAYS = 750
# retrain about once a month
STEP_DAYS = 21


def make_model(name: str) -> ClassifierMixin:
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    if name == "boosting":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0)
    raise ValueError(f"Unknown model {name!r}. Use one of {MODELS}")


MODELS = ["logistic", "boosting"]
BASELINES = ["always_up", "momentum"]


def walk_forward(df: pd.DataFrame, name: str, min_train: int = MIN_TRAIN_DAYS, step: int = STEP_DAYS) -> pd.Series:
    """Probability that each day is followed by an up day, using only earlier data.

    Days in the first `min_train` rows get no prediction.
    """
    if name in BASELINES:
        return baseline(df, name).iloc[min_train:]

    preds = []
    for start in range(min_train, len(df), step):
        # Row start-1's target is known at the close of day `start`, which is when we predict.
        train = df.iloc[:start].dropna(subset=["target"])
        test = df.iloc[start : start + step]
        model = make_model(name)
        model.fit(train[FEATURES], train["target"].astype(int))
        prob = model.predict_proba(test[FEATURES])[:, 1]
        preds.append(pd.Series(prob, index=test.index))
    if not preds:
        raise ValueError(f"Need more than {min_train} days of data, got {len(df)}")
    return pd.concat(preds).rename(name)


def baseline(df: pd.DataFrame, name: str) -> pd.Series:
    """Simple rules any model has to beat to be worth anything."""
    if name == "always_up":
        return pd.Series(1.0, index=df.index, name=name)
    if name == "momentum":
        # tomorrow goes the same way as today
        return (df["ret_1"] > 0).astype(float).rename(name)
    raise ValueError(name)


def fit_latest(df: pd.DataFrame, name: str) -> float:
    """Train on all known days and give the probability for the day after the last bar."""
    if name in BASELINES:
        return float(baseline(df, name).iloc[-1])
    train = df.dropna(subset=["target"])
    model = make_model(name)
    model.fit(train[FEATURES], train["target"].astype(int))
    return float(model.predict_proba(df[FEATURES].iloc[[-1]])[0, 1])
