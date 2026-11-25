import numpy as np
import pandas as pd

from predictor import model
from predictor.data import synthetic_bars
from predictor.features import FEATURES, build


def small_df(days=500):
    return build(synthetic_bars(days, seed=2))


def test_walk_forward_only_predicts_after_training_window():
    df = small_df()
    prob = model.walk_forward(df, "logistic", min_train=200, step=50)

    assert prob.index[0] == df.index[200]
    assert prob.index[-1] == df.index[-1]
    assert prob.between(0, 1).all()


def test_walk_forward_never_trains_on_the_day_it_predicts(monkeypatch):
    """Record what each model was trained on and check it all comes before the test days."""
    df = small_df()
    seen = []
    real = model.make_model

    def spy(name):
        m = real(name)
        fit = m.fit

        def record(X, y):
            seen.append(X.index.max())
            return fit(X, y)

        m.fit = record
        return m

    monkeypatch.setattr(model, "make_model", spy)
    model.walk_forward(df, "logistic", min_train=200, step=50)

    starts = [df.index[i] for i in range(200, len(df), 50)]
    assert len(seen) == len(starts)
    for last_train_day, first_test_day in zip(seen, starts):
        assert last_train_day < first_test_day


def test_model_learns_a_real_pattern():
    """If tomorrow really does follow today, the model should find it."""
    rng = np.random.default_rng(0)
    n = 1200
    ret = np.zeros(n)
    for t in range(1, n):
        ret[t] = 0.6 * np.sign(ret[t - 1]) * 0.01 + rng.normal(0, 0.008)
    close = 100 * np.cumprod(1 + ret)
    idx = pd.bdate_range("2015-01-01", periods=n)
    bars = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)
    df = build(bars)

    prob = model.walk_forward(df, "logistic", min_train=400, step=100)
    days = df.loc[prob.index].dropna(subset=["target"])
    accuracy = ((prob.loc[days.index] > 0.5) == (days["target"] == 1)).mean()
    assert accuracy > 0.65


def test_baselines():
    df = small_df()
    assert (model.baseline(df, "always_up") == 1).all()
    momentum = model.baseline(df, "momentum")
    assert ((momentum == 1) == (df["ret_1"] > 0)).all()


def test_fit_latest_gives_a_probability():
    p = model.fit_latest(small_df(), "boosting")
    assert 0 <= p <= 1


def test_features_list_matches_columns():
    assert set(FEATURES) <= set(small_df().columns)
