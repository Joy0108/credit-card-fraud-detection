"""Fast unit tests on synthetic data (no dataset download needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

from fraud_detection import config as C
from fraud_detection import evaluation as E
from fraud_detection.data import _row_hash, split_train_test, validate
from fraud_detection.features import FeatureBuilder, hour_of_day


@pytest.fixture()
def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=C.PCA_FEATURES)
    df.insert(0, "Time", rng.uniform(0, 2 * 86400, n).round())
    df["Amount"] = rng.exponential(80, n).round(2)
    df["Class"] = (rng.random(n) < 0.02).astype("int8")
    df.loc[df.Class == 1, "V14"] -= 4                 # make fraud learnable
    dupes = df.sample(200, random_state=1)            # inject exact duplicates
    return pd.concat([df, dupes], ignore_index=True)


def test_validate_rejects_missing_columns(synthetic):
    with pytest.raises(ValueError):
        validate(synthetic.drop(columns="V3"))


def test_split_is_stratified_and_leak_free(synthetic):
    train, test, s = split_train_test(synthetic)
    assert s.n_test == len(test)
    assert not train.duplicated().any()
    assert not _row_hash(train).isin(set(_row_hash(test))).any()
    assert abs(test.Class.mean() - synthetic.Class.mean()) < 0.01
    assert s.n_train == s.n_train_raw - s.train_internal_duplicates - s.train_rows_matching_test


def test_hour_of_day_wraps():
    assert hour_of_day(np.array([0, 3600, 86400 + 7200]))[2] == pytest.approx(2.0)


@pytest.mark.parametrize("fs,n", [("raw", 30), ("engineered", 30), ("cyclic", 31)])
def test_feature_builder_shapes(synthetic, fs, n):
    out = FeatureBuilder(fs).fit(synthetic[C.RAW_FEATURES]).transform(synthetic[C.RAW_FEATURES])
    assert out.shape[1] == n
    assert list(out.columns) == list(FeatureBuilder(fs).fit(synthetic).get_feature_names_out())
    if fs == "engineered":
        assert out["Hour"].between(0, 24).all()


def test_metrics_match_sklearn():
    rng = np.random.default_rng(3)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.where(rng.random(5000) < 0.9, y, 1 - y)
    m = E.metrics_from_counts(**E.counts(y, p))
    assert m["precision"] == pytest.approx(precision_score(y, p))
    assert m["recall"] == pytest.approx(recall_score(y, p))
    assert m["f1"] == pytest.approx(f1_score(y, p))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, p))


def test_threshold_selection_prefers_higher_on_ties():
    y = np.array([0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert E.select_threshold(y, prob) == pytest.approx(0.9)


def test_resampling_happens_only_at_fit_time(synthetic):
    X, y = synthetic[C.RAW_FEATURES], synthetic["Class"].to_numpy()
    pipe = Pipeline([("features", FeatureBuilder("raw")),
                     ("sampler", SMOTE(sampling_strategy=0.5, random_state=0)),
                     ("model", RandomForestClassifier(n_estimators=10, random_state=0))]).fit(X, y)
    assert pipe.predict_proba(X).shape == (len(X), 2)   # no synthetic rows at inference


def test_bootstrap_interval_contains_point_estimate():
    ci = E.bootstrap_from_counts(tp=80, fp=4, fn=20, tn=50_000, n_boot=300)
    point = E.metrics_from_counts(80, 4, 20, 50_000)
    for k in ("precision", "recall", "f1", "mcc"):
        assert ci[k][0] <= point[k] <= ci[k][1]
