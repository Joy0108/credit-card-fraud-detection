"""Benchmark models on exactly the same split and protocol as the Random Forest.

Each benchmark gets its own out-of-fold F1-optimal threshold (3-fold CV on
the training set) before it sees the test set, so the comparison is between
operating points chosen the same way - not between one tuned threshold and a
set of default 0.5 cut-offs.

Run with ``python -m fraud_detection baselines`` (after ``train``).
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from . import evaluation as E
from .data import load_raw, split_train_test, xy
from .features import FeatureBuilder

log = logging.getLogger(__name__)


class IsolationForestScorer:
    """Unsupervised anomaly detector exposing a predict_proba-like score."""

    def __init__(self):
        self.iforest = IsolationForest(n_estimators=300, max_samples=4096, contamination="auto",
                                       n_jobs=-1, random_state=C.RANDOM_STATE)

    def fit(self, X, y=None):
        self.iforest.fit(X)                     # labels are never used
        s = -self.iforest.score_samples(X)
        self.lo_, self.hi_ = s.min(), s.max()
        return self

    def predict_proba(self, X):
        s = (-self.iforest.score_samples(X) - self.lo_) / (self.hi_ - self.lo_)
        s = np.clip(s, 0, 1)
        return np.c_[1 - s, s]


def models() -> dict:
    rs = C.RANDOM_STATE
    return {
        "Majority class (always legitimate)": lambda: DummyClassifier(strategy="prior"),
        "Isolation Forest (unsupervised)": IsolationForestScorer,
        "Logistic Regression": lambda: Pipeline([
            ("features", FeatureBuilder("cyclic")), ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, C=0.1))]),
        "Decision Tree": lambda: Pipeline([
            ("features", FeatureBuilder("raw")),
            ("model", DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=rs))]),
        "Hist. Gradient Boosting": lambda: Pipeline([
            ("features", FeatureBuilder("raw")),
            ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                                     l2_regularization=1.0, early_stopping=True,
                                                     random_state=rs))]),
    }


def f1_threshold(y, score) -> float:
    """F1-optimal cut-off on arbitrary scores (works for non-probabilities)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    if len(t) == 0:
        return 0.5
    return float(t[int(np.argmax(f1))])


def run() -> pd.DataFrame:
    df = load_raw()
    train, test, _ = split_train_test(df)
    X, y = xy(train)
    Xt, yt = xy(test)
    rows, scores = [], {"y": yt}
    for name, make in models().items():
        log.info("Benchmark: %s", name)
        t0 = time.perf_counter()
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=C.RANDOM_STATE).split(X, y):
            oof[va] = make().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        thr = f1_threshold(y, oof) if "Majority" not in name else 0.5
        model = make().fit(X, y)
        prob = model.predict_proba(Xt)[:, 1]
        scores[name] = prob
        m = E.full_report(yt, prob, thr)
        m.update(model=name, seconds=time.perf_counter() - t0)
        rows.append(m)
        log.info("  P=%.3f R=%.3f F1=%.3f MCC=%.3f PR-AUC=%.3f", m["precision"], m["recall"], m["f1"],
                 m["mcc"], m["pr_auc"])

    rf = json.loads((C.RESULTS_DIR / "final_model_results.json").read_text())
    rf_test = dict(rf["test"]["at_threshold"], model="Random Forest (tuned)")
    rows.append(rf_test)
    rf_pred = pd.read_csv(C.RESULTS_DIR / "test_predictions.csv")
    scores["Random Forest (tuned)"] = rf_pred["fraud_probability"].to_numpy()

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / "baseline_comparison.csv", index=False)
    np.savez_compressed(C.RESULTS_DIR / "baseline_test_scores.npz",
                        **{k.replace(" ", "_").replace(".", "").replace("(", "").replace(")", ""): v
                           for k, v in scores.items()})
    return res
