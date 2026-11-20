"""Randomized hyper-parameter search for the Random Forest.

The search is scored on six metrics at once - precision, recall, F1, MCC,
balanced accuracy and average precision - so every candidate's
precision/recall trade-off is visible, not just a single winner. The winning
configuration is the one with the best mean cross-validated MCC, the
"balanced" metric that rewards precision and recall together and is not
fooled by the 99.8 % majority class.

Run with ``python -m fraud_detection tune``.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config as C
from .data import load_raw, split_train_test, xy
from .features import FeatureBuilder

log = logging.getLogger(__name__)

FEATURE_SET = "raw"   # 30 inputs: Time, V1..V28, Amount (see imbalance ablation)

SEARCH_SPACE = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 16, 24],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.3],
    "model__criterion": ["gini", "entropy"],
    "model__class_weight": [None, "balanced_subsample", {0: 1, 1: 5}, {0: 1, 1: 25}],
    "model__max_samples": [None, 0.5, 0.75],
}

SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "mcc": make_scorer(matthews_corrcoef),
    "balanced_accuracy": "balanced_accuracy",
    "average_precision": "average_precision",
}
SELECTION_METRIC = "mcc"


def base_pipeline() -> Pipeline:
    return Pipeline([
        ("features", FeatureBuilder(FEATURE_SET)),
        ("model", RandomForestClassifier(n_jobs=-1, random_state=C.RANDOM_STATE)),
    ])


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def describe_class_weight(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, str):
        return v
    return f"{{0:{v[0]}, 1:{v[1]}}}"


def run() -> dict:
    df = load_raw()
    train, _test, _ = split_train_test(df)
    X, y = xy(train)

    search = RandomizedSearchCV(
        base_pipeline(),
        param_distributions=SEARCH_SPACE,
        n_iter=C.TUNING_ITERATIONS,
        scoring=SCORING,
        refit=False,                 # the train stage refits on the full training set
        cv=StratifiedKFold(C.TUNING_FOLDS, shuffle=True, random_state=C.RANDOM_STATE),
        random_state=C.RANDOM_STATE,
        n_jobs=1,                    # parallelism lives inside the forest (n_jobs=-1)
        verbose=2,
        error_score="raise",
        return_train_score=False,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - t0

    res = pd.DataFrame(search.cv_results_)
    keep = ["params", "mean_fit_time"] + [c for c in res.columns if c.startswith(("mean_test_", "std_test_"))]
    tbl = res[keep].copy()
    params = pd.DataFrame([
        {k.replace("model__", ""): (describe_class_weight(v) if k.endswith("class_weight")
                                    else ("None" if v is None else v)) for k, v in prm.items()}
        for prm in tbl.pop("params")
    ])
    tbl = pd.concat([params, tbl], axis=1)
    tbl = tbl.sort_values(f"mean_test_{SELECTION_METRIC}", ascending=False).reset_index(drop=True)
    tbl.insert(0, "rank", np.arange(1, len(tbl) + 1))
    tbl.to_csv(C.RESULTS_DIR / "tuning_results.csv", index=False)

    best_idx = int(np.argmax(res[f"mean_test_{SELECTION_METRIC}"]))
    best_params = {k.replace("model__", ""): _jsonable(v) for k, v in res["params"][best_idx].items()}
    out = {
        "selection_metric": SELECTION_METRIC,
        "n_iter": C.TUNING_ITERATIONS,
        "cv_folds": C.TUNING_FOLDS,
        "n_fits": C.TUNING_ITERATIONS * C.TUNING_FOLDS,
        "search_seconds": elapsed,
        "feature_set": FEATURE_SET,
        "search_space": {k.replace("model__", ""): [describe_class_weight(x) if isinstance(x, dict) or x is None else x
                                                   for x in v] for k, v in SEARCH_SPACE.items()},
        "best_params": best_params,
        "best_cv": {m: float(res[f"mean_test_{m}"][best_idx]) for m in SCORING},
        "best_cv_std": {m: float(res[f"std_test_{m}"][best_idx]) for m in SCORING},
    }
    (C.RESULTS_DIR / "tuning_best.json").write_text(json.dumps(out, indent=2))
    log.info("Best params (%s=%.4f): %s", SELECTION_METRIC, out["best_cv"][SELECTION_METRIC], best_params)
    return out
