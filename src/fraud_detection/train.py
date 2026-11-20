"""Final model: threshold selection, full-train fit, hold-out evaluation.

Steps
1. Rebuild the best pipeline found by ``tune``.
2. 5-fold out-of-fold (OOF) probabilities on the training set -> choose the
   decision threshold that maximises F1. The test set plays no part in this.
3. Fit the pipeline on the whole (leakage-cleaned) training set and save it.
4. Score the untouched 20 % hold-out once and record every metric, bootstrap
   confidence intervals, and feature importances (MDI + permutation).

Run with ``python -m fraud_detection train``.
"""
from __future__ import annotations

import json
import logging
from datetime import date

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from . import __version__
from . import config as C
from . import evaluation as E
from .data import load_raw, split_train_test, xy
from .features import FeatureBuilder
from .imbalance import cross_validate_oof

log = logging.getLogger(__name__)

THRESHOLD_METRIC = "f1"


def load_best_params() -> tuple[dict, str]:
    best = json.loads((C.RESULTS_DIR / "tuning_best.json").read_text())
    params = dict(best["best_params"])
    cw = params.get("class_weight")
    if isinstance(cw, dict):
        params["class_weight"] = {int(k): float(v) for k, v in cw.items()}
    return params, best.get("feature_set", "raw")


def build_model(params: dict, feature_set: str = "raw") -> Pipeline:
    return Pipeline([
        ("features", FeatureBuilder(feature_set)),
        ("model", RandomForestClassifier(n_jobs=-1, random_state=C.RANDOM_STATE, **params)),
    ])


def run() -> dict:
    df = load_raw()
    train, test, split = split_train_test(df)
    X, y = xy(train)
    Xt, yt = xy(test)
    params, feature_set = load_best_params()
    log.info("Best params: %s (features=%s)", params, feature_set)

    # 1) threshold from out-of-fold scores
    oof, folds, fit_s, _ = cross_validate_oof(lambda: build_model(params, feature_set), X, y)
    curve = E.threshold_curve(y, oof)
    curve.to_csv(C.RESULTS_DIR / "threshold_curve_oof.csv", index=False)
    folds.to_csv(C.RESULTS_DIR / "final_cv_folds.csv", index=False)
    threshold = E.select_threshold(y, oof, metric=THRESHOLD_METRIC)
    oof_summary = {
        "threshold": threshold,
        "at_threshold": E.full_report(y, oof, threshold),
        "at_0.5": E.full_report(y, oof, 0.5),
        "fold_mean": folds.drop(columns="fold").mean().to_dict(),
        "fold_std": folds.drop(columns="fold").std().to_dict(),
        "fit_seconds_per_fold": fit_s,
    }
    np.save(C.RESULTS_DIR / "final_oof.npy", oof)
    log.info("OOF threshold=%.2f -> %s", threshold,
             {k: round(v, 4) for k, v in oof_summary["at_threshold"].items() if isinstance(v, float)})

    # 2) final fit on the full training set
    model = build_model(params, feature_set).fit(X, y)
    rf = model.named_steps["model"]
    joblib.dump({"pipeline": model, "threshold": threshold, "features": C.RAW_FEATURES,
                 "version": __version__}, C.MODEL_PATH, compress=3)

    # 3) one look at the hold-out set
    prob = model.predict_proba(Xt)[:, 1]
    test_metrics = {
        "at_threshold": E.full_report(yt, prob, threshold),
        "at_0.5": E.full_report(yt, prob, 0.5),
        "bootstrap_95ci": {
            **E.bootstrap_from_counts(**E.counts(yt, prob >= threshold), n_boot=C.N_BOOTSTRAP),
            **E.bootstrap_ranking(yt, prob, n_boot=C.N_BOOTSTRAP),
        },
        "threshold": threshold,
    }
    pred = test[[*C.RAW_FEATURES, C.TARGET]].copy()
    pred["fraud_probability"] = prob
    pred.to_csv(C.RESULTS_DIR / "test_predictions.csv", index=False)

    # 4) interpretation
    names = model.named_steps["features"].get_feature_names_out()
    perm = permutation_importance(model, Xt, yt, scoring="average_precision", n_repeats=5,
                                  random_state=C.RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({
        "feature": names,
        "mdi": rf.feature_importances_,
        "mdi_std": np.std([t.feature_importances_ for t in rf.estimators_], axis=0),
        "permutation_mean": perm.importances_mean,
        "permutation_std": perm.importances_std,
    }).sort_values("permutation_mean", ascending=False)
    imp.to_csv(C.RESULTS_DIR / "feature_importance.csv", index=False)

    depths = [e.get_depth() for e in rf.estimators_]
    leaves = [e.get_n_leaves() for e in rf.estimators_]
    summary = {
        "split": split.to_dict(),
        "params": {k: (str(v) if isinstance(v, dict) else v) for k, v in params.items()},
        "feature_set": feature_set,
        "threshold_metric": THRESHOLD_METRIC,
        "oof": oof_summary,
        "test": test_metrics,
        "forest": {"mean_depth": float(np.mean(depths)), "max_depth": int(np.max(depths)),
                   "mean_leaves": float(np.mean(leaves)), "n_trees": len(rf.estimators_)},
    }
    (C.RESULTS_DIR / "final_model_results.json").write_text(json.dumps(summary, indent=2, default=float))

    card = {
        "model": "RandomForestClassifier (scikit-learn) inside an imblearn Pipeline",
        "version": __version__,
        "trained_on": str(date.today()),
        "inputs": C.RAW_FEATURES,
        "hyperparameters": summary["params"],
        "decision_threshold": threshold,
        "threshold_rule": f"maximise out-of-fold {THRESHOLD_METRIC} on the training set",
        "training_rows": split.n_train,
        "training_frauds": split.fraud_train,
        "intended_use": "Ranking card transactions for fraud review; not a sole basis for blocking.",
    }
    C.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
    t = test_metrics["at_threshold"]
    log.info("TEST @%.2f  TP=%d FP=%d FN=%d  P=%.4f R=%.4f F1=%.4f MCC=%.4f  PR-AUC=%.4f ROC-AUC=%.4f",
             threshold, t["tp"], t["fp"], t["fn"], t["precision"], t["recall"], t["f1"], t["mcc"],
             t["pr_auc"], t["roc_auc"])
    return summary
