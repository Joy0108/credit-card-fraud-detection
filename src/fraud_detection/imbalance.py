"""Head-to-head comparison of class-imbalance strategies for a Random Forest.

Every strategy is wrapped in an ``imblearn`` Pipeline so that resampling is
applied **only to the training folds** of each cross-validation split. The
validation fold always keeps the real 0.17 % fraud rate - resampling before
splitting would leak synthetic copies of validation frauds into training and
inflate every score.

Run with ``python -m fraud_detection compare``.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from . import config as C
from . import evaluation as E
from .data import load_raw, split_train_test, xy
from .features import FeatureBuilder

log = logging.getLogger(__name__)


def _rf(**kw) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=C.COMPARISON_TREES, n_jobs=-1,
                                  random_state=C.RANDOM_STATE, **kw)


def strategies() -> dict[str, dict]:
    """name -> {label, family, pipeline factory}."""
    rs = C.RANDOM_STATE
    return {
        "none": dict(
            label="No treatment", family="Baseline",
            make=lambda: Pipeline([("features", FeatureBuilder("raw")), ("model", _rf())])),
        "class_weight": dict(
            label="Class weighting", family="Cost-sensitive",
            make=lambda: Pipeline([("features", FeatureBuilder("raw")),
                                   ("model", _rf(class_weight="balanced_subsample"))])),
        "undersample": dict(
            label="Random undersampling 1:1", family="Undersampling",
            make=lambda: Pipeline([("features", FeatureBuilder("raw")),
                                   ("sampler", RandomUnderSampler(sampling_strategy=1.0, random_state=rs)),
                                   ("model", _rf())])),
        "oversample": dict(
            label="Random oversampling 1:10", family="Oversampling",
            make=lambda: Pipeline([("features", FeatureBuilder("raw")),
                                   ("sampler", RandomOverSampler(sampling_strategy=C.SMOTE_RATIO, random_state=rs)),
                                   ("model", _rf())])),
        "smote": dict(
            label="SMOTE 1:10", family="Synthetic oversampling",
            make=lambda: Pipeline([("features", FeatureBuilder("raw")),
                                   ("sampler", SMOTE(sampling_strategy=C.SMOTE_RATIO, k_neighbors=5, random_state=rs)),
                                   ("model", _rf())])),
        "smote_tomek": dict(
            label="SMOTE + Tomek links", family="Hybrid",
            make=lambda: Pipeline([("features", FeatureBuilder("raw")),
                                   ("sampler", SMOTETomek(smote=SMOTE(sampling_strategy=C.SMOTE_RATIO, random_state=rs),
                                                          tomek=TomekLinks(n_jobs=-1), random_state=rs)),
                                   ("model", _rf())])),
        "balanced_rf": dict(
            label="Balanced Random Forest", family="Ensemble undersampling",
            make=lambda: Pipeline([("features", FeatureBuilder("raw")),
                                   ("model", BalancedRandomForestClassifier(
                                       n_estimators=C.COMPARISON_TREES, sampling_strategy="all",
                                       replacement=True, bootstrap=False, n_jobs=-1, random_state=rs))])),
        # feature-set ablation (same RF as "class_weight", different inputs)
        "class_weight_engineered": dict(
            label="Class weighting + hour/log-amount features", family="Feature ablation",
            make=lambda: Pipeline([("features", FeatureBuilder("engineered")),
                                   ("model", _rf(class_weight="balanced_subsample"))])),
    }


def cross_validate_oof(make_pipeline, X: pd.DataFrame, y: np.ndarray, n_splits: int = C.CV_FOLDS):
    """Out-of-fold probabilities + per-fold metrics for one pipeline."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    oof = np.zeros(len(y))
    folds, fit_seconds, train_sizes = [], [], []
    for k, (tr, va) in enumerate(skf.split(X, y)):
        pipe = make_pipeline()
        t0 = time.perf_counter()
        pipe.fit(X.iloc[tr], y[tr])
        fit_seconds.append(time.perf_counter() - t0)
        if "sampler" in pipe.named_steps and k == 0:   # resampled size, measured once
            _, yr = pipe.named_steps["sampler"].fit_resample(
                pipe.named_steps["features"].transform(X.iloc[tr]), y[tr])
            train_sizes.append((len(yr), int(yr.sum())))
        elif "sampler" not in pipe.named_steps:
            train_sizes.append((len(tr), int(y[tr].sum())))
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        fm = E.full_report(y[va], oof[va], 0.5)
        fm["fold"] = k
        folds.append(fm)
        log.info("  fold %d  P=%.3f R=%.3f F1=%.3f MCC=%.3f AP=%.3f (%.0fs)", k, fm["precision"],
                 fm["recall"], fm["f1"], fm["mcc"], fm["pr_auc"], fit_seconds[-1])
    return oof, pd.DataFrame(folds), float(np.mean(fit_seconds)), train_sizes


def run() -> pd.DataFrame:
    df = load_raw()
    train, _test, split = split_train_test(df)
    (C.RESULTS_DIR / "split_summary.json").write_text(json.dumps(split.to_dict(), indent=2))
    X, y = xy(train)

    rows, oof_store, fold_store = [], {"y": y}, []
    for name, spec in strategies().items():
        log.info("Strategy: %s", spec["label"])
        oof, folds, fit_s, sizes = cross_validate_oof(spec["make"], X, y)
        oof_store[name] = oof
        folds["strategy"] = name
        fold_store.append(folds)
        thr = E.select_threshold(y, oof, metric="f1")
        at05 = E.at_threshold(y, oof, 0.5)
        best = E.at_threshold(y, oof, thr)
        rank = E.ranking(y, oof)
        rows.append({
            "strategy": name, "label": spec["label"], "family": spec["family"],
            "train_rows_after_resampling": int(np.mean([s[0] for s in sizes])),
            "train_frauds_after_resampling": int(np.mean([s[1] for s in sizes])),
            "fit_seconds": fit_s,
            "pr_auc": rank["pr_auc"], "roc_auc": rank["roc_auc"], "brier": rank["brier"],
            "pr_auc_fold_std": folds["pr_auc"].std(),
            **{f"{k}@0.5": at05[k] for k in ("precision", "recall", "f1", "mcc", "fp", "fn")},
            "mcc@0.5_fold_std": folds["mcc"].std(),
            "f1@0.5_fold_std": folds["f1"].std(),
            "best_threshold": thr,
            **{f"{k}@best": best[k] for k in ("precision", "recall", "f1", "mcc", "fp", "fn")},
        })
        log.info("  OOF: AP=%.4f  F1@0.5=%.4f  MCC@0.5=%.4f  best thr=%.2f F1=%.4f",
                 rank["pr_auc"], at05["f1"], at05["mcc"], thr, best["f1"])

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / "imbalance_comparison.csv", index=False)
    pd.concat(fold_store).to_csv(C.RESULTS_DIR / "imbalance_folds.csv", index=False)
    np.savez_compressed(C.RESULTS_DIR / "imbalance_oof.npz", **oof_store)
    return res
