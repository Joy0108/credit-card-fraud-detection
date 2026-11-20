"""Metrics for rare-event classification.

Accuracy is useless here (always predicting "legitimate" scores 99.83 %), so
everything is built on the confusion matrix and on ranking quality:

* precision  - of the transactions we block, how many are really fraud
* recall     - of all frauds, how many we catch
* F1         - harmonic mean of the two
* MCC        - Matthews correlation coefficient, uses all four cells and stays
               informative under extreme imbalance
* PR-AUC     - average precision, threshold-free ranking quality on the
               minority class (ROC-AUC is reported but is optimistic here)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)


# --------------------------------------------------------------------------- #
# Confusion-matrix metrics
# --------------------------------------------------------------------------- #
def counts(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def mcc_from_counts(tp, fp, fn, tn) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom else 0.0


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "mcc": mcc_from_counts(tp, fp, fn, tn),
        "accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
        "balanced_accuracy": (recall + specificity) / 2,
        "specificity": specificity,
        "npv": _safe_div(tn, tn + fn),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "alerts": tp + fp,
    }


def at_threshold(y_true, prob, threshold: float) -> dict:
    out = metrics_from_counts(**counts(y_true, np.asarray(prob) >= threshold))
    out["threshold"] = float(threshold)
    return out


def ranking(y_true, prob) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def full_report(y_true, prob, threshold: float) -> dict:
    return {**at_threshold(y_true, prob, threshold), **ranking(y_true, prob)}


# --------------------------------------------------------------------------- #
# Threshold analysis
# --------------------------------------------------------------------------- #
def threshold_curve(y_true, prob, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Precision / recall / F1 / MCC as the decision threshold moves."""
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2) if grid is None else grid
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    rows = [at_threshold(y_true, prob, t) for t in grid]
    return pd.DataFrame(rows)


def select_threshold(y_true, prob, metric: str = "f1", min_precision: float | None = None) -> float:
    """Choose the threshold that maximises ``metric`` on out-of-fold scores.

    Ties are broken towards the *higher* threshold (fewer alerts). If
    ``min_precision`` is given, only thresholds meeting it are eligible.
    """
    curve = threshold_curve(y_true, prob, grid=np.round(np.arange(0.05, 0.96, 0.01), 2))
    if min_precision is not None:
        eligible = curve[curve["precision"] >= min_precision]
        curve = eligible if len(eligible) else curve
    best = curve[metric].max()
    return float(curve.loc[np.isclose(curve[metric], best), "threshold"].max())


def pr_curve(y_true, prob) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    return recall, precision


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_from_counts(tp: int, fp: int, fn: int, tn: int, n_boot: int = 2000,
                          seed: int = 42) -> dict:
    """95 % percentile intervals by resampling the test set's confusion cells.

    Each bootstrap replicate draws ``N`` transactions with replacement, which is
    a multinomial draw over the four cells - equivalent to resampling rows.
    """
    rng = np.random.default_rng(seed)
    n = tp + fp + fn + tn
    draws = rng.multinomial(n, np.array([tp, fp, fn, tn]) / n, size=n_boot)
    stats = {k: [] for k in ("precision", "recall", "f1", "mcc")}
    for d in draws:
        m = metrics_from_counts(*map(int, d))
        for k in stats:
            stats[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in stats.items()}


def bootstrap_ranking(y_true, prob, n_boot: int = 2000, seed: int = 42) -> dict:
    """95 % intervals for PR-AUC and ROC-AUC via stratified row bootstrap."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    pr, roc = [], []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        pr.append(average_precision_score(y_true[idx], prob[idx]))
        roc.append(roc_auc_score(y_true[idx], prob[idx]))
    return {"pr_auc": (float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))),
            "roc_auc": (float(np.percentile(roc, 2.5)), float(np.percentile(roc, 97.5)))}
