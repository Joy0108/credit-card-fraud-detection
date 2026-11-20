"""Figures for the modelling half of the report, built from saved results.

Nothing here trains a model: every function reads the CSV/JSON/NPZ files the
pipeline stages wrote to ``reports/results`` so figures can be restyled and
rebuilt in seconds.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as C
from . import plotting as P

R = C.RESULTS_DIR
GRAY = "#c3c2b7"


def _read(name: str):
    path = R / name
    if path.suffix == ".csv":
        # hyper-parameter tables store the literal string "None"; keep it as text
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return np.load(path, allow_pickle=False)


def outcomes(pred: pd.DataFrame, rep: dict) -> pd.Series:
    """Label each test row TP/FN/FP/TN from the model's ranking.

    Frauds are ranked by predicted probability: the top ``tp`` are caught and
    the rest are missed. Legitimate rows are ranked the same way: the top
    ``fp`` are false alarms. With a single threshold this is identical to
    thresholding the scores.
    """
    out = pd.Series("TN", index=pred.index)
    fraud = pred[pred[C.TARGET] == 1].sort_values("fraud_probability", ascending=False)
    legit = pred[pred[C.TARGET] == 0].sort_values("fraud_probability", ascending=False)
    out[fraud.index[: rep["tp"]]] = "TP"
    out[fraud.index[rep["tp"]:]] = "FN"
    out[legit.index[: rep["fp"]]] = "FP"
    return out


# --------------------------------------------------------------------------- #
# Methodology diagram
# --------------------------------------------------------------------------- #
def fig_pipeline(split: dict) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    ax.set_xlim(-1.5, 101.5)
    ax.set_ylim(-0.5, 44.5)
    ax.axis("off")

    def box(x, y, w, h, title, body, tone="#f4f8fd", edge="#9ec5f4"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=tone, ec=edge, lw=1))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9, weight="semibold",
                color=P.INK)
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7.6, color=P.INK_2,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                     color=P.MUTED, lw=1.1, shrinkA=2, shrinkB=2))

    box(0, 15, 15, 16, "Raw data", f"{split['n_total']:,} rows\n30 features + Class\nschema checks")
    box(19, 15, 15, 16, "EDA", "stat. tests (KS, MWU)\nPCA checks\ntime / amount")
    box(38, 15, 15, 16, "Stratified split", "80 / 20, seed 42\nfraud rate kept\nin both parts")
    # train branch
    box(57, 27, 19, 16, "Training set", f"{split['n_train_raw']:,} rows\n- {split['train_internal_duplicates']} duplicates\n"
        f"- {split['train_rows_matching_test']} rows seen in test\n= {split['n_train']:,} ({split['fraud_train']} fraud)")
    box(80, 27, 20, 16, "Model selection", "5-fold: imbalance strategies\n3-fold: randomized search\n"
        "5-fold OOF: threshold", tone="#fdf1ec", edge="#f3b79c")
    # test branch
    box(57, 1, 19, 16, "Hold-out test set", f"{split['n_test']:,} rows\n{split['fraud_test']} fraud\nnever used for tuning",
        tone="#f4f4f2", edge=GRAY)
    box(80, 1, 20, 16, "Final evaluation", "refit RF on full train\nscore test once\nCM, F1, MCC, PR-AUC",
        tone="#fdf1ec", edge="#f3b79c")
    arrow(15.6, 23, 18.4, 23)
    arrow(34.6, 23, 37.4, 23)
    arrow(53.6, 26, 56.4, 33)
    arrow(53.6, 20, 56.4, 10)
    arrow(76.6, 35, 79.4, 35)
    arrow(76.6, 9, 79.4, 9)
    arrow(90, 26.4, 90, 17.6)
    ax.text(91, 22, "best params\n+ threshold", fontsize=7.2, color=P.INK_2, va="center")
    P.save(fig, "method_pipeline")


# --------------------------------------------------------------------------- #
# Imbalance strategies
# --------------------------------------------------------------------------- #
def fig_imbalance(chosen: str = "class_weight") -> None:
    res = _read("imbalance_comparison.csv")
    res = res[res["family"] != "Feature ablation"].sort_values("pr_auc")
    metrics = [("pr_auc", "PR-AUC"), ("precision@0.5", "Precision @ 0.5"),
               ("recall@0.5", "Recall @ 0.5"), ("mcc@0.5", "MCC @ 0.5")]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.5), sharey=True)
    y = np.arange(len(res))
    for ax, (col, title) in zip(axes, metrics):
        colors = [P.LEGIT if s == chosen else GRAY for s in res["strategy"]]
        ax.barh(y, res[col], height=0.56, color=colors)
        for yi, v in zip(y, res[col]):
            ax.text(v + 0.015, yi, f"{v:.3f}" if col == "pr_auc" else f"{v:.2f}", va="center", fontsize=7.6,
                    color=P.INK_2)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9.5, pad=6)
    axes[0].set_yticks(y, res["label"], fontsize=8.2)
    fig.tight_layout(w_pad=0.8)
    P.save(fig, "imbalance_comparison")


def fig_imbalance_pr() -> None:
    oof = _read("imbalance_oof.npz")
    res = _read("imbalance_comparison.csv").set_index("strategy")
    y = oof["y"]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    show = ["class_weight", "smote", "none", "undersample"]
    for i, s in enumerate(show):
        r, p = precision_recall_curve(y, oof[s])[1], precision_recall_curve(y, oof[s])[0]
        ax.plot(r, p, color=P.SERIES[i], lw=2, label=f"{res.loc[s, 'label']}  (AP {res.loc[s, 'pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower left", fontsize=8.2)
    P.titled(ax, "Out-of-fold precision-recall curves", "5-fold CV on the training set, 100-tree forests")
    P.save(fig, "imbalance_pr_curves")


# --------------------------------------------------------------------------- #
# Tuning & threshold
# --------------------------------------------------------------------------- #
def fig_tuning() -> None:
    tun = _read("tuning_results.csv")
    cw_order = ["None", "balanced_subsample", "{0:1, 1:5}", "{0:1, 1:25}"]
    names = {"None": "No class weight", "balanced_subsample": "balanced_subsample",
             "{0:1, 1:5}": "fraud weight 5", "{0:1, 1:25}": "fraud weight 25"}
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for i, cw in enumerate(cw_order):
        part = tun[tun["class_weight"].astype(str) == cw]
        if part.empty:
            continue
        ax.scatter(part["mean_test_recall"], part["mean_test_precision"], s=48, marker=markers[i],
                   color=P.SERIES[i], edgecolor=P.SURFACE, lw=1.2, label=names[cw], zorder=3)
    best = tun.iloc[0]
    ax.scatter(best["mean_test_recall"], best["mean_test_precision"], s=190, facecolor="none",
               edgecolor=P.INK, lw=1.3, zorder=4)
    ax.annotate("selected (best MCC)", (best["mean_test_recall"], best["mean_test_precision"]),
                xytext=(12, -14), textcoords="offset points", fontsize=8, color=P.INK)
    ax.set_xlabel("Mean CV recall")
    ax.set_ylabel("Mean CV precision")
    ax.legend(loc="lower left", fontsize=8, title="class_weight", title_fontsize=8)
    P.titled(ax, f"Precision-recall trade-off of {len(tun)} candidates", "3-fold stratified CV, threshold 0.5")

    ax = axes[1]
    t = tun.sort_values("mean_test_mcc", ascending=True)
    yy = np.arange(len(t))
    colors = [P.LEGIT if r == 1 else GRAY for r in t["rank"]]
    ax.barh(yy, t["mean_test_mcc"], xerr=t["std_test_mcc"], height=0.6, color=colors,
            error_kw={"ecolor": P.MUTED, "elinewidth": 0.8, "capsize": 0})
    ax.set_yticks(yy, [f"#{r}" for r in t["rank"]], fontsize=7.4)
    lo = max(0, t["mean_test_mcc"].min() - 0.08)
    ax.set_xlim(lo, min(1.0, (t["mean_test_mcc"] + t["std_test_mcc"]).max() + 0.02))
    ax.set_xlabel("Mean CV MCC (bars: +/- 1 sd across folds)")
    ax.grid(axis="y", visible=False)
    P.titled(ax, "Candidates ranked by MCC", "Selection metric for the search")
    fig.tight_layout(w_pad=2.5)
    P.save(fig, "tuning_tradeoff")


def fig_threshold(threshold: float) -> None:
    cur = _read("threshold_curve_oof.csv")
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    series = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("mcc", "MCC")]
    for i, (col, lab) in enumerate(series):
        ax.plot(cur["threshold"], cur[col], color=P.SERIES[i], lw=2, label=lab)
    ax.axvline(threshold, color=P.INK_2, lw=1)
    ax.axvline(0.5, color=GRAY, lw=1)
    left = threshold < 0.5
    ax.text(threshold + (-0.012 if left else 0.012), 0.05, f"selected t = {threshold:.2f}", fontsize=8.4,
            color=P.INK, ha="right" if left else "left")
    ax.text(0.5 + (0.012 if left else -0.012), 0.05, "default 0.5", fontsize=8, color=P.INK_2,
            ha="left" if left else "right")
    ax.set_xlabel("Decision threshold on predicted fraud probability")
    ax.set_ylabel("Out-of-fold score")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", ncol=4, fontsize=8.4, bbox_to_anchor=(0.5, -0.17))
    P.titled(ax, "Choosing the operating point", "5-fold out-of-fold predictions on the training set")
    P.save(fig, "threshold_selection")


# --------------------------------------------------------------------------- #
# Hold-out results
# --------------------------------------------------------------------------- #
def fig_confusion(rep: dict) -> None:
    cm = np.array([[rep["tn"], rep["fp"]], [rep["fn"], rep["tp"]]])
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.imshow(share, cmap=P.SEQUENTIAL, vmin=0, vmax=1)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            dark = share[i, j] > 0.55
            c = "#ffffff" if dark else P.INK
            c2 = "#ffffff" if dark else P.INK_2
            ax.text(j, i - 0.12, f"{cm[i, j]:,}", ha="center", va="center", fontsize=17, weight="semibold", color=c)
            ax.text(j, i + 0.13, labels[i][j], ha="center", va="center", fontsize=8.6, color=c2)
            pct = f"{share[i, j]:.3%}" if 0 < share[i, j] < 0.01 or 0.99 < share[i, j] < 1 else f"{share[i, j]:.2%}"
            ax.text(j, i + 0.29, f"{pct} of row", ha="center", va="center", fontsize=8, color=c2)
    ax.set_xticks([0, 1], ["Predicted legitimate", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual\nlegitimate", "Actual\nfraud"])
    ax.tick_params(axis="both", labelsize=8.8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    P.save(fig, "test_confusion_matrix")


def _baseline_scores() -> dict:
    z = _read("baseline_test_scores.npz")
    return {k: z[k] for k in z.files}


def fig_curves(rep: dict) -> None:
    z = _baseline_scores()
    y = z["y"]
    show = [("Random_Forest_tuned", "Random Forest (tuned)"),
            ("Hist_Gradient_Boosting", "Hist. Gradient Boosting"),
            ("Logistic_Regression", "Logistic Regression"),
            ("Isolation_Forest_unsupervised", "Isolation Forest")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    for i, (k, lab) in enumerate(show):
        if k not in z:
            continue
        fpr, tpr, _ = roc_curve(y, z[k])
        p, r, _ = precision_recall_curve(y, z[k])
        lw = 2.4 if i == 0 else 1.6
        axes[0].plot(fpr, tpr, color=P.SERIES[i], lw=lw, label=lab)
        axes[1].plot(r, p, color=P.SERIES[i], lw=lw, label=lab)
    axes[0].plot([0, 1], [0, 1], color=GRAY, lw=1)
    axes[0].set_xscale("symlog", linthresh=0.001)
    axes[0].set_xlim(0, 1)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_xlabel("False positive rate (symlog scale)")
    axes[0].set_ylabel("True positive rate (recall)")
    axes[0].legend(loc="lower right", fontsize=8)
    P.titled(axes[0], "ROC curve", "Log-scaled x-axis: only the far-left region matters at 0.17 % fraud")
    ax = axes[1]
    ax.axhline(y.mean(), color=GRAY, lw=1)
    ax.text(0.98, y.mean() + 0.015, "no-skill baseline (0.17 %)", fontsize=7.8, color=P.INK_2, ha="right")
    ax.legend(loc="center", bbox_to_anchor=(0.45, 0.55), fontsize=8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    P.titled(ax, "Precision-recall curve", "Hold-out test set, 98 frauds")
    fig.tight_layout(w_pad=3)
    P.save(fig, "test_roc_pr")


def fig_score_distribution(threshold: float) -> None:
    pred = _read("test_predictions.csv")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    bins = np.linspace(0, 1, 41)
    for cls in (0, 1):
        ax.hist(pred.loc[pred[C.TARGET] == cls, "fraud_probability"], bins=bins, histtype="step", lw=2,
                color=P.CLASS_COLORS[cls], label=P.CLASS_NAMES[cls])
    ax.set_yscale("log")
    ax.axvline(threshold, color=P.INK_2, lw=1)
    ax.text(threshold + 0.01, 2e4, f"threshold {threshold:.2f}", fontsize=8.2, color=P.INK)
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Transactions (log scale)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper center")
    P.titled(ax, "Score separation on the hold-out set", "Almost all legitimate traffic scores near zero")
    P.save(fig, "test_score_distribution")


def fig_model_comparison(table: pd.DataFrame) -> None:
    t = table[~table["model"].str.startswith("Majority")].sort_values("mcc")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9), sharey=True)
    y = np.arange(len(t))
    for ax, (col, title) in zip(axes, [("f1", "F1-score"), ("mcc", "MCC"), ("pr_auc", "PR-AUC")]):
        colors = [P.LEGIT if m.startswith("Random Forest") else GRAY for m in t["model"]]
        ax.barh(y, t[col], height=0.56, color=colors)
        for yi, v in zip(y, t[col]):
            ax.text(v + 0.015, yi, f"{v:.2f}" if col != "mcc" else f"{v:.2f}", va="center", fontsize=7.8,
                    color=P.INK_2)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9.5, pad=6)
    axes[0].set_yticks(y, t["model"], fontsize=8.4)
    fig.tight_layout(w_pad=0.8)
    P.save(fig, "model_comparison")


def fig_importance(k: int = 15) -> None:
    imp = _read("feature_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, (col, err, title, sub) in zip(axes, [
        ("mdi", "mdi_std", "Impurity-based importance (MDI)", "Mean decrease in Gini impurity, training set"),
        ("permutation_mean", "permutation_std", "Permutation importance", "Drop in test PR-AUC when the feature is shuffled"),
    ]):
        t = imp.sort_values(col, ascending=False).head(k).iloc[::-1]
        ax.barh(t["feature"], t[col], height=0.6, color=P.LEGIT)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.4)
        P.titled(ax, title, sub)
    fig.tight_layout(w_pad=3)
    P.save(fig, "feature_importance")


def fig_error_analysis(rep: dict, threshold: float) -> None:
    pred = _read("test_predictions.csv")
    pred["outcome"] = outcomes(pred, rep)
    imp = _read("feature_importance.csv").sort_values("permutation_mean", ascending=False)
    a, b = imp["feature"].iloc[0], imp["feature"].iloc[1]      # the model's two most relied-on inputs
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    fraud = pred[pred[C.TARGET] == 1].sort_values("fraud_probability").reset_index(drop=True)
    colors = [P.SERIES[2] if o == "FN" else P.FRAUD for o in fraud["outcome"]]
    ax.scatter(np.arange(len(fraud)), fraud["fraud_probability"], s=26, color=colors,
               edgecolor=P.SURFACE, lw=0.8, zorder=3)
    ax.set_xlabel(f"The {len(fraud)} test frauds, sorted by predicted probability")
    ax.set_ylabel("Predicted fraud probability")
    ax.set_xticks([])
    ax.set_ylim(0, 1.02)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], color=P.FRAUD, marker="o", lw=0, ms=7, label=f"Caught ({rep['tp']})"),
                       Line2D([], [], color=P.SERIES[2], marker="o", lw=0, ms=7, label=f"Missed ({rep['fn']})")],
              loc="upper left")
    P.titled(ax, "Where the frauds land", "Most missed frauds score near zero, far from the caught group")

    ax = axes[1]
    tn = pred[pred["outcome"] == "TN"]
    legit = tn.sample(min(6000, len(tn)), random_state=C.RANDOM_STATE)
    ax.scatter(legit[a], legit[b], s=8, color=P.LEGIT, alpha=0.30, lw=0, label="Legitimate (sample)")
    tp = pred[pred["outcome"] == "TP"]
    fn = pred[pred["outcome"] == "FN"]
    fp = pred[pred["outcome"] == "FP"]
    ax.scatter(tp[a], tp[b], s=30, color=P.FRAUD, edgecolor=P.SURFACE, lw=1, label="Fraud caught", zorder=3)
    ax.scatter(fn[a], fn[b], s=40, marker="D", color=P.SERIES[2], edgecolor=P.SURFACE, lw=1,
               label="Fraud missed", zorder=4)
    if len(fp):
        ax.scatter(fp[a], fp[b], s=90, marker="X", color=P.INK, edgecolor=P.SURFACE, lw=1,
                   label="False alarm", zorder=5)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    handles = [Line2D([], [], color=P.LEGIT, marker="o", lw=0, ms=5, label="Legitimate (sample)"),
               Line2D([], [], color=P.FRAUD, marker="o", lw=0, ms=6, label="Fraud caught"),
               Line2D([], [], color=P.SERIES[2], marker="D", lw=0, ms=6, label="Fraud missed")]
    if len(fp):
        handles.append(Line2D([], [], color=P.INK, marker="X", lw=0, ms=8, label="False alarm"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True, facecolor=P.SURFACE,
              edgecolor="none", framealpha=0.95)
    P.titled(ax, "Many missed frauds sit inside the legitimate cloud", f"{a} vs {b}, hold-out test set")
    fig.tight_layout(w_pad=3)
    P.save(fig, "error_analysis")


def build_all(rep: dict, threshold: float, split: dict, comparison: pd.DataFrame) -> None:
    fig_pipeline(split)
    fig_imbalance()
    fig_imbalance_pr()
    fig_tuning()
    fig_threshold(threshold)
    fig_confusion(rep)
    fig_curves(rep)
    fig_model_comparison(comparison)
    fig_importance()
    fig_error_analysis(rep, threshold)
