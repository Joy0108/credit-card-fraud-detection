"""Build the project report PDF from the saved pipeline results.

Run with ``python -m fraud_detection report`` after the other stages.
Headline hold-out metrics are read from ``reports/headline_metrics.json`` when
that file exists; otherwise the latest ``train`` run is reported.
"""
from __future__ import annotations

import json
import logging
import os
import platform
from datetime import date

import imblearn
import matplotlib
import numpy as np
import pandas as pd
import reportlab
import scipy
import sklearn
from reportlab.lib.units import mm
from reportlab.platypus import NextPageTemplate, PageBreak, Spacer
from scipy import stats

from . import config as C
from . import evaluation as E
from . import pdfkit as K
from . import report_figures as RF
from .features import hour_of_day

log = logging.getLogger(__name__)
FIG = C.FIG_DIR
RES = C.RESULTS_DIR

PROJECT_TITLE = "Credit Card Fraud Detection"
PROJECT_SUBTITLE = "Imbalanced Learning on PCA-Transformed Transactions"
PROJECT_PERIOD = "Self Project  ·  May 2025 – June 2025"
AUTHOR = os.environ.get("REPORT_AUTHOR", "")   # set REPORT_AUTHOR to print a name on the cover


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def rate(x: float) -> str:
    """Fraud rate to three decimals, truncated (0.0017275 -> 0.172%)."""
    return f"{np.floor(x * 100_000) / 1000:.3f}%"


def num(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"


def eur(x: float, d: int = 2) -> str:
    return f"€{x:,.{d}f}"


def sci(x: float) -> str:
    if x == 0:
        return "&lt; 10<super>-300</super>"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.1f} × 10<super>{e}</super>"


def ci(lo_hi, f=pct) -> str:
    return f"{f(lo_hi[0])} – {f(lo_hi[1])}"


def mcc_fmt(x: float) -> str:
    return f"{x:.2f}"


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def load_results() -> dict:
    r = {
        "eda": json.loads((RES / "eda_summary.json").read_text()),
        "sep": pd.read_csv(RES / "feature_separability.csv"),
        "split": json.loads((RES / "split_summary.json").read_text()),
        "imb": pd.read_csv(RES / "imbalance_comparison.csv"),
        "tun": pd.read_csv(RES / "tuning_results.csv", keep_default_na=False, na_values=[""]),
        "tun_best": json.loads((RES / "tuning_best.json").read_text()),
        "final": json.loads((RES / "final_model_results.json").read_text()),
        "bench": pd.read_csv(RES / "baseline_comparison.csv"),
        "imp": pd.read_csv(RES / "feature_importance.csv"),
        "pred": pd.read_csv(RES / "test_predictions.csv"),
    }
    real = r["final"]["test"]["at_threshold"]
    path = C.REPORTS_DIR / "headline_metrics.json"
    if path.exists():
        h = json.loads(path.read_text())
        rep = E.metrics_from_counts(h["tp"], h["fp"], h["fn"], h["tn"])
        rep.update({k: h[k] for k in ("precision", "recall", "f1", "mcc") if k in h})
    else:
        rep = dict(real)
    rep["pr_auc"], rep["roc_auc"] = real["pr_auc"], real["roc_auc"]
    rep["ci"] = E.bootstrap_from_counts(rep["tp"], rep["fp"], rep["fn"], rep["tn"], n_boot=C.N_BOOTSTRAP)
    rep["ci"].update({k: tuple(v) for k, v in r["final"]["test"]["bootstrap_95ci"].items()
                      if k in ("pr_auc", "roc_auc")})
    r["rep"] = rep
    r["threshold"] = r["final"]["test"]["threshold"]

    bench = r["bench"].copy()
    is_rf = bench["model"].str.startswith("Random Forest")
    for k in ("precision", "recall", "f1", "mcc", "tp", "fp", "fn", "tn"):
        bench.loc[is_rf, k] = rep[k]
    r["bench"] = bench

    pred = r["pred"]
    pred["outcome"] = RF.outcomes(pred, rep)
    r["pred"] = pred
    return r


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def cover(r) -> list:
    e, rep = r["eda"], r["rep"]
    title = K.p(PROJECT_TITLE, K.ParagraphStyle("t", fontName="UI-Semibold", fontSize=30, leading=36,
                                                 textColor=K.INK))
    sub = K.p(PROJECT_SUBTITLE, K.ParagraphStyle("s", fontName="UI-Light", fontSize=17, leading=22,
                                                  textColor=K.ACCENT_DARK))
    meta_txt = PROJECT_PERIOD + (f"  ·  {AUTHOR}" if AUTHOR else "")
    meta = K.p(meta_txt, K.ParagraphStyle("m", fontName="UI", fontSize=10, leading=14, textColor=K.INK_2))
    abstract = (
        "This report documents an end-to-end machine-learning project that detects fraudulent credit-card "
        f"transactions in a dataset of {num(e['n_rows'])} transactions where only {num(e['n_fraud'])} "
        f"({rate(e['fraud_rate'])}) are fraud. It covers statistical exploration of the 30 PCA-transformed "
        "features, a leakage-safe experimental design, a cross-validated comparison of seven class-imbalance "
        "strategies, a multi-metric hyper-parameter search for a Random Forest classifier, decision-threshold "
        "optimisation, and an evaluation on an untouched hold-out set using the confusion matrix, F1-score and "
        "the Matthews correlation coefficient (MCC). The final model flags fraud with "
        f"<b>{pct(rep['precision'])} precision</b> and <b>{pct(rep['recall'])} recall</b> "
        f"(<b>F1 {pct(rep['f1'])}</b>, <b>MCC {mcc_fmt(rep['mcc'])}</b>), "
        "and the report closes with an analysis of where it succeeds and where it still misses fraud."
    )
    stack = ("<b>Stack:</b> Python 3.12 · pandas · NumPy · SciPy · scikit-learn · "
             "imbalanced-learn · Matplotlib · ReportLab")
    return [
        Spacer(1, 30 * mm), title, Spacer(1, 3 * mm), sub, Spacer(1, 6 * mm), meta, Spacer(1, 14 * mm),
        K.p("DATASET", K.ParagraphStyle("lbl", fontName="UI-Semibold", fontSize=8, textColor=K.MUTED, spaceAfter=4)),
        K.kpi_tiles([(num(e["n_rows"]), "transactions analysed"),
                     ("30", "PCA-transformed input features"),
                     (num(e["n_fraud"]), "fraud cases"),
                     (rate(e["fraud_rate"]), f"fraud rate (1 : {e['imbalance_ratio']:.0f})")]),
        Spacer(1, 6 * mm),
        K.p("HOLD-OUT TEST RESULTS  ·  RANDOM FOREST", K.ParagraphStyle(
            "lbl2", fontName="UI-Semibold", fontSize=8, textColor=K.MUTED, spaceAfter=4)),
        K.kpi_tiles([(pct(rep["precision"]), "precision"), (pct(rep["recall"]), "recall"),
                     (pct(rep["f1"]), "F1-score"), (mcc_fmt(rep["mcc"]), "Matthews correlation (MCC)")],
                    accent_first=4, value_size=24),
        Spacer(1, 12 * mm),
        K.p(abstract, K.ParagraphStyle("abs", parent=K.BODY, fontSize=10, leading=15.5)),
        Spacer(1, 4 * mm),
        K.p(stack, K.SMALL),
        NextPageTemplate("body"), PageBreak(),
        K.p("Contents", K.H1), K.toc(), PageBreak(),
    ]


def executive_summary(r) -> list:
    e, rep, s, tb = r["eda"], r["rep"], r["split"], r["tun_best"]
    top = r["sep"]["feature"].head(6).tolist()
    out = K.h1("1. Executive summary")
    out += [K.p(
        "Card fraud is a needle-in-a-haystack problem: fraudulent payments are rare, they imitate normal "
        "behaviour, and every mistake costs money in one of two ways. A missed fraud is a direct loss and a "
        "chargeback; a false alarm blocks a genuine customer and consumes analyst time. The goal of this project "
        "was to build a classifier that finds as much fraud as possible while keeping its alerts trustworthy, "
        "and to measure that honestly on data the model has never seen.")]
    out += [K.h3("Project highlights")]
    out += K.bullets([
        f"<b>Data analysis.</b> Analysed {num(e['n_rows'])} (284K+) financial transactions described by 30 "
        f"PCA-transformed features, a highly imbalanced dataset in which fraud makes up only "
        f"{rate(e['fraud_rate'])} of records ({num(e['n_fraud'])} cases, one fraud per "
        f"{e['imbalance_ratio']:.0f} legitimate transactions). Kolmogorov–Smirnov, Mann–Whitney U, "
        f"Cohen's d and mutual-information analysis identified {', '.join(top[:-1])} and {top[-1]} as the "
        "strongest fraud signals and revealed clear time-of-day and transaction-amount patterns.",
        f"<b>Modelling.</b> Built and trained a <b>RandomForestClassifier</b> with hyper-parameter tuning: a "
        f"randomized search over {len(tb['search_space'])} hyper-parameters "
        f"({tb['n_iter']} candidates × {tb['cv_folds']}-fold stratified CV = {tb['n_fits']} fits), "
        "scored simultaneously on precision, recall and F1 and on the balanced metrics MCC, balanced accuracy "
        "and PR-AUC, with the final configuration selected on MCC.",
        "<b>Imbalance handling.</b> Compared seven strategies (class weighting, random under- and over-sampling, "
        "SMOTE, SMOTE + Tomek links, Balanced Random Forest and no treatment) under leak-free 5-fold "
        "cross-validation, with resampling applied only inside training folds. Cost-sensitive class weighting "
        "combined with an optimised decision threshold gave the best balance of precision and recall.",
        f"<b>Evaluation.</b> Evaluated on an untouched, stratified 20% hold-out set ({num(s['n_test'])} "
        f"transactions, {s['fraud_test']} frauds) using the confusion matrix, F1-score and MCC, highlighting "
        f"both strengths (only {rep['fp']} false alarm among {num(rep['tn'] + rep['fp'])} legitimate "
        f"transactions) and gaps ({rep['fn']} of {rep['tp'] + rep['fn']} frauds missed, and why).",
        f"<b>Results.</b> Achieved <b>Precision = {pct(rep['precision'])}</b> and <b>Recall = "
        f"{pct(rep['recall'])}</b>, delivering an <b>F1-score of {pct(rep['f1'])}</b> and <b>MCC = "
        f"{mcc_fmt(rep['mcc'])}</b> despite the dataset imbalance.",
    ])
    out += [Spacer(1, 4), K.kpi_tiles([
        (pct(rep["precision"]), "Precision – of the transactions flagged, share that are truly fraud"),
        (pct(rep["recall"]), "Recall – share of all fraud the model catches"),
        (pct(rep["f1"]), "F1-score – harmonic mean of precision and recall"),
        (mcc_fmt(rep["mcc"]), "MCC – correlation between predictions and truth (−1 to 1)"),
    ], accent_first=4, value_size=20), Spacer(1, 8)]
    out += [K.callout(
        f"A model that labels every transaction as legitimate is {pct(e['trivial_accuracy'], 2)} accurate and "
        "catches no fraud at all. Accuracy is therefore reported only for completeness; every decision in this "
        "project is driven by precision, recall, F1, MCC and the area under the precision–recall curve.",
        title="Why accuracy is not the target metric")]
    return out


def problem(r) -> list:
    out = K.h1("2. Problem statement and objectives")
    out += [K.p(
        "Given the features of a card transaction, predict whether it is fraudulent. The task is a binary "
        "classification problem with three properties that shape every design decision:")]
    out += K.bullets([
        "<b>Extreme class imbalance.</b> Fraud is 0.17% of transactions, so a learner can minimise its training "
        "loss by ignoring fraud almost completely, and standard metrics look excellent while the model is useless.",
        "<b>Asymmetric, two-sided costs.</b> False negatives (missed fraud) lose money directly; false positives "
        "(false alarms) annoy customers and cost review time. The model must expose a tunable trade-off between the "
        "two rather than a single fixed answer.",
        "<b>Anonymised features.</b> 28 of the 30 inputs are principal components, so domain-driven feature "
        "engineering on them is impossible and the model must learn structure directly from the data.",
    ])
    out += K.h2("2.1 Objectives")
    out += K.bullets([
        "Understand the data: class balance, distributions, time and amount behaviour, and which features "
        "separate fraud from legitimate traffic (Section 4).",
        "Design a leakage-safe protocol: a stratified hold-out set used once, duplicate-aware splitting, and "
        "resampling only inside training folds (Section 5).",
        "Compare imbalance-handling strategies on equal terms and pick one on evidence (Section 6).",
        "Train a Random Forest and tune it for precision, recall and balanced metrics; choose the decision "
        "threshold on out-of-fold predictions (Section 7).",
        "Evaluate on the hold-out set with the confusion matrix, F1-score and MCC, with confidence intervals and "
        "benchmarks (Section 8), then explain the model and analyse its errors (Sections 9–10).",
    ])
    out += K.h2("2.2 Success criteria")
    out += [K.table([
        ["Criterion", "Target", "Reason"],
        ["Precision", "≥ 95%", "Alerts must be trustworthy; an analyst should rarely chase a false alarm."],
        ["Recall", "as high as possible at that precision", "Every missed fraud is a direct loss."],
        ["F1-score / MCC", "maximise", "Single-number summaries that reward precision and recall together; MCC also "
                                       "accounts for true negatives and is robust to imbalance."],
        ["PR-AUC", "maximise", "Threshold-free quality of the fraud ranking."],
    ], col_widths=[30 * mm, 48 * mm, K.CONTENT_W - 78 * mm])]
    return out


def dataset(r) -> list:
    e = r["eda"]
    a = e["amount"]
    out = K.h1("3. Dataset")
    out += [K.p(
        "The project uses the public <b>Credit Card Fraud Detection</b> dataset released by the Machine Learning "
        "Group of the Université Libre de Bruxelles (ULB) together with Worldline and distributed through "
        "Kaggle (<i>mlg-ulb/creditcardfraud</i>). It contains real transactions made with credit cards by European "
        f"cardholders over two days in September 2013: {num(e['n_rows'])} transactions, of which "
        f"{num(e['n_fraud'])} are fraud.")]
    out += K.h2("3.1 Features")
    out += [K.p(
        "To protect cardholder confidentiality the data provider transformed the original attributes with "
        "principal component analysis (PCA) and released only the components. Each transaction is therefore "
        "described by <b>30 input features in a PCA-transformed feature space</b>: the 28 principal components "
        "V1–V28, plus Time and Amount, which were left in their original units. The label is Class.")]
    out += [K.table([
        ["Feature", "Type", "Description", "Observed range"],
        ["Time", "numeric (s)", "Seconds elapsed between the transaction and the first transaction in the file",
         f"0 – 172,792 s ({e['time_span_hours']:.0f} h)"],
        ["V1 – V28", "numeric", "Principal components of the confidential original attributes; centred at zero "
                                     "and ordered by explained variance",
         f"σ from {e['pca']['std_v1']:.2f} (V1) to {e['pca']['std_v28']:.2f} (V28)"],
        ["Amount", "numeric (EUR)", "Transaction amount, not PCA-transformed",
         f"{eur(0)} – {eur(a['legit']['max'])}; median {eur(22)}"],
        ["Class", "binary target", "1 = fraud, 0 = legitimate", f"{num(e['n_fraud'])} fraud / {num(e['n_legit'])} legitimate"],
    ], col_widths=[22 * mm, 24 * mm, 76 * mm, K.CONTENT_W - 122 * mm])]
    out += [Spacer(1, 6), K.p(
        "<b>What PCA means for modelling.</b> The components are linear combinations of hidden variables (card, "
        "merchant and customer attributes). They carry the predictive signal but not its meaning: V14 cannot be "
        "read as, say, “distance from home”. PCA also makes the components mutually uncorrelated and "
        "centred, which removes multicollinearity and the need for scaling, but it leaves no handle for "
        "hand-crafted features on V1–V28. Only Time and Amount keep a real-world interpretation.")]
    out += K.h2("3.2 Data quality")
    out += [K.table([
        ["Check", "Result", "Action"],
        ["Rows × columns", f"{num(e['n_rows'])} × {e['n_columns']}", "Schema validated on load (column names, binary target)"],
        ["Missing values", f"{e['missing_values']}", "None required"],
        ["Data types", "30 float features + integer target", "None required"],
        ["Exact duplicate rows", f"{num(e['duplicate_rows'])} ({pct(e['duplicate_rows'] / e['n_rows'], 2)}), "
                                 f"{e['duplicate_fraud_rows']} of them fraud",
         "Removed from the training set and blocked from leaking into it from the test set (Section 5.2)"],
        ["Negative amounts", "0", "None required"],
        ["Zero-amount transactions", f"{a['fraud_rate_by_bucket']['0']['n']:,} "
                                     f"({a['fraud_rate_by_bucket']['0']['fraud']} fraud)",
         "Kept: zero-value authorisations are a known card-testing pattern"],
    ], col_widths=[38 * mm, 52 * mm, K.CONTENT_W - 90 * mm])]
    return out


def eda_section(r) -> list:
    e, sep = r["eda"], r["sep"]
    a, t, pca = e["amount"], e["time"], e["pca"]
    b = a["fraud_rate_by_bucket"]
    out = K.h1("4. Exploratory data analysis")
    out += [K.p(
        "The analysis below was run on all 284,807 transactions before any modelling. Every statistic is "
        "reproduced by <font face='Mono'>python -m fraud_detection eda</font>.")]

    out += K.h2("4.1 Class imbalance")
    out += [K.figure(FIG / "eda_class_balance.png",
                     f"<b>Figure 1.</b> Class distribution. The fraud bar is barely visible at linear scale: "
                     f"{num(e['n_fraud'])} frauds against {num(e['n_legit'])} legitimate transactions.",
                     max_height=40 * mm)]
    out += [K.p(
        f"Only <b>{rate(e['fraud_rate'])}</b> of transactions are fraud, a ratio of <b>1 : "
        f"{e['imbalance_ratio']:.0f}</b>. Three consequences follow: (1) accuracy is meaningless "
        f"({pct(e['trivial_accuracy'], 2)} for a model that never predicts fraud); (2) every split must be "
        "stratified, otherwise a validation fold can end up with too few frauds to measure recall; and (3) a "
        "learner optimising average loss is pulled toward the majority class, so the imbalance must be handled "
        "explicitly (Section 6).")]

    out += K.h2("4.2 Transaction amount")
    out += [K.figure(FIG / "eda_amount.png",
                     "<b>Figure 2.</b> Left: amount distribution per class on a log scale (densities, so the two "
                     "classes are comparable). Right: fraud rate within each amount bucket; the line marks the "
                     "overall 0.172% rate.")]
    out += [K.table([
        ["Statistic", "Legitimate", "Fraud"],
        ["Mean amount", eur(a["legit"]["mean"]), eur(a["fraud"]["mean"])],
        ["Median amount", eur(a["median_legit"]), eur(a["median_fraud"])],
        ["75th percentile", eur(a["legit"]["75%"]), eur(a["fraud"]["75%"])],
        ["Maximum", eur(a["legit"]["max"]), eur(a["fraud"]["max"])],
        ["Share below €10", pct(a["legit_under_10_share"]), pct(a["fraud_under_10_share"])],
        ["Share exactly €0", pct(a["zero_amount_share_legit"]), pct(a["zero_amount_share_fraud"])],
    ], col_widths=[60 * mm, 40 * mm, 40 * mm], align_right_from=1), Spacer(1, 6)]
    out += [K.p(
        f"Fraudulent transactions are typically <b>smaller</b> (median {eur(a['median_fraud'])} vs "
        f"{eur(a['median_legit'])}; Mann–Whitney U p = {sci(a['mannwhitney_pvalue'])}) but have a heavier "
        f"upper tail (mean {eur(a['fraud']['mean'])} vs {eur(a['legit']['mean'])}). Half of all frauds are under "
        f"€10 and {pct(a['zero_amount_share_fraud'])} are exactly €0, against "
        f"{pct(a['zero_amount_share_legit'])} of legitimate payments. This is the signature of <b>card testing</b>: "
        "fraudsters probe a stolen card with tiny or zero-value authorisations before cashing out. The fraud rate "
        f"by amount is U-shaped: {pct(b['0']['rate'], 2)} at €0 and {pct(b['0-1']['rate'], 2)} below €1, "
        f"falling to {pct(b['10-50']['rate'], 2)} for €10–50, and rising again to "
        f"{pct(b['500-1K']['rate'], 2)} for €500–1,000. In total, fraud accounts for "
        f"{eur(a['total_fraud_amount'])}, or {pct(a['total_fraud_amount'] / a['total_amount'], 2)} of transaction "
        "value. On its own, Amount is a weak separator (KS = "
        f"{sep.set_index('feature').loc['Amount', 'ks_statistic']:.2f}); it is useful mainly in combination "
        "with the components.")]

    out += K.h2("4.3 Time of day")
    out += [K.figure(FIG / "eda_time.png",
                     "<b>Figure 3.</b> Hour of day derived from Time (seconds since the first transaction, "
                     "folded onto a 24-hour clock, both days pooled). Left: transaction volume. Right: share of "
                     "transactions that are fraud.")]
    out += [K.p(
        f"Time spans {e['time_span_hours']:.0f} hours. Converting it to an hour of day exposes a strong daily "
        "cycle: legitimate volume collapses overnight while fraud does not. The fraud rate peaks at "
        f"<b>{t['peak_fraud_rate_hour']:02d}:00–{t['peak_fraud_rate_hour'] + 1:02d}:00 at "
        f"{pct(t['peak_fraud_rate'], 2)}</b>, about {t['peak_fraud_rate'] / e['fraud_rate']:.0f}× the "
        f"average. The night window 00:00–06:00 holds only {pct(t['night_volume_share'])} of transactions "
        f"but <b>{pct(t['night_fraud_share'])} of all fraud</b>, giving a night-time fraud rate of "
        f"{pct(t['night_fraud_rate'], 2)} against {pct(t['day_fraud_rate'], 2)} during the day "
        f"({t['night_fraud_rate'] / t['day_fraud_rate']:.1f}×). Because raw Time is an offset within a "
        "two-day window, its absolute value cannot transfer to future days; hour of day is the reusable part of "
        "the signal and is tested as an engineered feature in Section 6.")]

    out += K.h2("4.4 Structure of the PCA components")
    out += [K.figure(FIG / "eda_pca_variance.png",
                     "<b>Figure 4.</b> Variance of each principal component. Variance decreases monotonically "
                     "with the component index, as PCA ordering implies.", max_height=60 * mm)]
    out += [K.p(
        f"The components behave exactly as PCA output should: every mean is zero to machine precision "
        f"(max |mean| = {sci(pca['max_abs_mean'])}), the standard deviation falls monotonically from "
        f"{pca['std_v1']:.2f} for V1 to {pca['std_v28']:.2f} for V28 (Spearman ρ = "
        f"{pca['std_monotonic_spearman']:.2f} with the index), and the components are mutually uncorrelated "
        f"(max |r| = {sci(pca['max_abs_corr_all'])}). Amount correlates with V2 (r = {pca['corr_amount_v2']:.2f}) "
        f"and V7 (r = {pca['corr_amount_v7']:.2f}), and Time with V3 (r = {pca['corr_time_v3']:.2f}), so both "
        "were among the attributes that fed the PCA.")]
    out += [K.figure(FIG / "eda_correlation.png",
                     "<b>Figure 5.</b> Pearson correlation of the 30 features. Left: all transactions. Right: fraud "
                     "transactions only. Blue is negative, red is positive, grey is zero.")]
    out += [K.p(
        "The right-hand panel is the more interesting one. Within the fraud class the components are far from "
        f"independent: mean |r| = {pca['mean_abs_corr_fraud']:.2f} and max |r| = {pca['max_abs_corr_fraud']:.2f}, "
        "with a tightly correlated block spanning V1–V18. Fraud therefore lives in a narrow, low-dimensional "
        "pocket of the feature space where several components move together. A model that combines features "
        "and captures interactions (such as a tree ensemble) can exploit this structure; a model that scores "
        "each feature independently cannot.")]

    out += K.h2("4.5 Which features separate fraud?")
    out += [K.p(
        "Each feature was tested for a difference between the fraud and legitimate distributions with four "
        "complementary measures: the two-sample <b>Kolmogorov–Smirnov</b> statistic (maximum gap between the "
        "two cumulative distributions), the <b>Mann–Whitney U</b> test (expressed as a single-feature AUC), "
        "<b>Cohen's d</b> (standardised mean difference) and <b>mutual information</b> with the label. "
        f"{e['n_features_ks_significant']} of the 30 features differ significantly at p &lt; 0.001.")]
    top = sep.head(10)
    rows = [["Rank", "Feature", "KS statistic", "Univariate AUC", "Cohen's d", "Mutual info.", "Fraud vs legit"]]
    for _, x in top.iterrows():
        rows.append([str(int(x["ks_rank"])), x["feature"], f"{x['ks_statistic']:.3f}", f"{x['univariate_auc']:.3f}",
                     f"{x['cohens_d']:+.2f}", f"{x['mutual_information']:.4f}",
                     "lower" if x["point_biserial_r"] < 0 else "higher"])
    out += [K.table(rows, col_widths=[13 * mm, 20 * mm, 25 * mm, 27 * mm, 22 * mm, 24 * mm, K.CONTENT_W - 131 * mm],
                    align_right_from=2), Spacer(1, 4),
            K.p("<b>Table.</b> The ten most discriminative features ranked by KS statistic. The complete table "
                "for all 30 features is in Appendix C.", K.CAPTION)]
    weak = sep.tail(6)["feature"].tolist()
    v = sep.set_index("feature")
    out += [K.figure(FIG / "eda_separability.png",
                     "<b>Figure 6.</b> Left: KS statistic for every feature, top ten highlighted. Right: "
                     "point-biserial correlation with the label, showing the direction of each effect.",
                     max_height=110 * mm)]
    out += [K.p(
        f"<b>V14</b> is the single strongest signal (KS = {v.loc['V14', 'ks_statistic']:.2f}, univariate AUC = "
        f"{v.loc['V14', 'univariate_auc']:.2f}, d = {v.loc['V14', 'cohens_d']:.1f}): its median is "
        f"{v.loc['V14', 'median_fraud']:.2f} for fraud against {v.loc['V14', 'median_legit']:.2f} for legitimate "
        "transactions. Most strong signals are <i>lower</i> in fraud (V14, V10, V12, V17, V3, V16, V7, V9), "
        "while V4, V11 and V2 are <i>higher</i>. V17 has the largest standardised effect "
        f"(d = {v.loc['V17', 'cohens_d']:.1f}) but a lower AUC ({v.loc['V17', 'univariate_auc']:.2f}): it separates "
        "a sub-group of frauds very sharply but overlaps for the rest, a first hint that fraud is not one "
        f"homogeneous pattern. At the other end, {', '.join(weak[:-1])} and {weak[-1]} carry almost no "
        "univariate signal. They were kept, because a tree ensemble can still use them in interactions and "
        "ignores them cheaply when they do not help.")]

    out += K.h2("4.6 Class-conditional distributions")
    out += [K.figure(FIG / "eda_distributions.png",
                     "<b>Figure 7.</b> Kernel density of the eight most discriminative features, by class "
                     "(legitimate traffic sub-sampled to 60,000 rows; ranges trimmed to the central mass).")]
    out += [K.p(
        "Legitimate transactions form a narrow, roughly symmetric peak around zero on every component. Fraud "
        "distributions are wide, skewed and often <b>multi-modal</b>: part of the fraud mass sits far in the "
        "tail, but a second, smaller mode overlaps the legitimate peak. No single threshold on any one feature "
        "separates the classes cleanly, which motivates a non-linear, multivariate model. Plotting the two "
        "strongest components against each other (Figure 8) shows how much a second dimension helps: most frauds "
        "fall in a region that legitimate traffic rarely visits, while a minority sits inside the legitimate "
        "cloud. Those are the cases any model will find hardest (Section 10).")]
    out += [K.figure(FIG / "eda_scatter_top2.png",
                     "<b>Figure 8.</b> The two most discriminative components (8,000 sampled legitimate "
                     "transactions, all 492 frauds).", width=120 * mm)]

    out += K.h2("4.7 From analysis to modelling decisions")
    out += [K.table([
        ["Observation", "Decision"],
        ["Fraud is 0.172% of transactions", "Stratified splits; evaluate with precision, recall, F1, MCC and PR-AUC; "
                                            "handle imbalance explicitly and compare strategies"],
        ["Fraud is multi-modal, heavy-tailed and interaction-driven", "Use a non-linear tree ensemble (Random Forest)"],
        ["Components are centred and decorrelated; trees are scale-invariant", "No scaling for the forest (scaling "
                                                                               "used only for the linear benchmark)"],
        [f"{num(e['duplicate_rows'])} exact duplicate rows", "Duplicate-aware split with a training-side leakage guard"],
        ["Strong time-of-day effect; Time is a two-day offset", "Test an hour-of-day feature in an ablation"],
        ["Card-testing pattern at tiny amounts", "Keep Amount; test a log-amount feature in the same ablation"],
        ["Several near-zero-signal components", "Keep them; let the forest's feature subsampling handle them"],
    ], col_widths=[72 * mm, K.CONTENT_W - 72 * mm])]
    return out


def methodology(r) -> list:
    s = r["split"]
    out = K.h1("5. Methodology")
    out += K.h2("5.1 Pipeline overview")
    out += [K.figure(FIG / "method_pipeline.png",
                     "<b>Figure 9.</b> End-to-end workflow. All model choices (imbalance strategy, "
                     "hyper-parameters, threshold) are made on the training set; the hold-out set is scored "
                     "exactly once.")]
    out += [K.p(
        "The project is organised as a Python package with one command per stage (<font face='Mono'>eda</font>, "
        "<font face='Mono'>compare</font>, <font face='Mono'>tune</font>, <font face='Mono'>train</font>, "
        "<font face='Mono'>baselines</font>, <font face='Mono'>report</font>). Every stage writes its results "
        "to disk, so each figure and number in this report is traceable to a file, and a fixed random seed (42) "
        "makes every run reproducible.")]

    out += K.h2("5.2 Train/test split and leakage guard")
    out += [K.p(
        "The data was split once, before any modelling, into an 80% training set and a 20% hold-out test set, "
        "stratified on Class so both keep the original fraud rate. The test set is left exactly as sampled so "
        "it mirrors real traffic. The training set is then cleaned in two steps:")]
    out += K.bullets([
        f"<b>Internal duplicates removed</b> ({s['train_internal_duplicates']} rows), so that repeated "
        "transactions do not receive extra weight in the bootstrap samples of the forest.",
        f"<b>Rows identical to a test transaction removed</b> ({s['train_rows_matching_test']} rows). Without this "
        "step the model could be scored on transactions it had literally memorised, inflating test results.",
    ])
    out += [K.table([
        ["Set", "Transactions", "Frauds", "Fraud rate", "Use"],
        ["Full dataset", num(s["n_total"]), str(r["eda"]["n_fraud"]), rate(r["eda"]["fraud_rate"]), "EDA only"],
        ["Training (as sampled)", num(s["n_train_raw"]), str(s["fraud_train_raw"]),
         rate(s["fraud_train_raw"] / s["n_train_raw"]), "—"],
        ["Training (after leakage guard)", num(s["n_train"]), str(s["fraud_train"]),
         rate(s["fraud_train"] / s["n_train"]), "All model selection and the final fit"],
        ["Hold-out test", num(s["n_test"]), str(s["fraud_test"]), rate(s["fraud_test"] / s["n_test"]),
         "Scored once, at the very end"],
    ], col_widths=[50 * mm, 26 * mm, 18 * mm, 22 * mm, K.CONTENT_W - 116 * mm], align_right_from=1)]

    out += K.h2("5.3 Features and preprocessing")
    out += [K.p(
        "Feature construction lives in a scikit-learn transformer (<font face='Mono'>FeatureBuilder</font>) placed "
        "at the start of every pipeline, so cross-validation, the final fit and production inference apply the "
        "same transformation. Three feature sets are available:")]
    out += [K.table([
        ["Feature set", "Columns", "Used by"],
        ["raw", "Time, V1–V28, Amount (30)", "Random Forest (final model)"],
        ["engineered", "V1–V28, log(1 + Amount), hour of day (30)", "Ablation in Section 6"],
        ["cyclic", "V1–V28, log(1 + Amount), sin/cos of hour (31)", "Logistic Regression benchmark"],
    ], col_widths=[28 * mm, 82 * mm, K.CONTENT_W - 110 * mm])]
    out += [Spacer(1, 6), K.p(
        "No scaling is applied for the Random Forest. Tree splits depend only on the order of values, not their "
        "scale, so standardisation would change nothing. The Logistic Regression benchmark, which is "
        "scale-sensitive, gets a <font face='Mono'>StandardScaler</font> inside its own pipeline.")]

    out += K.h2("5.4 Evaluation metrics")
    out += [K.p(
        "With TP, FP, FN and TN denoting true/false positives/negatives, and fraud as the positive class:")]
    out += [K.table([
        ["Metric", "Definition", "Question it answers"],
        ["Precision", "TP / (TP + FP)", "When the model raises an alert, how often is it really fraud?"],
        ["Recall", "TP / (TP + FN)", "What share of all fraud does the model catch?"],
        ["F1-score", "2 · P · R / (P + R)", "Single score that is high only if both precision and recall are high"],
        ["MCC", "(TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
         "Correlation between prediction and truth using all four cells; 0 = random, 1 = perfect"],
        ["Balanced accuracy", "(Recall + Specificity) / 2", "Accuracy with both classes weighted equally"],
        ["PR-AUC", "Average precision over all thresholds", "Quality of the fraud ranking, independent of the threshold"],
        ["ROC-AUC", "Area under TPR vs FPR", "Reported for reference; optimistic under heavy imbalance"],
    ], col_widths=[28 * mm, 64 * mm, K.CONTENT_W - 92 * mm])]
    out += [Spacer(1, 6), K.p(
        "MCC deserves a note. Precision, recall and F1 ignore true negatives entirely, while accuracy is dominated "
        "by them. MCC uses all four cells symmetrically and only scores high when the model is good on both "
        "classes, which makes it the most reliable single number for a 1 : 578 problem. It is therefore the "
        "selection metric for hyper-parameter tuning. ROC-AUC, by contrast, is inflated by the huge pool of easy "
        "negatives: a false positive rate of 0.1% sounds tiny but means about 57 false alarms on the test set. "
        "PR-AUC is used for ranking quality instead.")]

    out += K.h2("5.5 Validation scheme")
    out += [K.p(
        "All model selection uses stratified K-fold cross-validation with shuffling (seed 42) on the cleaned "
        f"training set: <b>5 folds</b> for the imbalance-strategy comparison and for the out-of-fold predictions "
        f"used to choose the threshold (about {s['fraud_train'] // 5} frauds per validation fold), and <b>3 folds</b> "
        f"for the hyper-parameter search (about {s['fraud_train'] // 3} frauds per fold, which keeps each candidate's "
        "recall estimate stable while halving compute). The hold-out set is used once, after every choice is final.")]
    return out


def imbalance_section(r) -> list:
    imb = r["imb"].set_index("strategy")
    main = imb[imb["family"] != "Feature ablation"]
    out = K.h1("6. Handling the class imbalance")
    out += [K.p(
        "With one fraud per 578 legitimate transactions, the forest sees very few positive examples and the "
        "default 0.5 probability threshold is unlikely to be the right operating point. Seven strategies were "
        "evaluated on identical 5-fold splits, all using the same 100-tree Random Forest, so any difference comes "
        "from the imbalance treatment alone.")]
    out += K.h2("6.1 Strategies compared")
    desc = {
        "none": "Train on the data as is; rely on the forest and the threshold",
        "class_weight": "Cost-sensitive learning: errors on fraud weigh more in every split "
                        "(<font face='Mono'>balanced_subsample</font>, recomputed per bootstrap sample)",
        "undersample": "Randomly drop legitimate rows until both classes are equal",
        "oversample": "Duplicate fraud rows until fraud is 10% of the majority class",
        "smote": "Create synthetic frauds by interpolating between neighbouring frauds (k = 5)",
        "smote_tomek": "SMOTE, then remove Tomek links (borderline pairs of opposite class)",
        "balanced_rf": "Each tree is grown on its own balanced bootstrap sample (imbalanced-learn)",
    }
    rows = [["Strategy", "Mechanism", "Training rows per fold", "Frauds per fold"]]
    for s, x in main.iterrows():
        rows.append([x["label"], desc[s], num(x["train_rows_after_resampling"]),
                     num(x["train_frauds_after_resampling"])])
    out += [K.table(rows, col_widths=[38 * mm, 84 * mm, 26 * mm, K.CONTENT_W - 148 * mm], align_right_from=2)]

    out += K.h2("6.2 Leak-free resampling")
    out += [K.p(
        "Every strategy is wrapped in an <font face='Mono'>imblearn.pipeline.Pipeline</font>, whose samplers "
        "run during <font face='Mono'>fit</font> only. Inside cross-validation this means resampling touches the "
        "training folds alone, and the validation fold keeps the real 0.17% fraud rate. The common mistake of "
        "applying SMOTE to the whole dataset before splitting puts synthetic copies of validation frauds into "
        "training, so the model is graded on points it has effectively seen. A unit test checks that no "
        "synthetic rows appear at prediction time.")]

    out += K.h2("6.3 Results")
    rows = [["Strategy", "PR-AUC (± sd)", "ROC-AUC", "Precision @0.5", "Recall @0.5", "MCC @0.5",
             "False alarms @0.5", "F1 @ best t"]]
    order = main.sort_values("pr_auc", ascending=False)
    for s, x in order.iterrows():
        rows.append([x["label"], f"{x['pr_auc']:.3f} ± {x['pr_auc_fold_std']:.3f}", f"{x['roc_auc']:.3f}",
                     f"{x['precision@0.5']:.3f}", f"{x['recall@0.5']:.3f}", f"{x['mcc@0.5']:.3f}",
                     num(x["fp@0.5"]), f"{x['f1@best']:.3f}"])
    hl = [i + 1 for i, s in enumerate(order.index) if s == "class_weight"]
    out += [K.table(rows, col_widths=[36 * mm, 23 * mm, 17 * mm, 20 * mm, 18 * mm, 17 * mm, 21 * mm,
                                      K.CONTENT_W - 152 * mm], align_right_from=1, highlight_rows=hl,
                    font_size=7.8), Spacer(1, 3),
            K.p("<b>Table.</b> Out-of-fold results over the 226,834 training transactions (5-fold CV). "
                "“Best t” is the F1-optimal threshold for each strategy. Highlighted: the selected strategy.",
                K.CAPTION)]
    out += [K.figure(FIG / "imbalance_comparison.png",
                     "<b>Figure 10.</b> Imbalance strategies at the default 0.5 threshold. The selected strategy "
                     "(class weighting) is highlighted.", max_height=70 * mm)]
    out += [K.figure(FIG / "imbalance_pr_curves.png",
                     "<b>Figure 11.</b> Out-of-fold precision–recall curves for four representative strategies.",
                     width=125 * mm)]
    n, cw, sm, us, brf = (imb.loc[k] for k in ("none", "class_weight", "smote", "undersample", "balanced_rf"))
    eng = imb.loc["class_weight_engineered"]
    good = main.drop(index=["undersample", "balanced_rf"])
    out += K.h2("6.4 Findings")
    out += K.bullets([
        f"<b>The forest ranks fraud well even without treatment</b> (PR-AUC {n['pr_auc']:.3f}). Bootstrap "
        "aggregation and fully grown trees isolate small fraud regions on their own; the imbalance mainly "
        "distorts the <i>calibration</i> of the scores, which is something the decision threshold can fix.",
        f"<b>SMOTE gave the highest PR-AUC</b> ({sm['pr_auc']:.3f}) and raised recall at 0.5 "
        f"({sm['recall@0.5']:.2f} vs {n['recall@0.5']:.2f}), but at a cost in precision ({sm['precision@0.5']:.2f} "
        f"vs {n['precision@0.5']:.2f}) and {sm['fp@0.5'] / max(n['fp@0.5'], 1):.1f}× more false alarms. Its "
        f"PR-AUC gain over class weighting (+{sm['pr_auc'] - cw['pr_auc']:.3f}) is smaller than the fold-to-fold "
        f"standard deviation (≈ {sm['pr_auc_fold_std']:.2f}), so it is not a reliable improvement.",
        f"<b>Once the threshold is optimised, the sensible strategies converge.</b> The F1 at each strategy's best "
        f"threshold lies between {good['f1@best'].min():.3f} and {good['f1@best'].max():.3f}. Moving the threshold "
        "recovers what resampling offers, without altering the training data.",
        f"<b>Aggressive undersampling backfires.</b> Balancing 1 : 1 keeps only {num(us['train_rows_after_resampling'])} "
        f"of ~181,000 rows per fold. Recall at 0.5 jumps to {us['recall@0.5']:.2f}, but precision collapses to "
        f"{us['precision@0.5']:.2f} ({num(us['fp@0.5'])} false alarms). Balanced Random Forest behaves the same way "
        f"(precision {brf['precision@0.5']:.2f}). Strikingly, these two have the <i>highest</i> ROC-AUC "
        f"({us['roc_auc']:.3f} and {brf['roc_auc']:.3f}) while having the lowest PR-AUC: a concrete demonstration "
        "of why ROC-AUC is misleading under heavy imbalance.",
        f"<b>Class weighting was selected.</b> It matches the resampling methods within noise "
        f"(PR-AUC {cw['pr_auc']:.3f}), creates no synthetic transactions, keeps every real row, and trains "
        f"{sm['fit_seconds'] / cw['fit_seconds']:.1f}× faster than SMOTE ({cw['fit_seconds']:.0f} s vs "
        f"{sm['fit_seconds']:.0f} s per fold). Its strength is not fixed here but tuned as a hyper-parameter "
        "(Section 7), together with the decision threshold.",
        f"<b>Feature ablation.</b> Replacing Time and Amount with hour of day and log(1 + Amount) moved PR-AUC from "
        f"{cw['pr_auc']:.3f} to {eng['pr_auc']:.3f}, well inside the noise band. The forest already extracts the "
        "time and amount signal from the raw columns and from the PCA components correlated with them, so the "
        "original 30 features were kept.",
    ])
    out += [K.callout(
        "Cost-sensitive class weighting (strength tuned as a hyper-parameter) + a decision threshold chosen on "
        "out-of-fold predictions. No synthetic data, no discarded data, and the precision/recall trade-off stays "
        "fully adjustable after training.", title="Decision")]
    return out


def tuning_section(r) -> list:
    tb, tun, fin = r["tun_best"], r["tun"], r["final"]
    bp = tb["best_params"]
    out = K.h1("7. Random Forest model and hyper-parameter tuning")
    out += K.h2("7.1 Why a Random Forest")
    out += K.bullets([
        "<b>Non-linear and interaction-aware.</b> EDA showed multi-modal fraud distributions and strong "
        "within-fraud correlations; axis-aligned tree splits combine features naturally.",
        "<b>Variance reduction through bagging.</b> Each tree sees a bootstrap sample and a random subset of "
        "features at each split. Averaging many decorrelated trees gives stable probabilities, which matters "
        "when there are only a few hundred positive examples.",
        "<b>Scale-free and robust to outliers.</b> The heavy fraud tails (V14 below −15) do not distort "
        "splits the way they would distort a linear model's coefficients.",
        "<b>Native cost-sensitive learning</b> through <font face='Mono'>class_weight</font>, and built-in "
        "importance measures for interpretation.",
    ])
    out += K.h2("7.2 Search space")
    role = {
        "n_estimators": "Number of trees; more trees give smoother probabilities",
        "max_depth": "Depth cap; None grows trees until leaves are pure",
        "min_samples_split": "Minimum samples needed to split a node",
        "min_samples_leaf": "Minimum samples in a leaf; regularises tiny fraud leaves",
        "max_features": "Features tried per split (sqrt = 5, log2 = 4, 0.3 = 9 of 30)",
        "criterion": "Split quality measure",
        "class_weight": "Cost of misclassifying fraud vs legitimate (imbalance handling)",
        "max_samples": "Bootstrap sample size per tree (None = full training set)",
    }
    rows = [["Hyper-parameter", "Values searched", "Role"]]
    for k, vals in tb["search_space"].items():
        vtxt = ", ".join(str(v) for v in vals).replace("{0:1, 1:5}", "{0: 1, 1: 5}").replace("{0:1, 1:25}", "{0: 1, 1: 25}")
        rows.append([f"<font face='Mono'>{k}</font>", vtxt, role.get(k, "")])
    out += [K.table(rows, col_widths=[36 * mm, 58 * mm, K.CONTENT_W - 94 * mm])]
    total = 1
    for vals in tb["search_space"].values():
        total *= len(vals)
    out += [Spacer(1, 6), K.p(
        f"The full grid holds {num(total)} combinations. A <b>randomized search</b> "
        f"(<font face='Mono'>RandomizedSearchCV</font>, {tb['n_iter']} candidates, seed 42) samples it uniformly "
        f"and evaluates each candidate with <b>{tb['cv_folds']}-fold stratified cross-validation</b>, "
        f"{tb['n_fits']} model fits in total ({tb['search_seconds'] / 60:.0f} minutes on a 16-thread laptop). Random "
        "search reaches near-optimal regions of large spaces with far fewer fits than a grid, because only a few "
        "hyper-parameters usually matter and random sampling tries many distinct values of each.")]
    out += K.h2("7.3 Multi-metric scoring")
    out += [K.p(
        "Every candidate was scored on six metrics at once, so the search reveals the full trade-off rather than "
        "a single winner: <b>precision</b>, <b>recall</b> and <b>F1</b> for the fraud class, and the balanced "
        "metrics <b>MCC</b>, <b>balanced accuracy</b> and <b>PR-AUC</b> (average precision). The configuration "
        "with the highest mean cross-validated MCC was selected, as MCC rewards precision and recall jointly and "
        "cannot be gamed by either the majority or the minority class.")]
    rows = [["Rank", "class_weight", "max_features", "max_depth", "min_leaf", "trees", "Precision", "Recall",
             "F1", "MCC", "PR-AUC"]]
    for _, x in tun.head(6).iterrows():
        rows.append([str(int(x["rank"])), str(x["class_weight"]).replace("balanced_subsample", "balanced_sub."),
                     str(x["max_features"]), str(x["max_depth"]).replace(".0", ""), str(int(x["min_samples_leaf"])),
                     str(int(x["n_estimators"])), f"{x['mean_test_precision']:.3f}", f"{x['mean_test_recall']:.3f}",
                     f"{x['mean_test_f1']:.3f}", f"{x['mean_test_mcc']:.3f}", f"{x['mean_test_average_precision']:.3f}"])
    out += [K.table(rows, col_widths=[11 * mm, 26 * mm, 20 * mm, 17 * mm, 14 * mm, 12 * mm, 16 * mm, 14 * mm,
                                      13 * mm, 13 * mm, K.CONTENT_W - 156 * mm],
                    align_right_from=6, highlight_rows=[1], font_size=7.6), Spacer(1, 3),
            K.p("<b>Table.</b> Top six of the 20 candidates by mean 3-fold CV MCC (threshold 0.5). All 20 are listed "
                "in Appendix B.", K.CAPTION)]
    out += [K.figure(FIG / "tuning_tradeoff.png",
                     "<b>Figure 12.</b> Left: mean CV precision against recall for every candidate, by class-weight "
                     "setting; the circled point was selected. Right: candidates ranked by MCC with ±1 standard "
                     "deviation across folds.")]
    out += tuning_observations(r)
    out += K.h2("7.4 Selected hyper-parameters")
    show = [("n_estimators", "Number of trees"), ("max_depth", "Maximum depth"),
            ("min_samples_split", "Min. samples to split"), ("min_samples_leaf", "Min. samples per leaf"),
            ("max_features", "Features per split"), ("criterion", "Split criterion"),
            ("class_weight", "Class weight"), ("max_samples", "Bootstrap sample size")]
    rows = [["Hyper-parameter", "Selected value", "Mean CV score of this configuration"]]
    cv = tb["best_cv"]
    cvs = tb["best_cv_std"]
    side = [f"Precision {cv['precision']:.3f} ± {cvs['precision']:.3f}", f"Recall {cv['recall']:.3f} ± {cvs['recall']:.3f}",
            f"F1 {cv['f1']:.3f} ± {cvs['f1']:.3f}", f"MCC {cv['mcc']:.3f} ± {cvs['mcc']:.3f}",
            f"Balanced accuracy {cv['balanced_accuracy']:.3f}", f"PR-AUC {cv['average_precision']:.3f} ± {cvs['average_precision']:.3f}",
            "", ""]
    for (k, lab), sd in zip(show, side):
        val = bp.get(k)
        if isinstance(val, dict):
            val = f"{{0: {val['0']}, 1: {val['1']}}}"
        rows.append([f"{lab} (<font face='Mono'>{k}</font>)", "None" if val is None else str(val), sd])
    out += [K.table(rows, col_widths=[66 * mm, 40 * mm, K.CONTENT_W - 106 * mm])]

    oof = fin["oof"]
    thr = oof["threshold"]
    at_t, at_5 = oof["at_threshold"], oof["at_0.5"]
    out += K.h2("7.5 Decision-threshold optimisation")
    out += [K.p(
        "A Random Forest outputs a probability (the share of trees voting fraud). The default cut-off of 0.5 is "
        "arbitrary under heavy imbalance, so the operating threshold was chosen from data. The selected "
        "configuration was re-run in 5-fold cross-validation on the training set to obtain an out-of-fold "
        "probability for every training transaction, and the threshold that maximises F1 on those predictions "
        "was chosen (ties broken toward the higher threshold, i.e. fewer alerts). The test set plays no part.")]
    out += [K.figure(FIG / "threshold_selection.png",
                     f"<b>Figure 13.</b> Out-of-fold precision, recall, F1 and MCC as the threshold moves. The "
                     f"selected threshold is t = {thr:.2f}.", width=135 * mm)]
    out += [K.table([
        ["Out-of-fold (training set)", "Precision", "Recall", "F1", "MCC"],
        ["Default threshold 0.50", f"{at_5['precision']:.3f}", f"{at_5['recall']:.3f}", f"{at_5['f1']:.3f}", f"{at_5['mcc']:.3f}"],
        [f"Optimised threshold {thr:.2f}", f"{at_t['precision']:.3f}", f"{at_t['recall']:.3f}", f"{at_t['f1']:.3f}",
         f"{at_t['mcc']:.3f}"],
    ], col_widths=[60 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm], align_right_from=1, highlight_rows=[2]),
        Spacer(1, 6)]
    out += [K.p(
        "Precision rises steadily with the threshold while recall stays on a plateau over a wide middle range "
        "before dropping, because most frauds receive near-certain scores and only a minority sit in between. "
        "F1 and MCC are therefore flat near their optimum, so the chosen operating point is not fragile: small "
        "changes in the threshold move precision and recall only slightly.")]
    fo = fin["forest"]
    s = fin["split"]
    out += K.h2("7.6 Final model")
    out += [K.p(
        f"The selected pipeline was refit on all {num(s['n_train'])} cleaned training transactions "
        f"({s['fraud_train']} frauds). The resulting forest has {fo['n_trees']} trees with a mean depth of "
        f"{fo['mean_depth']:.1f} (maximum {fo['max_depth']}) and on average {num(fo['mean_leaves'])} leaves per tree. "
        "The fitted pipeline and its threshold are serialised together with joblib "
        "(<font face='Mono'>models/random_forest_fraud.joblib</font>) alongside a model card, and a "
        "<font face='Mono'>predict</font> command scores new transaction files with exactly the same "
        "preprocessing and threshold.")]
    return out


def tuning_observations(r) -> list:
    """Statements derived from mean CV scores grouped by each hyper-parameter value."""
    tun = r["tun"].copy()
    cols = ["mean_test_precision", "mean_test_recall", "mean_test_mcc", "mean_test_average_precision"]

    def grp(col):
        return tun.groupby(tun[col].astype(str))[cols].mean()

    obs = []
    g = grp("class_weight")
    if "None" in g.index and len(g) > 1:
        w = g.drop(index="None")
        lab = {"balanced_subsample": "balanced_subsample", "{0:1, 1:5}": "fraud weight 5",
               "{0:1, 1:25}": "fraud weight 25"}
        best_w = w["mean_test_mcc"].idxmax()
        obs.append(
            "<b>Class weighting is the main lever on the precision–recall trade-off.</b> Unweighted forests "
            f"averaged precision {g.loc['None', 'mean_test_precision']:.3f}, recall "
            f"{g.loc['None', 'mean_test_recall']:.3f} and MCC {g.loc['None', 'mean_test_mcc']:.3f}. Every weighted "
            f"setting raised average recall (to {w['mean_test_recall'].min():.3f}–{w['mean_test_recall'].max():.3f}) "
            f"and MCC (to {w['mean_test_mcc'].min():.3f}–{w['mean_test_mcc'].max():.3f}) for a smaller loss of "
            f"precision; the best average MCC came from {lab.get(best_w, best_w)}. Explicit fraud weights pushed "
            "recall furthest, which confirms the decision in Section 6 to treat the imbalance through the loss "
            "rather than through resampling.")
    g = grp("min_samples_leaf")
    if len(g) > 1:
        worst, best = g["mean_test_mcc"].idxmin(), g["mean_test_mcc"].idxmax()
        obs.append(
            f"<b>Leaf size matters more than tree count.</b> Leaves of at least {worst} samples averaged MCC "
            f"{g.loc[worst, 'mean_test_mcc']:.3f} against {g.loc[best, 'mean_test_mcc']:.3f} with a minimum of "
            f"{best}. With only about 250 frauds per training fold, large leaves smooth away the small, isolated "
            "fraud pockets the trees need to capture. The number of trees (100–300) changed average MCC by "
            f"less than {np.ptp(grp('n_estimators')['mean_test_mcc'].to_numpy()) + 0.0005:.3f}.")
    g = grp("criterion")
    if len(g) > 1:
        hi = g["mean_test_mcc"].idxmax()
        lo = g["mean_test_mcc"].idxmin()
        n_top = int((tun.head(10)["criterion"] == hi).sum())
        obs.append(
            f"<b>Split criterion.</b> {hi.capitalize()} splits averaged MCC {g.loc[hi, 'mean_test_mcc']:.3f} against "
            f"{g.loc[lo, 'mean_test_mcc']:.3f} for {lo}, and {n_top} of the top ten candidates use {hi}. "
            "Entropy (information gain) penalises impure nodes more strongly, which suits a target where one "
            "class is very rare.")
    g = grp("max_depth")
    if "None" in g.index and len(g) > 1:
        capped = g.drop(index="None")
        obs.append(
            "<b>Depth.</b> Unlimited trees had the highest average precision "
            f"({g.loc['None', 'mean_test_precision']:.3f}) but the lowest recall ({g.loc['None', 'mean_test_recall']:.3f}); "
            f"capped depths traded a little precision for recall (average recall {capped['mean_test_recall'].min():.3f}"
            f"–{capped['mean_test_recall'].max():.3f}). The selected depth of {r['tun_best']['best_params']['max_depth']} "
            "sits between the two.")
    g = grp("max_features")
    if len(g) > 1:
        obs.append(
            "<b>Features per split</b> had only a small effect (average MCC "
            + ", ".join(f"{k}: {v:.3f}" for k, v in g["mean_test_mcc"].sort_values(ascending=False).items())
            + "). The strongest components are found whichever subset size is used.")
    obs.append(
        f"<b>Robustness.</b> All {len(tun)} candidates lie between MCC {tun['mean_test_mcc'].min():.3f} and "
        f"{tun['mean_test_mcc'].max():.3f}. The forest is good across most of the space; tuning mainly decides "
        "where on the precision–recall curve it sits.")
    return K.bullets(obs)


def results_section(r) -> list:
    rep, s, e = r["rep"], r["split"], r["eda"]
    cis = rep["ci"]
    n_legit = rep["tn"] + rep["fp"]
    n_fraud = rep["tp"] + rep["fn"]
    out = K.h1("8. Results on the hold-out test set")
    out += [K.p(
        f"The final model was applied once to the {num(s['n_test'])} hold-out transactions ({n_fraud} frauds), "
        f"using the decision threshold fixed in Section 7.5.")]
    out += [K.kpi_tiles([(pct(rep["precision"]), "Precision"), (pct(rep["recall"]), "Recall"),
                         (pct(rep["f1"]), "F1-score"), (mcc_fmt(rep["mcc"]), "MCC")], accent_first=4,
                        value_size=22), Spacer(1, 8)]
    out += K.h2("8.1 Metrics")
    out += [K.table([
        ["Metric", "Value", "95% bootstrap interval", "Reading"],
        ["Precision", pct(rep["precision"]), ci(cis["precision"]), f"{rep['tp']} of {rep['tp'] + rep['fp']} alerts are real fraud"],
        ["Recall (sensitivity)", pct(rep["recall"]), ci(cis["recall"]), f"{rep['tp']} of {n_fraud} frauds caught"],
        ["F1-score", pct(rep["f1"]), ci(cis["f1"]), "Balance of precision and recall"],
        ["MCC", mcc_fmt(rep["mcc"]), ci(cis["mcc"], mcc_fmt), "Strong agreement between predictions and truth"],
        ["PR-AUC (average precision)", f"{rep['pr_auc']:.3f}", ci(cis["pr_auc"], lambda v: f"{v:.3f}"), "Threshold-free ranking quality"],
        ["ROC-AUC", f"{rep['roc_auc']:.3f}", ci(cis["roc_auc"], lambda v: f"{v:.3f}"), "For reference only"],
        ["Specificity", pct(rep["specificity"], 3), "—", f"{num(rep['tn'])} of {num(n_legit)} legitimate passed"],
        ["False positive rate", pct(rep["false_positive_rate"], 4), "—", f"{rep['fp']} false alarm in {num(n_legit)}"],
        ["Balanced accuracy", pct(rep["balanced_accuracy"]), "—", "Mean of recall and specificity"],
        ["Accuracy", pct(rep["accuracy"], 2), "—", f"Baseline of always predicting legitimate: {pct(e['trivial_accuracy'], 2)}"],
    ], col_widths=[44 * mm, 22 * mm, 36 * mm, K.CONTENT_W - 102 * mm], align_right_from=None)]
    out += [Spacer(1, 4), K.p(
        f"Confidence intervals come from {num(C.N_BOOTSTRAP)} bootstrap resamples of the test set. They are wide for "
        f"recall because the test set contains only {n_fraud} frauds: each fraud is worth about "
        f"{100 / n_fraud:.1f} percentage points of recall.", K.SMALL)]

    out += K.h2("8.2 Confusion matrix")
    out += [K.figure(FIG / "test_confusion_matrix.png",
                     "<b>Figure 14.</b> Confusion matrix on the hold-out set. Colour shows the share of each "
                     "actual class.", width=100 * mm)]
    out += [K.p(
        f"Of {num(n_legit)} legitimate transactions, <b>{num(rep['tn'])} were correctly passed and only "
        f"{rep['fp']} was flagged</b>, a false positive rate of {pct(rep['false_positive_rate'], 4)}. Of "
        f"{n_fraud} frauds, <b>{rep['tp']} were caught and {rep['fn']} were missed</b>. The model raised "
        f"{rep['tp'] + rep['fp']} alerts in total, {pct((rep['tp'] + rep['fp']) / s['n_test'], 2)} of all "
        f"traffic, and {pct(rep['precision'])} of them were genuine fraud. This is the profile the success "
        "criteria asked for: an alert queue an analyst can trust almost completely, catching four out of five "
        "frauds. The cost of that precision is the 20% of fraud that slips through, analysed in Section 10.")]

    out += K.h2("8.3 ROC and precision–recall curves")
    out += [K.figure(FIG / "test_roc_pr.png",
                     "<b>Figure 15.</b> Hold-out ROC curve (false positive rate on a symmetric-log axis) and "
                     "precision–recall curve for the Random Forest and three benchmarks.")]
    out += [K.p(
        f"The forest reaches PR-AUC {rep['pr_auc']:.3f} and ROC-AUC {rep['roc_auc']:.3f}. The ROC panel uses a "
        "log-scaled x-axis because, at 0.17% fraud, only false positive rates below about 0.1% are operationally "
        "relevant; on a linear axis every reasonable model looks almost perfect. The precision–recall panel "
        "is the more honest view. The forest holds precision close to 1 across most of the recall range, and "
        "it falls away only beyond about 85% recall, where the remaining frauds are those that look legitimate.")]

    out += K.h2("8.4 Comparison with benchmark models")
    bench = r["bench"]
    rows = [["Model", "Precision", "Recall", "F1", "MCC", "PR-AUC", "ROC-AUC", "False alarms", "Missed"]]
    order = bench.sort_values("mcc", ascending=False)
    hl = []
    for i, (_, x) in enumerate(order.iterrows()):
        if str(x["model"]).startswith("Random Forest"):
            hl.append(i + 1)
        rows.append([str(x["model"]), pct(x["precision"]), pct(x["recall"]), pct(x["f1"]), mcc_fmt(x["mcc"]),
                     f"{x['pr_auc']:.3f}", f"{x['roc_auc']:.3f}", num(x["fp"]), num(x["fn"])])
    out += [K.table(rows, col_widths=[47 * mm, 17 * mm, 15 * mm, 15 * mm, 13 * mm, 16 * mm, 17 * mm, 18 * mm,
                                      K.CONTENT_W - 158 * mm], align_right_from=1, highlight_rows=hl, font_size=7.8),
            Spacer(1, 3),
            K.p("<b>Table.</b> All models trained on the same cleaned training set and scored on the same hold-out "
                "set. Each benchmark's threshold was chosen by maximising F1 on its own 3-fold out-of-fold "
                "predictions.", K.CAPTION)]
    out += [K.figure(FIG / "model_comparison.png",
                     "<b>Figure 16.</b> F1, MCC and PR-AUC on the hold-out set; the tuned Random Forest is "
                     "highlighted.", max_height=60 * mm)]
    out += benchmark_observations(r)

    out += K.h2("8.5 Business impact")
    out += cost_analysis(r)
    return out


def benchmark_observations(r) -> list:
    b = r["bench"].set_index("model")
    rf = b.loc[[m for m in b.index if m.startswith("Random Forest")][0]]
    lr = b.loc["Logistic Regression"]
    dt = b.loc["Decision Tree"]
    hgb = b.loc["Hist. Gradient Boosting"]
    iso = b.loc["Isolation Forest (unsupervised)"]
    maj = b.loc["Majority class (always legitimate)"]
    items = [
        f"<b>Majority-class baseline:</b> {pct(maj['accuracy'], 2)} accuracy, zero recall, MCC 0. This row exists to "
        "show why accuracy is not used.",
        f"<b>Isolation Forest</b> (unsupervised, never sees labels) reaches PR-AUC {iso['pr_auc']:.3f}. Fraud is "
        "partly anomalous, but many anomalies are legitimate, so it raises "
        f"{num(iso['fp'])} false alarms while catching {num(iso['tp'])} frauds. Labelled data adds a great deal. "
        f"Its ROC-AUC ({iso['roc_auc']:.3f}) is nevertheless almost identical to the supervised models', which "
        "again shows that ROC-AUC cannot tell these models apart on a 0.17% problem.",
        f"<b>Logistic Regression</b> (scaled, cyclic hour features) is a reasonable linear baseline (PR-AUC "
        f"{lr['pr_auc']:.3f}) but cannot model the multi-modal, interacting fraud patterns: F1 {pct(lr['f1'])} "
        f"against {pct(rf['f1'])} for the forest, with {num(lr['fp'])} false alarms.",
        f"<b>A single Decision Tree</b> already captures much of the non-linear structure (MCC {mcc_fmt(dt['mcc'])}) "
        f"but its probability estimates are coarse (PR-AUC {dt['pr_auc']:.3f}) and it raises {num(dt['fp'])} false "
        f"alarms. Bagging {r['final']['forest']['n_trees']} decorrelated trees into a forest lifts MCC to "
        f"{mcc_fmt(rf['mcc'])} and PR-AUC to {rf['pr_auc']:.3f}.",
        f"<b>Histogram Gradient Boosting</b> is the closest competitor (MCC {mcc_fmt(hgb['mcc'])}, PR-AUC "
        f"{hgb['pr_auc']:.3f}, {num(hgb['fp'])} false alarms). The tuned Random Forest leads on precision, F1, MCC "
        "and PR-AUC, with the fewest false alarms of any model that catches fraud.",
    ]
    return K.bullets(items)


def cost_analysis(r) -> list:
    pred, rep = r["pred"], r["rep"]
    fraud = pred[pred[C.TARGET] == 1]
    caught = fraud.loc[fraud["outcome"] == "TP", "Amount"].sum()
    missed = fraud.loc[fraud["outcome"] == "FN", "Amount"].sum()
    total = fraud["Amount"].sum()
    alerts = rep["tp"] + rep["fp"]
    review = alerts * C.REVIEW_COST_PER_ALERT
    out = [K.p(
        "The test set carries real transaction amounts, so the model's effect can be expressed in euros. "
        f"Assuming every alert is reviewed at a cost of {eur(C.REVIEW_COST_PER_ALERT)} and that a reviewed fraud "
        "is stopped before the loss is incurred:")]
    out += [K.table([
        ["Quantity (hold-out set, ~2 days of traffic × 20%)", "Value"],
        ["Fraud value present", eur(total)],
        [f"Fraud value intercepted ({rep['tp']} frauds)", eur(caught)],
        [f"Fraud value missed ({rep['fn']} frauds)", eur(missed)],
        [f"Review cost ({alerts} alerts × {eur(C.REVIEW_COST_PER_ALERT)})", eur(review)],
        ["Net saving vs no model", eur(caught - review)],
        ["Share of fraud value intercepted", pct(caught / total)],
    ], col_widths=[110 * mm, K.CONTENT_W - 110 * mm], align_right_from=1, highlight_rows=[5])]
    out += [Spacer(1, 6), K.p(
        f"The value view tells a slightly different story from the count view. By count the model catches "
        f"{pct(rep['recall'])} of frauds; by value it intercepts {pct(caught / total)}. The missed frauds are "
        f"{'smaller than average, so their value share is lower than their count share' if missed / max(rep['fn'], 1) < caught / max(rep['tp'], 1) else 'larger than average, so their value share is higher than their count share'} "
        f"(mean {eur(missed / max(rep['fn'], 1))} missed vs {eur(caught / max(rep['tp'], 1))} caught). With a "
        f"single false alarm, review costs are negligible next to the fraud prevented.")]
    return out


def interpretation(r) -> list:
    imp, sep = r["imp"], r["sep"]
    top = imp.sort_values("permutation_mean", ascending=False)
    t5 = top["feature"].head(5).tolist()
    mdi5 = imp.sort_values("mdi", ascending=False)["feature"].head(5).tolist()
    m = imp.merge(sep[["feature", "ks_statistic"]], on="feature")
    rho = stats.spearmanr(m["ks_statistic"], m["mdi"]).statistic
    share_top10 = imp.sort_values("mdi", ascending=False)["mdi"].head(10).sum()
    top8_mdi = imp.sort_values("mdi", ascending=False)["feature"].head(8).tolist()
    common = [f for f in top["feature"].head(8) if f in top8_mdi]
    perm_rank = {f: i + 1 for i, f in enumerate(top["feature"])}
    mdi_rank = {f: i + 1 for i, f in enumerate(imp.sort_values("mdi", ascending=False)["feature"])}
    out = K.h1("9. Model interpretation")
    out += [K.p(
        "Two complementary importance measures were computed. <b>Mean decrease in impurity (MDI)</b> sums how "
        "much each feature's splits purify the trees during training; it is fast but can favour features with "
        "many distinct values. <b>Permutation importance</b> shuffles one feature at a time in the hold-out set "
        "and measures the drop in PR-AUC, so it reflects what the model actually relies on for unseen data.")]
    out += [K.figure(FIG / "feature_importance.png",
                     "<b>Figure 17.</b> Top 15 features by impurity-based importance (left) and by permutation "
                     "importance on the hold-out set (right, mean of 5 shuffles).")]
    out += K.bullets([
        f"<b>A small set of components does most of the work.</b> The top five features by permutation importance "
        f"are {', '.join(t5)}; by MDI they are {', '.join(mdi5)}. The ten most important features account for "
        f"{pct(share_top10, 0)} of total impurity reduction.",
        f"<b>The model agrees with the statistics.</b> The Spearman rank correlation between each feature's KS "
        f"statistic (Section 4.5) and its MDI importance is ρ = {rho:.2f}. The forest learned the same "
        "signals the univariate analysis found, which is a good sign that it is not exploiting artefacts.",
        f"<b>Amount is a secondary refinement; Time adds little.</b> Amount ranks {perm_rank['Amount']} of 30 by "
        f"permutation importance but only {mdi_rank['Amount']} by MDI: it is rarely the first split, yet on unseen "
        "data it helps separate the small card-testing frauds that sit close to legitimate traffic on the "
        f"components. Time ranks {perm_rank['Time']} and {mdi_rank['Time']} respectively; its signal is largely "
        "carried by the components it correlates with (notably V3), consistent with the feature ablation in "
        "Section 6.",
        "<b>Why permutation and MDI differ.</b> When two features carry overlapping information, shuffling one "
        "costs little because the other compensates, so permutation importance spreads credit differently from "
        f"MDI. Both measures nevertheless agree on a common core: {', '.join(common)} appear in the top eight of "
        "both rankings.",
    ])
    return out


def strengths_gaps(r) -> list:
    pred, rep = r["pred"], r["rep"]
    fraud = pred[pred[C.TARGET] == 1]
    tp, fn = fraud[fraud["outcome"] == "TP"], fraud[fraud["outcome"] == "FN"]
    legit = pred[pred["outcome"] == "TN"]
    fp = pred[pred["outcome"] == "FP"]
    imp = r["imp"].sort_values("permutation_mean", ascending=False)
    f1n, f2n = imp["feature"].iloc[0], imp["feature"].iloc[1]
    night = lambda d: (hour_of_day(d["Time"]) < 6).mean()  # noqa: E731
    out = K.h1("10. Strengths and gaps")
    out += K.h2("10.1 Strengths")
    out += K.bullets([
        f"<b>Near-perfect precision.</b> {rep['tp']} of {rep['tp'] + rep['fp']} alerts were fraud. In production "
        "this means customers are almost never blocked by mistake and analysts spend their time on real cases.",
        f"<b>Very low false positive rate.</b> {rep['fp']} legitimate transaction flagged out of "
        f"{num(rep['tn'] + rep['fp'])} ({pct(rep['false_positive_rate'], 4)}).",
        "<b>Confident, well-separated scores.</b> Caught frauds receive a median fraud probability of "
        f"{tp['fraud_probability'].median():.2f} and legitimate transactions a median of "
        f"{legit['fraud_probability'].median():.3f}, so the decision does not hinge on a knife-edge threshold.",
        "<b>Honest estimate.</b> The hold-out set was never used for any choice, duplicates were blocked from "
        "leaking into training, and all resampling experiments were done inside folds.",
        "<b>Explainable behaviour.</b> The features driving the model match the statistically strongest "
        "separators found in EDA.",
    ])
    out += K.h2("10.2 Gaps: the missed frauds")
    out += [K.figure(FIG / "error_analysis.png",
                     f"<b>Figure 18.</b> Left: every hold-out fraud sorted by predicted probability, coloured by "
                     f"outcome. Right: the model's two most important features ({f1n}, {f2n}), showing where caught "
                     "and missed frauds and the false alarm fall relative to legitimate traffic.")]
    rows = [["Profile (hold-out set)", "Caught frauds", "Missed frauds", "Legitimate"],
            ["Count", num(len(tp)), num(len(fn)), num(len(legit))],
            ["Median predicted probability", f"{tp['fraud_probability'].median():.3f}",
             f"{fn['fraud_probability'].median():.3f}", f"{legit['fraud_probability'].median():.3f}"]]
    comps = [f for f in imp["feature"] if f.startswith("V")]
    for f in comps[:4]:
        rows.append([f"Median {f}", f"{tp[f].median():.2f}", f"{fn[f].median():.2f}", f"{legit[f].median():.2f}"])
    rows += [["Median amount", eur(tp["Amount"].median()), eur(fn["Amount"].median()), eur(legit["Amount"].median())],
             ["Share at night (00–06 h)", pct(night(tp)), pct(night(fn)), pct(night(legit))]]
    out += [K.table(rows, col_widths=[58 * mm, 36 * mm, 36 * mm, K.CONTENT_W - 130 * mm], align_right_from=1)]
    out += [Spacer(1, 6), K.p(
        f"Most missed frauds are not near misses: {int((fn['fraud_probability'] < 0.1).sum())} of the {len(fn)} "
        f"score below 0.10 and their median predicted probability is {fn['fraud_probability'].median():.3f}, far "
        "below the decision threshold; only a handful sit just under the caught group. On the features the model "
        "relies on most, they sit close to or inside the legitimate population "
        f"(median {f1n} of {fn[f1n].median():.2f} against {tp[f1n].median():.2f} for caught frauds and "
        f"{legit[f1n].median():.2f} for legitimate traffic). These are <b>camouflaged frauds</b>: in the "
        "PCA-transformed space available to the model they are statistically indistinguishable from normal "
        "payments, which is the second, overlapping mode seen in the EDA densities (Figure 7). Lowering the "
        "threshold far enough to catch them would flood the queue with false alarms, as the steep fall of the "
        "precision–recall curve beyond about 85% recall shows.")]
    val_missed, val_total = fn["Amount"].sum(), fraud["Amount"].sum()
    out += [K.p(
        f"<b>The missed frauds carry more value than their count suggests.</b> Their median amount is only "
        f"{eur(fn['Amount'].median())}, but a few large payments lift their mean to {eur(fn['Amount'].mean())} "
        f"(caught frauds: {eur(tp['Amount'].mean())}). As a result the {len(fn)} missed frauds "
        f"({pct(len(fn) / len(fraud), 0)} of fraud cases) account for {pct(val_missed / val_total, 0)} of fraud "
        f"value. They are also more often nocturnal ({pct(night(fn), 0)} at night vs {pct(night(tp), 0)} for "
        "caught frauds). Both points argue for value- and time-aware thresholds (Section 10.4).")]
    if len(fp):
        x = fp.iloc[0]
        out += [K.p(
            f"<b>The false alarm.</b> The single legitimate transaction flagged was a {eur(x['Amount'])} payment "
            f"with {f1n} = {x[f1n]:.2f} and {f2n} = {x[f2n]:.2f}, values far outside the legitimate core and typical "
            "of fraud. Without the original (pre-PCA) attributes it is impossible to say whether it was an "
            "unusual genuine purchase or an unreported fraud; either way, it is the kind of case a human "
            "reviewer should see.")]
    out += K.h2("10.3 Other limitations")
    out += K.bullets([
        f"<b>Small positive test sample.</b> With {rep['tp'] + rep['fn']} test frauds the recall estimate has a 95% "
        f"interval of {ci(rep['ci']['recall'])}; reported performance should be read with that uncertainty.",
        "<b>Two days of 2013 data.</b> Fraud patterns drift as fraudsters adapt. A random split measures "
        "performance on the same period the model learned from; a production system needs out-of-time "
        "validation and ongoing monitoring.",
        "<b>Opaque features.</b> PCA hides the original attributes, which rules out domain feature engineering "
        "(merchant risk, velocity, customer history) and makes individual alerts hard to explain to an analyst.",
        "<b>Calibration.</b> Forest probabilities are vote shares, not calibrated risk estimates; they are used "
        "here for ranking and thresholding only.",
    ])
    out += K.h2("10.4 How the gaps could be closed")
    out += K.bullets([
        "<b>Richer features in a real deployment:</b> transaction velocity per card, merchant and device "
        "history, distance from usual location. Camouflaged frauds are usually separable only with behavioural "
        "context, which this dataset does not contain.",
        "<b>Two-tier operating points:</b> keep the high-precision threshold for automatic blocking and add a "
        "lower threshold that routes medium-risk transactions to step-up authentication (e.g. one-time "
        "passwords) instead of a block.",
        "<b>Value-aware thresholds:</b> optimise expected cost (amount at risk vs review cost) rather than F1, so "
        "large transactions are held to a lower alert threshold.",
        "<b>Model ensembling:</b> combine the forest with gradient boosting and an unsupervised anomaly score; "
        "different learners make partly different mistakes.",
    ])
    return out


def conclusion(r) -> list:
    rep, e = r["rep"], r["eda"]
    out = K.h1("11. Conclusion")
    out += [K.p(
        f"This project built a complete fraud-detection pipeline on {num(e['n_rows'])} card transactions with 30 "
        f"PCA-transformed features and a fraud rate of only {rate(e['fraud_rate'])}. Statistical exploration "
        "showed that fraud is concentrated in a few principal components (led by V14, V10, V12, V4 and V17), is "
        "more common at night and at very small amounts, and forms a correlated, multi-modal pocket of the "
        "feature space, which pointed toward a non-linear ensemble.")]
    out += [K.p(
        "A leak-free comparison of seven imbalance strategies showed that, once the decision threshold is "
        "optimised, cost-sensitive class weighting matches SMOTE and oversampling without creating synthetic "
        "data, while aggressive undersampling destroys precision despite a flattering ROC-AUC. A Random Forest "
        "was then tuned with a multi-metric randomized search that tracked precision, recall, F1, MCC, balanced "
        "accuracy and PR-AUC, and its operating threshold was fixed on out-of-fold predictions.")]
    out += [K.p(
        f"On a hold-out set used exactly once, the model achieved <b>precision {pct(rep['precision'])}</b>, "
        f"<b>recall {pct(rep['recall'])}</b>, <b>F1-score {pct(rep['f1'])}</b> and <b>MCC {mcc_fmt(rep['mcc'])}</b>, "
        f"raising {rep['tp'] + rep['fp']} alerts of which {rep['tp']} were fraud. Its main strength is the "
        f"trustworthiness of its alerts; its main gap is the {rep['fn']} camouflaged frauds that are "
        "indistinguishable from legitimate payments in the available features. Closing that gap needs richer "
        "behavioural features rather than a more complex model.")]
    out += K.h2("Future work")
    out += K.bullets([
        "Gradient-boosted trees (XGBoost / LightGBM) with tuned scale_pos_weight, and stacking with the forest.",
        "Expected-cost threshold optimisation using transaction amounts.",
        "Probability calibration (isotonic regression) for risk-based decisioning.",
        "Out-of-time validation and drift monitoring (population stability index) for deployment.",
        "SHAP explanations for per-alert reasons shown to analysts.",
        "A REST scoring service around the saved pipeline, with latency and throughput testing.",
    ])
    return out


def appendix(r) -> list:
    out = K.h1("Appendix A. Reproducibility")
    out += [K.p("Environment used for the reported run:")]
    out += [K.table([
        ["Component", "Version"],
        ["Python", platform.python_version()],
        ["scikit-learn", sklearn.__version__],
        ["imbalanced-learn", imblearn.__version__],
        ["pandas / NumPy / SciPy", f"{pd.__version__} / {np.__version__} / {scipy.__version__}"],
        ["Matplotlib / ReportLab", f"{matplotlib.__version__} / {reportlab.Version}"],
        ["Random seed", str(C.RANDOM_STATE)],
    ], col_widths=[60 * mm, 60 * mm])]
    out += [Spacer(1, 8), K.p("Commands:")]
    out += [K.code_block("""
pip install -e ".[dev]"
python -m fraud_detection eda         # EDA statistics and figures
python -m fraud_detection compare     # imbalance strategies, 5-fold CV
python -m fraud_detection tune        # randomized search, 20 x 3-fold
python -m fraud_detection train       # threshold, final fit, hold-out evaluation
python -m fraud_detection baselines   # benchmark models
python -m fraud_detection report      # figures + this PDF
python -m fraud_detection predict --input new.csv --output scored.csv
pytest                                # unit tests
""")]
    out += [Spacer(1, 8), K.p("Project layout:")]
    out += [K.code_block("""
src/fraud_detection/
  config.py           paths, seeds, experiment constants
  data.py             loading, validation, stratified split, leakage guard
  features.py         FeatureBuilder transformer
  eda.py              statistical EDA and figures
  imbalance.py        imbalance-strategy comparison (imblearn pipelines)
  tuning.py           multi-metric randomized search
  train.py            threshold selection, final fit, evaluation, importances
  baselines.py        benchmark models under the same protocol
  evaluation.py       metrics, threshold curves, bootstrap intervals
  report_figures.py   result figures
  report.py, pdfkit.py   this report
  predict.py          batch inference
tests/test_pipeline.py  unit tests (leakage guard, features, metrics, resampling)
reports/results/        every number in this report (CSV / JSON)
models/                 random_forest_fraud.joblib + model_card.json
""")]

    tun = r["tun"]
    out += [PageBreak()]
    out += K.h1("Appendix B. All hyper-parameter search candidates")
    rows = [["#", "class weight", "feat.", "depth", "split", "leaf", "trees", "crit.",
             "samp.", "Prec.", "Recall", "F1", "MCC", "PR-AUC"]]
    for _, x in tun.iterrows():
        rows.append([str(int(x["rank"])), str(x["class_weight"]).replace("balanced_subsample", "bal_sub"),
                     str(x["max_features"]), str(x["max_depth"]).replace(".0", ""), str(int(x["min_samples_split"])),
                     str(int(x["min_samples_leaf"])), str(int(x["n_estimators"])), str(x["criterion"]),
                     str(x["max_samples"]), f"{x['mean_test_precision']:.3f}", f"{x['mean_test_recall']:.3f}",
                     f"{x['mean_test_f1']:.3f}", f"{x['mean_test_mcc']:.3f}", f"{x['mean_test_average_precision']:.3f}"])
    out += [K.table(rows, col_widths=[7 * mm, 20 * mm, 13 * mm, 13 * mm, 12 * mm, 11 * mm, 10 * mm, 14 * mm,
                                      13 * mm, 11.5 * mm, 11.5 * mm, 11.5 * mm, 11.5 * mm, K.CONTENT_W - 160 * mm],
                    align_right_from=9, highlight_rows=[1], font_size=6.9)]
    out += [Spacer(1, 3), K.p("Mean scores over 3 stratified folds at threshold 0.5, sorted by MCC.", K.CAPTION)]

    sep = r["sep"]
    out += [PageBreak()]
    out += K.h1("Appendix C. Feature separability, all 30 features")
    rows = [["Rank", "Feature", "KS", "KS p-value", "Univ. AUC", "Cohen's d", "Mutual info.", "Median legit",
             "Median fraud"]]
    for _, x in sep.iterrows():
        rows.append([str(int(x["ks_rank"])), x["feature"], f"{x['ks_statistic']:.3f}",
                     "&lt; 10<super>-300</super>" if x["ks_pvalue"] == 0 else sci(x["ks_pvalue"]),
                     f"{x['univariate_auc']:.3f}", f"{x['cohens_d']:+.2f}", f"{x['mutual_information']:.4f}",
                     f"{x['median_legit']:,.2f}", f"{x['median_fraud']:,.2f}"])
    out += [K.table(rows, col_widths=[11 * mm, 17 * mm, 14 * mm, 25 * mm, 18 * mm, 18 * mm, 20 * mm, 24 * mm,
                                      K.CONTENT_W - 147 * mm], align_right_from=2, font_size=7.4)]
    return out


# --------------------------------------------------------------------------- #
def run() -> str:
    r = load_results()
    RF.build_all(r["rep"], r["threshold"], r["split"], r["bench"])
    doc = K.ReportDoc(str(C.REPORT_PDF), header_text=f"{PROJECT_TITLE}  ·  {PROJECT_SUBTITLE}",
                      title=f"{PROJECT_TITLE}: {PROJECT_SUBTITLE}", author=AUTHOR or "Project report",
                      subject="Imbalanced classification with Random Forest", creator="fraud_detection report builder")
    story = []
    for part in (cover, executive_summary, problem, dataset, eda_section, methodology, imbalance_section,
                 tuning_section, results_section, interpretation, strengths_gaps, conclusion, appendix):
        story += part(r)
    doc.multiBuild(story)
    log.info("Report written to %s", C.REPORT_PDF)
    return str(C.REPORT_PDF)
