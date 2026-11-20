"""Exploratory data analysis: statistics + figures, written to reports/.

Run with ``python -m fraud_detection eda``. Produces
``reports/results/eda_summary.json``, ``reports/results/feature_separability.csv``
and the ``eda_*.png`` figures.
"""
from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from . import config as C
from . import plotting as P
from .data import load_raw
from .features import hour_of_day

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def feature_separability(df: pd.DataFrame) -> pd.DataFrame:
    """Univariate evidence of how well each feature separates fraud from legit."""
    y = df[C.TARGET].to_numpy()
    fraud, legit = df[y == 1], df[y == 0]
    mi = mutual_info_classif(df[C.RAW_FEATURES], y, random_state=C.RANDOM_STATE, n_neighbors=3)
    rows = []
    for i, col in enumerate(C.RAW_FEATURES):
        f, l = fraud[col].to_numpy(), legit[col].to_numpy()
        ks = stats.ks_2samp(f, l)
        mw = stats.mannwhitneyu(f, l, alternative="two-sided")
        rows.append({
            "feature": col,
            "ks_statistic": ks.statistic,
            "ks_pvalue": ks.pvalue,
            "mannwhitney_pvalue": mw.pvalue,
            # AUC of the single feature used as a score (direction-free)
            "univariate_auc": max(mw.statistic / (len(f) * len(l)), 1 - mw.statistic / (len(f) * len(l))),
            "cohens_d": _cohens_d(f, l),
            "point_biserial_r": stats.pointbiserialr(y, df[col].to_numpy()).statistic,
            "mutual_information": mi[i],
            "mean_legit": l.mean(), "mean_fraud": f.mean(),
            "median_legit": np.median(l), "median_fraud": np.median(f),
        })
    out = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).reset_index(drop=True)
    out["ks_rank"] = np.arange(1, len(out) + 1)
    return out


def summarise(df: pd.DataFrame, sep: pd.DataFrame) -> dict:
    y = df[C.TARGET]
    fraud, legit = df[y == 1], df[y == 0]
    dup_mask = df.duplicated()
    hours = hour_of_day(df["Time"])
    hour_bin = np.floor(hours).astype(int)
    by_hour = df.groupby(hour_bin)[C.TARGET].agg(["size", "sum", "mean"])
    night = (hour_bin < 6)

    v = df[C.PCA_FEATURES]
    corr_all = v.corr().to_numpy()
    corr_fraud = fraud[C.PCA_FEATURES].corr().to_numpy()
    off = ~np.eye(len(C.PCA_FEATURES), dtype=bool)
    stds = v.std().to_numpy()

    amt_mw = stats.mannwhitneyu(fraud["Amount"], legit["Amount"], alternative="two-sided")
    buckets = pd.cut(df["Amount"], [-0.01, 0, 1, 10, 50, 100, 500, 1000, np.inf],
                     labels=["0", "0-1", "1-10", "10-50", "50-100", "100-500", "500-1K", ">1K"])
    bucket_tbl = df.groupby(buckets, observed=True)[C.TARGET].agg(["size", "sum", "mean"])

    top = sep.head(10)
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_features": len(C.RAW_FEATURES),
        "n_pca_features": len(C.PCA_FEATURES),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(dup_mask.sum()),
        "duplicate_fraud_rows": int(df.loc[dup_mask, C.TARGET].sum()),
        "n_fraud": int(y.sum()),
        "n_legit": int((y == 0).sum()),
        "fraud_rate": float(y.mean()),
        "imbalance_ratio": float((y == 0).sum() / y.sum()),
        "trivial_accuracy": float(1 - y.mean()),
        "time_span_hours": float(df["Time"].max() / 3600),
        "amount": {
            "legit": legit["Amount"].describe().to_dict(),
            "fraud": fraud["Amount"].describe().to_dict(),
            "median_legit": float(legit["Amount"].median()),
            "median_fraud": float(fraud["Amount"].median()),
            "zero_amount_share_fraud": float((fraud["Amount"] == 0).mean()),
            "zero_amount_share_legit": float((legit["Amount"] == 0).mean()),
            "fraud_under_10_share": float((fraud["Amount"] < 10).mean()),
            "legit_under_10_share": float((legit["Amount"] < 10).mean()),
            "total_fraud_amount": float(fraud["Amount"].sum()),
            "total_amount": float(df["Amount"].sum()),
            "mannwhitney_pvalue": float(amt_mw.pvalue),
            "fraud_rate_by_bucket": {str(k): {"n": int(r["size"]), "fraud": int(r["sum"]), "rate": float(r["mean"])}
                                     for k, r in bucket_tbl.iterrows()},
        },
        "time": {
            "fraud_rate_by_hour": {int(h): float(r["mean"]) for h, r in by_hour.iterrows()},
            "volume_by_hour": {int(h): int(r["size"]) for h, r in by_hour.iterrows()},
            "peak_fraud_rate_hour": int(by_hour["mean"].idxmax()),
            "peak_fraud_rate": float(by_hour["mean"].max()),
            "night_fraud_rate": float(y[night].mean()),
            "day_fraud_rate": float(y[~night].mean()),
            "night_volume_share": float(night.mean()),
            "night_fraud_share": float(night[y == 1].mean()),
        },
        "pca": {
            "max_abs_mean": float(np.abs(v.mean()).max()),
            "std_v1": float(stds[0]), "std_v28": float(stds[-1]),
            "std_monotonic_spearman": float(stats.spearmanr(np.arange(len(stds)), stds).statistic),
            "max_abs_corr_all": float(np.abs(corr_all[off]).max()),
            "mean_abs_corr_all": float(np.abs(corr_all[off]).mean()),
            "max_abs_corr_fraud": float(np.abs(corr_fraud[off]).max()),
            "mean_abs_corr_fraud": float(np.abs(corr_fraud[off]).mean()),
            "corr_amount_v2": float(df["Amount"].corr(df["V2"])),
            "corr_amount_v7": float(df["Amount"].corr(df["V7"])),
            "corr_time_v3": float(df["Time"].corr(df["V3"])),
        },
        "separability_top10": top[["feature", "ks_statistic", "univariate_auc", "cohens_d",
                                   "mutual_information", "point_biserial_r"]].to_dict("records"),
        "n_features_ks_significant": int((sep["ks_pvalue"] < 0.001).sum()),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_class_balance(df: pd.DataFrame) -> None:
    counts = df[C.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ys = [1, 0]
    for yy, cls in zip(ys, (0, 1)):
        ax.barh(yy, counts[cls], height=0.42, color=P.CLASS_COLORS[cls])
        share = counts[cls] / counts.sum()
        share_txt = f"{np.floor(share * 100_000) / 1000:.3f}%"      # truncated: 0.172 %, 99.827 %
        ax.text(counts[cls] + counts.sum() * 0.01, yy, f"{counts[cls]:,}  ({share_txt})",
                va="center", ha="left", color=P.INK, fontsize=9.5)
    ax.set_yticks(ys, [P.CLASS_NAMES[0], P.CLASS_NAMES[1]])
    ax.set_xlim(0, counts.sum() * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(P.thousands))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Number of transactions")
    P.save(fig, "eda_class_balance")


def fig_amount(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    bins = np.linspace(0, np.log1p(df["Amount"].max()), 60)
    for cls in (0, 1):
        vals = np.log1p(df.loc[df[C.TARGET] == cls, "Amount"])
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=2,
                color=P.CLASS_COLORS[cls], label=P.CLASS_NAMES[cls])
    ticks = [0, 1, 10, 100, 1000, 10000]
    ax.set_xticks(np.log1p(ticks), [f"{t:,}" for t in ticks])
    ax.set_xlabel("Transaction amount (EUR, log scale)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    P.titled(ax, "Amount distribution by class", "Fraud is concentrated at very small and mid-size amounts")

    ax = axes[1]
    buckets = pd.cut(df["Amount"], [-0.01, 0, 1, 10, 50, 100, 500, 1000, np.inf],
                     labels=["0", "0-1", "1-10", "10-50", "50-100", "100-500", "500-1K", ">1K"])
    rate = df.groupby(buckets, observed=True)[C.TARGET].mean() * 100
    x = np.arange(len(rate))
    ax.bar(x, rate.to_numpy(), width=0.6, color=P.FRAUD)
    ax.axhline(df[C.TARGET].mean() * 100, color=P.INK_2, lw=1)
    ax.text(2.5, df[C.TARGET].mean() * 100 + 0.02, "overall 0.172%", ha="center", va="bottom",
            fontsize=8.4, color=P.INK_2)
    ax.set_xticks(x, rate.index.astype(str), rotation=0, fontsize=8.4)
    ax.set_xlabel("Amount bucket (EUR)")
    ax.set_ylabel("Fraud rate (%)")
    ax.grid(axis="x", visible=False)
    P.titled(ax, "Fraud rate by amount bucket", "Share of transactions in each bucket that are fraud")
    fig.tight_layout(w_pad=3)
    P.save(fig, "eda_amount")


def fig_time(df: pd.DataFrame) -> None:
    hours = np.floor(hour_of_day(df["Time"])).astype(int)
    grp = df.groupby(hours)[C.TARGET].agg(["size", "mean"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.bar(grp.index, grp["size"], width=0.7, color=P.LEGIT)
    ax.yaxis.set_major_formatter(FuncFormatter(P.thousands))
    ax.set_xlabel("Hour of day (derived from Time)")
    ax.set_ylabel("Transactions")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="x", visible=False)
    P.titled(ax, "Transaction volume by hour", "Both days pooled; activity collapses overnight")

    ax = axes[1]
    ax.bar(grp.index, grp["mean"] * 100, width=0.7, color=P.FRAUD)
    ax.axhline(df[C.TARGET].mean() * 100, color=P.INK_2, lw=1)
    ax.text(23.4, df[C.TARGET].mean() * 100, "overall", ha="right", va="bottom", fontsize=8.4, color=P.INK_2)
    ax.set_xlabel("Hour of day (derived from Time)")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="x", visible=False)
    P.titled(ax, "Fraud rate by hour", "Night hours carry a far higher share of fraud")
    fig.tight_layout(w_pad=3)
    P.save(fig, "eda_time")


def fig_pca_variance(df: pd.DataFrame) -> None:
    stds = df[C.PCA_FEATURES].std()
    fig, ax = plt.subplots(figsize=(10, 2.9))
    x = np.arange(len(stds))
    ax.bar(x, stds.to_numpy() ** 2, width=0.62, color=P.LEGIT)
    ax.set_xticks(x, [c for c in C.PCA_FEATURES], fontsize=8)
    ax.set_ylabel("Variance")
    ax.grid(axis="x", visible=False)
    P.titled(ax, "Variance of the principal components V1-V28",
             "Variance falls with the component index, as expected from PCA ordering")
    P.save(fig, "eda_pca_variance")


def fig_correlation(df: pd.DataFrame) -> None:
    cols = C.RAW_FEATURES
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))
    for ax, (title, sub, data) in zip(axes, [
        ("All transactions", "PCA makes V1-V28 mutually uncorrelated", df),
        ("Fraud transactions only", "Inside the fraud class the components move together", df[df[C.TARGET] == 1]),
    ]):
        corr = data[cols].corr().to_numpy()
        im = ax.imshow(corr, cmap=P.DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
        idx = [0, 4, 9, 14, 19, 24, 29]
        ax.set_xticks(idx, [cols[i] for i in idx], fontsize=7.5)
        ax.set_yticks(idx, [cols[i] for i in idx], fontsize=7.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        P.titled(ax, title, sub)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=P.INK_2)
    cbar.set_label("Pearson correlation", color=P.INK_2)
    P.save(fig, "eda_correlation")


def fig_separability(sep: pd.DataFrame) -> None:
    s = sep.sort_values("ks_statistic", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
    colors = [P.LEGIT if r <= 10 else "#c3c2b7" for r in s["ks_rank"]]
    ax = axes[0]
    ax.barh(s["feature"], s["ks_statistic"], height=0.62, color=colors)
    ax.set_xlabel("Kolmogorov-Smirnov statistic")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    P.titled(ax, "Distribution shift (KS)", "Top-10 features highlighted")
    ax = axes[1]
    r = s["point_biserial_r"]
    ax.barh(s["feature"], r, height=0.62, color=[P.SERIES[7] if v > 0 else P.SERIES[0] for v in r])
    ax.axvline(0, color=P.AXIS, lw=0.8)
    ax.set_xlabel("Point-biserial correlation with Class")
    ax.grid(axis="y", visible=False)
    P.titled(ax, "Direction of the effect", "Red: higher in fraud  |  Blue: lower in fraud")
    fig.tight_layout(w_pad=2)
    P.save(fig, "eda_separability")


def fig_distributions(df: pd.DataFrame, sep: pd.DataFrame, k: int = 8) -> None:
    top = sep["feature"].head(k).tolist()
    rng = np.random.default_rng(C.RANDOM_STATE)
    legit = df[df[C.TARGET] == 0]
    legit = legit.iloc[rng.choice(len(legit), 60_000, replace=False)]
    fraud = df[df[C.TARGET] == 1]
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 4.8))
    for ax, col in zip(axes.ravel(), top):
        lo = min(np.percentile(fraud[col], 3), np.percentile(legit[col], 0.5))
        hi = max(np.percentile(fraud[col], 97), np.percentile(legit[col], 99.5))
        grid = np.linspace(lo, hi, 300)
        for cls, part in ((0, legit), (1, fraud)):
            vals = part[col][(part[col] >= lo) & (part[col] <= hi)]
            kde = stats.gaussian_kde(vals, bw_method=0.18)
            dens = kde(grid)
            ax.plot(grid, dens, color=P.CLASS_COLORS[cls], lw=2, label=P.CLASS_NAMES[cls])
            ax.fill_between(grid, dens, color=P.CLASS_COLORS[cls], alpha=0.10, lw=0)
        ks = float(sep.loc[sep.feature == col, "ks_statistic"].iloc[0])
        ax.set_title(f"{col}", fontsize=10, pad=4)
        ax.text(1.0, 1.0, f"KS = {ks:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=P.INK_2)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=1.6)
    P.save(fig, "eda_distributions")


def fig_scatter(df: pd.DataFrame, sep: pd.DataFrame) -> None:
    a, b = sep["feature"].iloc[0], sep["feature"].iloc[1]
    rng = np.random.default_rng(C.RANDOM_STATE)
    legit = df[df[C.TARGET] == 0]
    legit = legit.iloc[rng.choice(len(legit), 8_000, replace=False)]
    fraud = df[df[C.TARGET] == 1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(legit[a], legit[b], s=9, color=P.LEGIT, alpha=0.35, lw=0, label="Legitimate (8,000 sample)")
    ax.scatter(fraud[a], fraud[b], s=16, color=P.FRAUD, alpha=0.85, edgecolor=P.SURFACE, lw=0.6,
               label="Fraud (all 492)")
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.legend(loc="lower left", markerscale=1.6)
    P.titled(ax, f"{a} vs {b}", "The two most discriminative components already isolate most fraud")
    P.save(fig, "eda_scatter_top2")


def run() -> dict:
    df = load_raw()
    log.info("Loaded %s rows", len(df))
    sep = feature_separability(df)
    sep.to_csv(C.RESULTS_DIR / "feature_separability.csv", index=False)
    summary = summarise(df, sep)
    (C.RESULTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    fig_class_balance(df)
    fig_amount(df)
    fig_time(df)
    fig_pca_variance(df)
    fig_correlation(df)
    fig_separability(sep)
    fig_distributions(df, sep)
    fig_scatter(df, sep)
    log.info("EDA complete: %d figures", 8)
    return summary
