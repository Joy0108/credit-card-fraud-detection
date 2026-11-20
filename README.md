# Credit Card Fraud Detection: Imbalanced Learning on PCA-Transformed Transactions

End-to-end fraud-detection project on the public ULB credit-card dataset: 284,807 transactions,
30 PCA-transformed input features, and only 492 frauds (0.172 %). The project covers statistical EDA,
leakage-safe preprocessing, a cross-validated comparison of class-imbalance strategies, a
multi-metric hyper-parameter search for a Random Forest, threshold optimisation, hold-out
evaluation with bootstrap confidence intervals, benchmark models, interpretation and error analysis.
A full write-up is generated as a PDF.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -e ".[dev]"

python -m fraud_detection all     # eda -> compare -> tune -> train -> baselines -> report
pytest                            # unit tests (synthetic data, ~15 s)
```

The dataset downloads automatically on first run (see `data/README.md`).
The full pipeline takes about 1.5 hours on a 16-thread laptop, most of it in the hyper-parameter search.

| Stage | Command | What it does | Outputs |
|---|---|---|---|
| EDA | `python -m fraud_detection eda` | KS / Mann-Whitney / Cohen's d / mutual information per feature, PCA checks, amount and time analysis | `reports/results/eda_summary.json`, `feature_separability.csv`, `eda_*.png` |
| Imbalance comparison | `python -m fraud_detection compare` | 8 strategies (none, class weights, under/over-sampling, SMOTE, SMOTE-Tomek, Balanced RF, feature ablation), 5-fold CV, resampling inside folds only | `imbalance_comparison.csv`, `imbalance_oof.npz` |
| Tuning | `python -m fraud_detection tune` | `RandomizedSearchCV`, 20 candidates x 3 folds, scored on precision, recall, F1, MCC, balanced accuracy, PR-AUC; selection by MCC | `tuning_results.csv`, `tuning_best.json` |
| Final model | `python -m fraud_detection train` | out-of-fold threshold selection, refit, single hold-out evaluation, bootstrap CIs, MDI + permutation importance | `models/random_forest_fraud.joblib`, `final_model_results.json` |
| Benchmarks | `python -m fraud_detection baselines` | Logistic Regression, Decision Tree, Hist. Gradient Boosting, Isolation Forest, majority class | `baseline_comparison.csv` |
| Report | `python -m fraud_detection report` | figures + PDF | `reports/Credit_Card_Fraud_Detection_Report.pdf` |
| Scoring | `python -m fraud_detection predict --input new.csv` | batch inference with the saved model and threshold | `new_scored.csv` |

## Project layout

```
src/fraud_detection/
    config.py          paths, seeds, experiment constants
    data.py            loading, schema validation, stratified split, training-side leakage guard
    features.py        FeatureBuilder transformer (raw / hour-of-day / log-amount / cyclic)
    eda.py             statistical EDA + figures
    imbalance.py       imbalance-strategy comparison (imblearn pipelines, OOF CV)
    tuning.py          multi-metric randomized search for the Random Forest
    train.py           threshold selection, final fit, hold-out evaluation, importances
    baselines.py       benchmark models under the same protocol
    evaluation.py      confusion-matrix metrics, MCC, threshold curves, bootstrap CIs
    report_figures.py  result figures
    report.py          PDF report builder (ReportLab)
    pdfkit.py          PDF layout helpers
    predict.py         batch inference
    cli.py             `python -m fraud_detection <stage>`
tests/                 unit tests
reports/               figures, results, PDF report
models/                trained model + model card
```

## Methodology in brief

* **Leakage control.** Stratified 80/20 split first; the test set is never used for any choice.
  Inside the training set, exact duplicate rows and rows identical to a test transaction are removed.
* **Imbalance.** Resampling (when used) happens only inside training folds via `imblearn.Pipeline`.
  The selected approach is cost-sensitive class weighting, tuned as a hyper-parameter, plus a
  decision threshold chosen on out-of-fold predictions.
* **Metrics.** Precision, recall, F1 and MCC at the operating threshold, plus PR-AUC. Accuracy is
  reported only to show why it is misleading (99.83 % for a model that never flags fraud).
