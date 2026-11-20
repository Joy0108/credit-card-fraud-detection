"""Central configuration: paths, seeds and experiment constants.

Every stage of the pipeline imports from here so that a single edit changes
the behaviour of the whole project (and keeps runs reproducible).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_PDF = REPORTS_DIR / "Credit_Card_Fraud_Detection_Report.pdf"

for _d in (FIG_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
TARGET = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", *PCA_FEATURES, "Amount"]          # 30 model inputs
EXPECTED_COLUMNS = [*RAW_FEATURES, TARGET]
SECONDS_PER_DAY = 86_400

# --------------------------------------------------------------------------- #
# Experiment protocol
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20                 # stratified hold-out, touched once at the end
CV_FOLDS = 5                     # imbalance-strategy comparison + threshold OOF
TUNING_FOLDS = 3                 # randomized hyper-parameter search
TUNING_ITERATIONS = 20
COMPARISON_TREES = 100           # trees per forest in the strategy comparison
SMOTE_RATIO = 0.10               # minority : majority after resampling (1:10)
N_BOOTSTRAP = 2_000              # bootstrap resamples for confidence intervals

# Business assumptions used in the cost analysis (EUR, per transaction).
REVIEW_COST_PER_ALERT = 5.0      # analyst time to review one flagged transaction

# Model artefacts
MODEL_PATH = MODELS_DIR / "random_forest_fraud.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
