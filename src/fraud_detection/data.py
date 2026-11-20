"""Data loading, validation, splitting and leakage control."""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading & validation
# --------------------------------------------------------------------------- #
def download_dataset(path: Path = C.DATA_PATH, url: str = C.DATA_URL) -> Path:
    """Fetch the public ULB credit-card dataset if it is not already on disk."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset from %s", url)
    urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed public URL
    return path


def validate(df: pd.DataFrame) -> None:
    """Fail fast if the file does not look like the expected dataset."""
    missing = set(C.EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if df[C.EXPECTED_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not set(df[C.TARGET].unique()) <= {0, 1}:
        raise ValueError("Target column must be binary {0, 1}")
    if (df["Amount"] < 0).any():
        raise ValueError("Negative transaction amounts found")


def load_raw(path: Path = C.DATA_PATH) -> pd.DataFrame:
    """Load the CSV, validate it and downcast the target to int8."""
    download_dataset(path)
    df = pd.read_csv(path)
    validate(df)
    df[C.TARGET] = df[C.TARGET].astype("int8")
    return df[C.EXPECTED_COLUMNS]


# --------------------------------------------------------------------------- #
# Splitting with leakage control
# --------------------------------------------------------------------------- #
@dataclass
class SplitSummary:
    n_total: int
    n_train_raw: int
    n_test: int
    fraud_train_raw: int
    fraud_test: int
    train_internal_duplicates: int
    train_rows_matching_test: int
    n_train: int
    fraud_train: int

    def to_dict(self) -> dict:
        return asdict(self)


def _row_hash(df: pd.DataFrame) -> pd.Series:
    """Stable hash of the feature values of each row (target excluded)."""
    return pd.util.hash_pandas_object(df[C.RAW_FEATURES], index=False)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = C.TEST_SIZE,
    random_state: int = C.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Stratified hold-out split followed by a training-side leakage guard.

    The test set is left exactly as sampled so that it mirrors the production
    distribution (duplicates included). The *training* set is then cleaned:

    1. exact duplicate rows inside the training set are dropped, so repeated
       transactions do not get extra weight in the forest; and
    2. any training row whose 30 feature values are identical to a test row is
       removed, so the model can never score a test transaction it has
       literally memorised.
    """
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[C.TARGET], random_state=random_state
    )
    n_train_raw, fraud_train_raw = len(train), int(train[C.TARGET].sum())

    dedup = train.drop_duplicates()
    internal_dups = n_train_raw - len(dedup)

    test_hashes = set(_row_hash(test))
    overlap_mask = _row_hash(dedup).isin(test_hashes).to_numpy()
    clean = dedup.loc[~overlap_mask]

    summary = SplitSummary(
        n_total=len(df),
        n_train_raw=n_train_raw,
        n_test=len(test),
        fraud_train_raw=fraud_train_raw,
        fraud_test=int(test[C.TARGET].sum()),
        train_internal_duplicates=int(internal_dups),
        train_rows_matching_test=int(overlap_mask.sum()),
        n_train=len(clean),
        fraud_train=int(clean[C.TARGET].sum()),
    )
    log.info("Split summary: %s", summary)
    return clean.reset_index(drop=True), test.reset_index(drop=True), summary


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the 30 raw input features and the target vector."""
    return df[C.RAW_FEATURES], df[C.TARGET].to_numpy()
