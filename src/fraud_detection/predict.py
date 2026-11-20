"""Batch inference with the saved model.

    python -m fraud_detection predict --input transactions.csv --output scored.csv

The input needs the 30 raw columns (Time, V1..V28, Amount); a ``Class``
column, if present, is ignored. The output adds ``fraud_probability`` and
``fraud_alert`` (1 when the probability reaches the tuned threshold).
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


def load_model(path: Path = C.MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m fraud_detection train` first")
    return joblib.load(path)


def score_frame(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle or load_model()
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if df[bundle["features"]].isna().any().any():
        raise ValueError("Input contains missing values in model features")
    out = df.copy()
    out["fraud_probability"] = bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1]
    out["fraud_alert"] = (out["fraud_probability"] >= bundle["threshold"]).astype(int)
    return out


def score_file(input_path: str, output_path: str | None) -> Path:
    if not input_path:
        raise SystemExit("--input is required for the predict stage")
    src = Path(input_path)
    dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_scored.csv")
    scored = score_frame(pd.read_csv(src))
    scored.to_csv(dst, index=False)
    log.info("Scored %d rows, %d alerts -> %s", len(scored), int(scored["fraud_alert"].sum()), dst)
    return dst
