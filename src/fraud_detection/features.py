"""Feature engineering as scikit-learn compatible transformers.

Keeping feature logic inside transformers means it is fitted only on the
training folds during cross-validation and is serialised together with the
model, so inference applies exactly the same transformation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C


def hour_of_day(time_seconds: pd.Series | np.ndarray) -> np.ndarray:
    """`Time` is seconds since the first transaction; fold it onto a 24 h clock."""
    return (np.asarray(time_seconds) % C.SECONDS_PER_DAY) / 3600.0


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Build the model input matrix from the 30 raw columns.

    feature_set
        ``"raw"``        - Time, V1..V28, Amount (the 30 original features).
        ``"engineered"`` - V1..V28, log1p(Amount), Hour-of-day (still 30
                           features). ``Time`` counts seconds across a 2-day
                           window, so a split on it cannot transfer to a new
                           day; hour-of-day is the reusable part of it.
        ``"cyclic"``     - like ``engineered`` but the hour is encoded as
                           sin/cos so linear models see 23:00 next to 00:00.
    """

    def __init__(self, feature_set: str = "raw"):
        self.feature_set = feature_set

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        if self.feature_set not in {"raw", "engineered", "cyclic"}:
            raise ValueError(f"Unknown feature_set {self.feature_set!r}")
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = np.array(self._names())
        return self

    def _names(self) -> list[str]:
        if self.feature_set == "raw":
            return list(C.RAW_FEATURES)
        if self.feature_set == "engineered":
            return [*C.PCA_FEATURES, "LogAmount", "Hour"]
        return [*C.PCA_FEATURES, "LogAmount", "HourSin", "HourCos"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=C.RAW_FEATURES) if not isinstance(X, pd.DataFrame) else X
        if self.feature_set == "raw":
            return X[C.RAW_FEATURES].copy()
        out = X[C.PCA_FEATURES].copy()
        out["LogAmount"] = np.log1p(X["Amount"].to_numpy())
        hour = hour_of_day(X["Time"])
        if self.feature_set == "engineered":
            out["Hour"] = hour
        else:
            angle = 2 * np.pi * hour / 24.0
            out["HourSin"] = np.sin(angle)
            out["HourCos"] = np.cos(angle)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self._names())
