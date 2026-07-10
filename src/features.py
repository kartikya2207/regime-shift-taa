"""Feature engineering for regime detection.

Every feature at date t uses ONLY data dated <= t (backward-looking rolling
windows). Cross-sectional scaling is deliberately NOT done here: the
StandardScaler is fit on each walk-forward training window only
(see walkforward.py), which is what keeps normalization leak-free.
"""
import numpy as np
import pandas as pd

VOL_COL = "vol_21"  # used to map anonymous HMM states -> Bull/Bear/Crisis


def build_features(index_px: pd.Series, vix: pd.Series) -> pd.DataFrame:
    logret = np.log(index_px / index_px.shift(1))
    feats = pd.DataFrame({
        "ret_1": logret,
        "mom_63": logret.rolling(63).sum(),                    # ~1 quarter
        VOL_COL: logret.rolling(21).std() * np.sqrt(252),      # ~1 month
        "vix_rel": vix / vix.rolling(252).mean() - 1.0,        # VIX vs 1y avg
    })
    return feats.dropna()


def asset_simple_returns(asset_px: pd.DataFrame) -> pd.DataFrame:
    return asset_px.pct_change().dropna()
