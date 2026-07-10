"""Walk-forward validation engine.

Leak-free by construction, per fold:
  - StandardScaler fit on the TRAINING window only.
  - HMM fit on the TRAINING window only (fresh re-fit every fold).
  - Per-regime asset return stats (mu_k, Sigma_k) computed from TRAINING
    days only, using the model's own training-window labels.
  - Test-day labels produced causally (Viterbi on data up to t only).
  - States relabeled to canonical Bull/Bear/Crisis per fold (see hmm_model).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import VOL_COL
from .hmm_model import fit_hmm, state_mapping, causal_labels


@dataclass
class WalkForwardResult:
    labels: pd.Series                 # canonical regime label per test day
    regime_stats: dict = field(default_factory=dict)  # fold_id -> {k:(mu,S)}
    fold_of_day: pd.Series = None     # fold_id per test day
    n_folds: int = 0


def expanding_folds(n: int, initial_train: int, test_size: int):
    start = initial_train
    fold = 0
    while start < n:
        yield fold, start, min(start + test_size, n)
        start += test_size
        fold += 1


def per_regime_asset_stats(asset_ret: pd.DataFrame, labels: np.ndarray,
                           min_days: int = 30):
    """(mu, Sigma) of asset returns per regime from TRAIN days; fall back to
    overall stats when a regime has too few days."""
    overall = (asset_ret.mean().values,
               np.cov(asset_ret.values.T) + 1e-8 * np.eye(asset_ret.shape[1]))
    out = {}
    for k in (0, 1, 2):
        sub = asset_ret.values[labels == k]
        if len(sub) < min_days:
            out[k] = overall
        else:
            out[k] = (sub.mean(axis=0),
                      np.cov(sub.T) + 1e-8 * np.eye(sub.shape[1]))
    return out


def run_walkforward(features: pd.DataFrame, asset_ret: pd.DataFrame,
                    initial_train: int = 750, test_size: int = 63,
                    seeds=(0, 1, 2, 3, 4)) -> WalkForwardResult:
    features, asset_ret = features.align(asset_ret, join="inner", axis=0)
    X = features.values
    vol_idx = list(features.columns).index(VOL_COL)
    n = len(X)

    all_labels, all_folds = [], []
    res = WalkForwardResult(labels=None)
    for fold, t0, t1 in expanding_folds(n, initial_train, test_size):
        scaler = StandardScaler().fit(X[:t0])           # train-only stats
        Xs = scaler.transform(X[:t1])                   # scaled train+test
        model = fit_hmm(Xs[:t0], seeds=seeds)           # train-only fit
        mapping = state_mapping(model, vol_idx)
        train_lbl = np.vectorize(mapping.get)(model.predict(Xs[:t0]))
        res.regime_stats[fold] = per_regime_asset_stats(
            asset_ret.iloc[:t0], train_lbl)
        test_lbl = causal_labels(model, mapping, Xs, t0)  # leak-free
        all_labels.append(pd.Series(test_lbl, index=features.index[t0:t1]))
        all_folds.append(pd.Series(fold, index=features.index[t0:t1]))
        res.n_folds = fold + 1

    res.labels = pd.concat(all_labels)
    res.fold_of_day = pd.concat(all_folds)
    return res
