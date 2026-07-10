"""Gaussian HMM wrappers.

Three design decisions live here, each interview-defensible:
1. Multi-seed fitting: Baum-Welch (EM) converges to a LOCAL optimum, so we
   fit from several random seeds and keep the best log-likelihood.
2. Canonical relabeling: HMM state indices are arbitrary (state "0" means
   nothing). We map states to Bull/Bear/Crisis by sorting on the mean of
   the volatility feature: lowest vol -> Bull(0), mid -> Bear(1),
   highest -> Crisis(2). This also solves the cross-fold relabeling
   problem in walk-forward validation.
3. Causal prediction: labeling day t runs Viterbi on data up to t only,
   never on the full window (predict/predict_proba on a whole window are
   smoothed, i.e. they use the future within that window).
"""
import numpy as np
from hmmlearn.hmm import GaussianHMM

BULL, BEAR, CRISIS = 0, 1, 2
REGIME_NAMES = {BULL: "Bull", BEAR: "Bear", CRISIS: "Crisis"}


def fit_hmm(X: np.ndarray, n_states: int = 3, seeds=(0, 1, 2, 3, 4)):
    """Fit GaussianHMM from several seeds, return the best-scoring model."""
    best, best_ll = None, -np.inf
    for s in seeds:
        m = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=300, tol=1e-4, random_state=s, min_covar=1e-6)
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("HMM failed to fit on all seeds")
    return best


def state_mapping(model, vol_idx: int) -> dict:
    """Map raw state index -> canonical label by ascending mean volatility."""
    order = np.argsort(model.means_[:, vol_idx])
    return {int(raw): canon for canon, raw in enumerate(order)}


def causal_labels(model, mapping: dict, X_hist: np.ndarray,
                  t_start: int) -> np.ndarray:
    """Canonical label for each t in [t_start, len(X_hist)), where the label
    for t is computed from X_hist[: t+1] only (leak-free filtering)."""
    out = np.empty(len(X_hist) - t_start, dtype=int)
    for k, t in enumerate(range(t_start, len(X_hist))):
        raw = model.predict(X_hist[: t + 1])[-1]
        out[k] = mapping[int(raw)]
    return out
