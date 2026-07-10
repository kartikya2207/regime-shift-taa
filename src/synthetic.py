"""Synthetic 3-regime market generator.

Simulates a sticky Markov regime chain and per-regime asset dynamics so the
full pipeline can be validated offline (and regime-recovery accuracy can be
measured against the KNOWN true states). Canonical regime order matches the
pipeline: 0=Bull (low vol), 1=Bear, 2=Crisis (high vol).
"""
import numpy as np
import pandas as pd

TRANSMAT = np.array([
    [0.97, 0.02, 0.01],
    [0.03, 0.94, 0.03],
    [0.02, 0.08, 0.90],
])
# per-regime daily (mu, vol) for EQUITY, GOLD, BOND
PARAMS = {
    0: {"EQUITY": (8e-4, 0.008), "GOLD": (2e-4, 0.007), "BOND": (2.5e-4, 0.003)},
    1: {"EQUITY": (-3e-4, 0.015), "GOLD": (3e-4, 0.009), "BOND": (2e-4, 0.004)},
    2: {"EQUITY": (-25e-4, 0.035), "GOLD": (8e-4, 0.014), "BOND": (1e-4, 0.006)},
}


def generate(n_days=2500, seed=7):
    rng = np.random.default_rng(seed)
    states = np.empty(n_days, dtype=int)
    states[0] = 0
    for t in range(1, n_days):
        states[t] = rng.choice(3, p=TRANSMAT[states[t - 1]])

    idx = pd.bdate_range("2015-01-01", periods=n_days)
    rets = {a: np.array([rng.normal(*PARAMS[s][a]) for s in states])
            for a in ("EQUITY", "GOLD", "BOND")}
    prices = pd.DataFrame({a: 100 * np.exp(np.cumsum(r))
                           for a, r in rets.items()}, index=idx)
    # VIX level tracks regime vol with noise
    base = np.array([12.0, 20.0, 42.0])
    vix = pd.Series(base[states] * np.exp(rng.normal(0, 0.08, n_days)),
                    index=idx, name="VIX")
    index_px = prices["EQUITY"].rename("NIFTY")
    true_states = pd.Series(states, index=idx, name="true_regime")
    return prices, vix, index_px, true_states
