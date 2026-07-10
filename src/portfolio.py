"""Per-regime convex optimization, cost-aware backtest, and metrics.

Per-regime objectives (all long-only, fully invested, per-asset cap):
  Bull   : maximize  mu'w - gamma_bull  * w'Sigma w   (return-seeking)
  Bear   : maximize  mu'w - gamma_bear  * w'Sigma w   (balanced)
  Crisis : minimize  w'Sigma w                        (pure min-variance)
gamma is a risk-aversion knob; higher = more variance-averse.

Backtest conventions:
  - Weights decided with information up to close of day t are applied to
    returns of day t+1 (weights.shift(1)) -> no same-day lookahead.
  - Transaction cost = cost_bps * sum(|delta w|), charged when targets move.
  - Weight drift between rebalances is ignored for ALL strategies
    (consistent simplification, documented in README).
"""
import numpy as np
import pandas as pd
import cvxpy as cp

GAMMA = {0: 2.0, 1: 10.0}  # Bull, Bear; Crisis handled as pure min-var
W_CAP = 0.70
TRADING_DAYS = 252


def solve_weights(mu: np.ndarray, Sigma: np.ndarray, regime: int,
                  w_cap: float = W_CAP) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, cp.psd_wrap(Sigma))
    cons = [cp.sum(w) == 1, w >= 0, w <= w_cap]
    if regime == 2:                       # Crisis: min variance
        prob = cp.Problem(cp.Minimize(risk), cons)
    else:
        prob = cp.Problem(cp.Maximize(mu @ w - GAMMA[regime] * risk), cons)
    try:
        prob.solve()
        if w.value is None:
            raise ValueError
        wv = np.clip(w.value, 0, None)
        return wv / wv.sum()
    except Exception:
        return np.ones(n) / n             # documented fallback: equal weight


def confirm_labels(labels: pd.Series, confirm_days: int = 2) -> pd.Series:
    """Debounce: only adopt a new regime after it persists `confirm_days`
    consecutive days. Cuts whipsaw turnover from single-day flips."""
    if confirm_days <= 1:
        return labels
    eff, cur, cand, streak = [], int(labels.iloc[0]), None, 0
    for k in labels.astype(int):
        if k == cur:
            cand, streak = None, 0
        elif k == cand:
            streak += 1
            if streak >= confirm_days:
                cur, cand, streak = k, None, 0
        else:
            cand, streak = k, 1
        eff.append(cur)
    return pd.Series(eff, index=labels.index)


def weights_from_labels(labels: pd.Series, fold_of_day: pd.Series,
                        regime_stats: dict, asset_cols,
                        confirm_days: int = 2) -> pd.DataFrame:
    """3 solves per fold (one per regime), then map each day's (debounced)
    label to its fold's solved weights."""
    labels = confirm_labels(labels, confirm_days)
    cache = {}
    rows = []
    for dt in labels.index:
        f, k = int(fold_of_day.loc[dt]), int(labels.loc[dt])
        if (f, k) not in cache:
            mu, S = regime_stats[f][k]
            cache[(f, k)] = solve_weights(mu, S, k)
        rows.append(cache[(f, k)])
    return pd.DataFrame(rows, index=labels.index, columns=list(asset_cols))


def backtest(weights: pd.DataFrame, asset_ret: pd.DataFrame,
             cost_bps: float = 10.0):
    W = weights.reindex(asset_ret.index).ffill().dropna()
    R = asset_ret.loc[W.index]
    W_eff = W.shift(1).dropna()
    R = R.loc[W_eff.index]
    gross = (W_eff * R).sum(axis=1)
    turnover = W_eff.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (cost_bps / 1e4)
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover})


def static_benchmark(asset_ret: pd.DataFrame, w: dict,
                     cost_bps: float = 10.0) -> pd.DataFrame:
    W = pd.DataFrame([[w.get(c, 0.0) for c in asset_ret.columns]],
                     index=[asset_ret.index[0]], columns=asset_ret.columns)
    return backtest(W, asset_ret, cost_bps)


def metrics(returns: pd.Series) -> dict:
    mu, sd = returns.mean(), returns.std()
    ann_ret, ann_vol = mu * TRADING_DAYS, sd * np.sqrt(TRADING_DAYS)
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    eq = (1 + returns).cumprod()
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    return {
        "AnnRet": ann_ret, "AnnVol": ann_vol,
        "Sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "Sortino": ann_ret / downside if downside > 0 else np.nan,
        "MaxDD": maxdd,
        "Calmar": ann_ret / abs(maxdd) if maxdd < 0 else np.nan,
    }
