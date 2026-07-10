"""Tests: lookahead-leak guards, optimizer sanity, end-to-end on synthetic
data with regime-recovery accuracy against the known true states."""
import numpy as np
import pandas as pd
import pytest

from src.features import build_features, asset_simple_returns
from src.synthetic import generate
from src.portfolio import solve_weights, weights_from_labels, backtest, metrics
from src.walkforward import run_walkforward


def test_features_have_no_lookahead():
    """Perturbing FUTURE prices must not change PAST feature rows."""
    prices, vix, index_px, _ = generate(n_days=800, seed=1)
    f1 = build_features(index_px, vix)
    cut = 500
    index_px2 = index_px.copy(); index_px2.iloc[cut:] *= 1.5
    vix2 = vix.copy(); vix2.iloc[cut:] *= 2.0
    f2 = build_features(index_px2, vix2)
    cut_date = index_px.index[cut]
    a = f1.loc[f1.index < cut_date]
    b = f2.loc[f2.index < cut_date]
    pd.testing.assert_frame_equal(a, b)


def test_optimizer_weights_valid():
    rng = np.random.default_rng(0)
    mu = rng.normal(5e-4, 2e-4, 3)
    A = rng.normal(size=(3, 3)); Sigma = A @ A.T * 1e-4 + 1e-6 * np.eye(3)
    for regime in (0, 1, 2):
        w = solve_weights(mu, Sigma, regime)
        assert w.shape == (3,)
        assert abs(w.sum() - 1) < 1e-6
        assert (w >= -1e-9).all() and (w <= 0.70 + 1e-6).all()
    w_crisis = solve_weights(mu, Sigma, 2)
    w_bull = solve_weights(mu, Sigma, 0)
    assert w_crisis @ Sigma @ w_crisis <= w_bull @ Sigma @ w_bull + 1e-12


def test_end_to_end_synthetic():
    prices, vix, index_px, true_states = generate(n_days=1700, seed=7)
    feats = build_features(index_px, vix)
    asset_ret = asset_simple_returns(prices)
    wf = run_walkforward(feats, asset_ret, initial_train=800, test_size=150,
                         seeds=(0, 1, 2))
    assert wf.n_folds >= 4
    assert set(np.unique(wf.labels.values)).issubset({0, 1, 2})

    # regime recovery vs known truth (canonical order = ascending vol)
    truth = true_states.loc[wf.labels.index]
    acc = (wf.labels.values == truth.values).mean()
    assert acc > 0.55, f"regime recovery too weak: {acc:.2f}"

    weights = weights_from_labels(wf.labels, wf.fold_of_day,
                                  wf.regime_stats, asset_ret.columns)
    bt = backtest(weights, asset_ret.loc[weights.index[0]:])
    m = metrics(bt["net"])
    assert np.isfinite(list(m.values())).all()
    assert bt["net"].std() > 0
    # costs must bite: net cumulative <= gross cumulative
    assert (1 + bt["net"]).prod() <= (1 + bt["gross"]).prod() + 1e-12


def test_crisis_deweights_equity_on_synthetic():
    """In synthetic data equity is toxic in Crisis; optimizer should hold
    less equity in Crisis than in Bull."""
    prices, vix, index_px, _ = generate(n_days=1700, seed=7)
    feats = build_features(index_px, vix)
    asset_ret = asset_simple_returns(prices)
    wf = run_walkforward(feats, asset_ret, initial_train=800, test_size=150,
                         seeds=(0, 1, 2))
    weights = weights_from_labels(wf.labels, wf.fold_of_day,
                                  wf.regime_stats, asset_ret.columns)
    eq_bull = weights.loc[wf.labels == 0, "EQUITY"].mean()
    eq_crisis = weights.loc[wf.labels == 2, "EQUITY"].mean()
    if np.isnan(eq_crisis):
        pytest.skip("no crisis days detected in OOS window")
    assert eq_crisis < eq_bull
