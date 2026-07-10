"""Run the full pipeline: data -> features -> walk-forward HMM regimes ->
per-regime cvxpy weights -> cost-aware backtest -> plots + RESULTS.md.

Usage:
    python run_pipeline.py                # real data (yfinance)
    python run_pipeline.py --synthetic    # offline validation run
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.features import build_features, asset_simple_returns
from src.walkforward import run_walkforward
from src.portfolio import (weights_from_labels, backtest, static_benchmark,
                           metrics)
from src.hmm_model import REGIME_NAMES

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REGIME_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}


def load(synthetic: bool):
    if synthetic:
        from src.synthetic import generate
        prices, vix, index_px, _ = generate()
        return prices, vix, index_px, "SYNTHETIC 3-regime data"
    from src.data import load_market_data
    prices, vix, index_px = load_market_data()
    return prices, vix, index_px, "NSE data via yfinance"


def plot_regimes(index_px, labels, path):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    px = index_px.loc[labels.index]
    ax.plot(px.index, px.values, color="black", lw=0.8)
    runs = (labels != labels.shift()).cumsum()
    for _, seg in labels.groupby(runs):
        ax.axvspan(seg.index[0], seg.index[-1],
                   color=REGIME_COLORS[int(seg.iloc[0])], alpha=0.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[k], alpha=0.4)
               for k in (0, 1, 2)]
    ax.legend(handles, [REGIME_NAMES[k] for k in (0, 1, 2)], loc="upper left")
    ax.set_title("Detected regimes (out-of-sample, causal labels)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_equity(curves: dict, path, title):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, r in curves.items():
        ax.plot((1 + r).cumprod(), lw=1.1, label=name)
    ax.set_yscale("log"); ax.legend(); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_drawdown(r: pd.Series, path):
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5)
    ax.set_title("Regime-TAA drawdown (net of costs)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(synthetic: bool, cost_bps: float = 10.0):
    os.makedirs(RESULTS, exist_ok=True)
    prices, vix, index_px, src_note = load(synthetic)
    feats = build_features(index_px, vix)
    asset_ret = asset_simple_returns(prices)

    wf = run_walkforward(feats, asset_ret)
    weights = weights_from_labels(wf.labels, wf.fold_of_day,
                                  wf.regime_stats, asset_ret.columns)
    ar = asset_ret.loc[weights.index[0]:]
    strat = backtest(weights, ar, cost_bps)
    b6040 = static_benchmark(ar, {"EQUITY": .6, "BOND": .4}, cost_bps)
    beq = static_benchmark(ar, {c: 1 / 3 for c in ar.columns}, cost_bps)

    rows = {
        "Regime-TAA (net)": metrics(strat["net"]),
        "Regime-TAA (gross)": metrics(strat["gross"]),
        "60/40 static": metrics(b6040["net"]),
        "Equal-weight": metrics(beq["net"]),
    }
    table = pd.DataFrame(rows).T
    ann_turn = strat["turnover"].sum() / (len(strat) / 252)
    counts = wf.labels.value_counts().sort_index()
    regime_share = ", ".join(f"{REGIME_NAMES[k]} {v/len(wf.labels):.0%}"
                             for k, v in counts.items())

    plot_regimes(index_px, wf.labels, f"{RESULTS}/regimes.png")
    plot_equity({"Regime-TAA (net)": strat["net"], "60/40": b6040["net"],
                 "Equal-weight": beq["net"]},
                f"{RESULTS}/equity_curves.png", "Equity curves (net, log scale)")
    plot_drawdown(strat["net"], f"{RESULTS}/drawdown.png")

    md = [
        "# Results", "",
        f"- Data: {src_note}",
        f"- Out-of-sample window: {wf.labels.index[0].date()} to "
        f"{wf.labels.index[-1].date()}  |  folds: {wf.n_folds} "
        f"(expanding, refit per fold)",
        f"- Training burn-in (used to fit models, NOT in OOS metrics or "
        f"plots): {prices.index[0].date()} to {wf.labels.index[0].date()} "
        f"- pre-OOS stress (e.g. 2022) is training-only",
        f"- Transaction cost: {cost_bps:.0f} bps on turnover  |  "
        f"annualized turnover: {ann_turn:.2f}x",
        f"- Regime shares (OOS): {regime_share}", "",
        table.round(3).to_markdown(), "",
        "![regimes](regimes.png)", "![equity](equity_curves.png)",
        "![drawdown](drawdown.png)", "",
    ]
    if synthetic:
        md.insert(1, "\n> **SYNTHETIC-DATA VALIDATION RUN** - numbers "
                     "validate the pipeline mechanics, not a real strategy.\n")
    with open(f"{RESULTS}/RESULTS.md", "w") as f:
        f.write("\n".join(md))
    print(table.round(3).to_string())
    print(f"\nWrote results/ (RESULTS.md + 3 plots). Folds: {wf.n_folds}. "
          f"Turnover: {ann_turn:.2f}x/yr")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()
    main(a.synthetic, a.cost_bps)
