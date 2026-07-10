"""Data layer: download, validate and align multi-asset Indian market data.

All series are inner-joined on trading dates (different holiday calendars).
VIX is kept as a LEVEL, never return-transformed.
"""
import pandas as pd

# Swap tickers here if a Yahoo symbol is dead. Alternates in comments.
TICKERS = {
    "EQUITY": "NIFTYBEES.NS",   # Nifty 50 ETF (alt: ^NSEI index itself)
    "GOLD": "GOLDBEES.NS",      # Gold ETF
    "BOND": "LTGILTBEES.NS",    # Long-term gilt ETF (alt: SETF10GILT.NS)
}
VIX_TICKER = "^INDIAVIX"        # India VIX (level)
INDEX_TICKER = "^NSEI"          # Nifty 50 index: regime-detection signal source
START = "2010-01-01"
MIN_DAYS = 750                  # refuse series shorter than ~3y


def _download(tickers, start):
    import yfinance as yf
    px = yf.download(list(tickers), start=start, auto_adjust=True,
                     progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(list(tickers)[0])
    return px


def load_market_data(start=START):
    """Returns (asset_prices[EQUITY,GOLD,BOND], vix_level, index_price),
    all aligned to the same trading dates."""
    raw = _download(list(TICKERS.values()) + [VIX_TICKER, INDEX_TICKER], start)
    missing = [t for t in list(TICKERS.values()) + [VIX_TICKER, INDEX_TICKER]
               if t not in raw.columns or raw[t].dropna().shape[0] < MIN_DAYS]
    if missing:
        raise RuntimeError(
            f"Tickers failed or too short: {missing}. "
            "Swap symbols in src/data.py TICKERS and rerun.")
    df = raw.dropna()  # inner join across all series
    assets = df[[TICKERS[k] for k in ("EQUITY", "GOLD", "BOND")]].copy()
    assets.columns = ["EQUITY", "GOLD", "BOND"]
    vix = df[VIX_TICKER].rename("VIX")
    index_px = df[INDEX_TICKER].rename("NIFTY")
    assert not assets.isna().any().any(), "NaNs after alignment"
    return assets, vix, index_px
