# Results

- Data: NSE data via yfinance
- Out-of-sample window: 2023-06-14 to 2026-07-10  |  folds: 12 (expanding, refit per fold)
- Training burn-in (used to fit models, NOT in OOS metrics or plots): 2018-05-11 to 2023-06-14 - pre-OOS stress (e.g. 2022) is training-only
- Transaction cost: 10 bps on turnover  |  annualized turnover: 8.26x
- Regime shares (OOS): Bull 38%, Bear 38%, Crisis 24%

|                    |   AnnRet |   AnnVol |   Sharpe |   Sortino |   MaxDD |   Calmar |
|:-------------------|---------:|---------:|---------:|----------:|--------:|---------:|
| Regime-TAA (net)   |    0.125 |    0.093 |    1.345 |     1.75  |  -0.104 |    1.196 |
| Regime-TAA (gross) |    0.133 |    0.093 |    1.438 |     1.862 |  -0.102 |    1.301 |
| 60/40 static       |    0.094 |    0.077 |    1.219 |     1.782 |  -0.095 |    0.99  |
| Equal-weight       |    0.165 |    0.091 |    1.822 |     2.144 |  -0.12  |    1.378 |

![regimes](regimes.png)
![equity](equity_curves.png)
![drawdown](drawdown.png)
