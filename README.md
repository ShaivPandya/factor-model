# Factor Model
## Quick start

1) Setup

`python3 -V`

`pip install pandas yfinance pandas-datareader statsmodels`

2) Prepare portfolio.csv

3) Run

`python3 factor_report.py portfolio.csv`

4. Optional

`python3 factor_report.py portfolio.csv --lookback 500 --nw_lags 5`

## Design choices

-  Time-series OLS on the portfolio
   -  Objective is portfolio betas, not factor returns. A single regression is sufficient and fast.
    
-  Newey–West standard errors
   -  Daily returns show heteroskedasticity and serial correlation. HAC SEs are standard for inference. The table prints betas only, but the fit uses HAC.
    
-  Fama–French + Momentum
   -  Public, stable, and widely used. The set spans market, size, value, quality-like (RMW), investment (CMA), and momentum.
    
-   Gross-capital weights for portfolio returns
    -  Normalizes by trading capital and preserves the sign of long and short sides. Works for long-short books and cash-neutral books.
    
-  Sample covariance of factor returns
   -  Simple and transparent for a first pass. Users can swap to EWMA or shrinkage if desired.
    
-  30-day sum, not compounding
   -  For daily factor returns at small magnitudes, arithmetic sum approximates compounding closely and keeps the table readable.


## Interpreting the table

-  Beta
	- sensitivity of the portfolio’s excess return to each factor.
    
-  Exposure ($M)
	- position-sized effect of a one-unit factor move.
    
-   % of Book Variance
	- how much each factor contributes to total variance across the sample window.
    
-   Factor Return % (30D)
	- the recent realized return of each factor, independent of your portfolio.
    


## Limitations

-   Data are US-centric. FF daily factors are US. If most holdings trade outside the US or in other currencies, factor mapping may be off.
    
-   Prices from Yahoo Finance can differ from other vendors. Adjusted close includes corporate actions.
    
-   The regression assumes constant betas over the lookback window.
    
-   Currency risk is ignored. Everything is treated in the quote currency of Yahoo prices and FF factors.



## Data sources
-   Yahoo Finance via  `yfinance`  for adjusted close prices.
    
-   Ken French Data Library via  `pandas-datareader`  for daily FF5 and Momentum, including the daily risk-free rate.
