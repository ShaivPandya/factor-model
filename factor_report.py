#!/usr/bin/env python3
import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# External deps: yfinance, pandas_datareader, statsmodels
# pip install pandas yfinance pandas-datareader statsmodels

def _as_dataframe(obj):
    """Ensure yfinance download result is a 2D DataFrame with columns of tickers."""
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    return obj

def load_portfolio(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"ticker", "price", "quantity"}
    missing = req - set(map(str.lower, df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Need: ticker, price, quantity")
    # Normalize column names
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    df["value"] = df["price"] * df["quantity"]
    gross = df["value"].abs().sum()
    if gross <= 0:
        raise ValueError("Gross notional is zero.")
    # Signed gross weights for return aggregation on gross capital
    df["w_gross"] = df["value"] / gross
    return df[["ticker", "price", "quantity", "value", "w_gross"]], gross

def fetch_price_returns(tickers, start, end):
    import yfinance as yf
    px = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False)
    if "Adj Close" in px.columns:
        px = px["Adj Close"]
    px = _as_dataframe(px)
    px = px.sort_index()
    rets = px.pct_change().dropna(how="all")
    return rets

def fetch_ff_factors(start, end):
    from pandas_datareader.data import DataReader
    # Daily 5 factors and momentum
    ff5 = DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench")[0]
    mom = DataReader("F-F_Momentum_Factor_daily", "famafrench")[0]
    # Convert to decimal and align
    ff5 = ff5 / 100.0
    mom = mom / 100.0
    ff = ff5.join(mom, how="left")
    ff.index = pd.to_datetime(ff.index)
    ff = ff.loc[(ff.index >= pd.to_datetime(start)) & (ff.index < pd.to_datetime(end))]
    # Standardize column names
    ff = ff.rename(
        columns={
            "Mkt-RF": "MKT",
            "SMB": "SMB",
            "HML": "HML",
            "RMW": "RMW",
            "CMA": "CMA",
            "Mom   ": "MOM",
            "RF": "RF",
        }
    )
    if "MOM" not in ff.columns:
        for c in ff.columns:
            if c.strip().lower() in ("mom", "momentum", "umd"):
                ff = ff.rename(columns={c: "MOM"})
                break
    return ff

def regress_portfolio_on_factors(rp_excess: pd.Series, F: pd.DataFrame, nw_lags=5):
    import statsmodels.api as sm
    X = F[["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]].copy()
    X = sm.add_constant(X)
    # Align
    df = pd.concat([rp_excess, X], axis=1, join="inner").dropna()
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    betas = model.params.drop("const")
    resid = model.resid
    return betas, resid, df.iloc[:, 1:].drop(columns=["const"])

def build_report(betas, resid, F_aligned, gross_notional, window_30=30):
    # Factor covariance
    Sigma = F_aligned.cov()
    b = betas.values
    # Contributions to variance: b ⊙ (Σ b)
    Sigma_b = Sigma.values @ b
    mc = b * Sigma_b
    var_factor = float(b.T @ Sigma.values @ b)
    var_resid = float(resid.var(ddof=1))
    sigma2 = var_factor + var_resid if np.isfinite(var_resid) else var_factor
    pct_var = mc / sigma2 if sigma2 > 0 else np.zeros_like(mc)

    # Dollar exposure: beta * gross notional
    exposure = betas * gross_notional

    # 30D factor returns
    last30 = F_aligned.tail(window_30).sum()  # additive approximation
    fact_30d = last30 * 100.0  # percent

    # Portfolio factor-predicted 30D return (for "Total" row)
    port_30d = float((F_aligned.tail(window_30).values @ b).sum()) * 100.0

    df = pd.DataFrame(
        {
            "Factor": betas.index,
            "Beta": betas.values,
            "Exposure ($M)": (exposure.values / 1e6),
            "% of Book Variance": (pct_var * 100.0),
            "Factor Return % (30D)": fact_30d.reindex(betas.index).values,
        }
    )
    # Sort by absolute Beta descending
    df = df.reindex(df["Beta"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    total_row = pd.DataFrame(
        {
            "Factor": ["Total"],
            "Beta": [np.nan],
            "Exposure ($M)": [gross_notional / 1e6],
            "% of Book Variance": [var_factor / sigma2 * 100.0 if sigma2 > 0 else np.nan],
            "Factor Return % (30D)": [port_30d],
        }
    )
    report = pd.concat([total_row, df], ignore_index=True)
    return report

def main():
    ap = argparse.ArgumentParser(description="Portfolio factor report")
    ap.add_argument("portfolio_csv", help="CSV with columns: ticker, price, quantity (shorts negative)")
    ap.add_argument("--lookback", type=int, default=365, help="Days of history for regression and covariances")
    ap.add_argument("--nw_lags", type=int, default=5, help="Newey–West lags for daily data")
    args = ap.parse_args()

    # Load portfolio
    pf, gross = load_portfolio(args.portfolio_csv)
    tickers = pf["ticker"].astype(str).unique().tolist()

    # Dates
    end = datetime.utcnow().date()
    start = end - timedelta(days=max(args.lookback + 60, 120))

    # Market data
    rets = fetch_price_returns(tickers, start, end)
    # Weight vector indexed by ticker
    w = pf.set_index("ticker")["w_gross"]
    # Align weights to returns columns
    w = w.reindex(rets.columns).fillna(0.0)
    # Portfolio returns on gross capital
    rp = rets.mul(w, axis=1).sum(axis=1)

    # Factors
    ff = fetch_ff_factors(start, end)
    # Align and compute excess
    df = pd.concat([rp.rename("rp"), ff], axis=1, join="inner").dropna()
    rp_excess = df["rp"] - df["RF"]
    F = df[["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]]

    # Regression
    betas, resid, F_aligned = regress_portfolio_on_factors(rp_excess, F, nw_lags=args.nw_lags)

    # Report
    report = build_report(betas, resid, F_aligned, gross_notional=gross, window_30=30)

    # Print
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\nPortfolio factor report\n")
    print(report.to_string(index=False))
    print("\nNotes:")
    print("- Returns and factor data are daily. Factors from Ken French library via pandas-datareader.")
    print("- Betas from OLS with Newey–West standard errors (SEs not shown).")
    print("- % of Book Variance uses factor covariance of the sample window.")
    print("- Exposure ($M) = beta × gross notional.")
    print("- 30D factor returns are a simple sum of daily factor returns (percent).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
