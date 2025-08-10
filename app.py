# app.py
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# Optional yfinance download (app still works without it)
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def to_returns(df: pd.DataFrame, price_like=True):
    """Return DataFrame of periodic returns from prices; pass-through if already returns."""
    df = df.sort_index()
    if price_like:
        r = df.pct_change().dropna(how="all")
    else:
        r = df.copy()
    r = r.select_dtypes(include=[np.number])
    return r.dropna(how="all", axis=1)

def annualize(ret: pd.DataFrame, freq: str):
    freq = freq.lower()
    periods = {"daily":252, "weekly":52, "monthly":12, "quarterly":4, "yearly":1}[freq]
    mu = ret.mean() * periods
    cov = ret.cov() * periods
    return mu, cov, periods

def portfolio_metrics(w, mu, cov, rf=0.0):
    w = np.asarray(w)
    exp_ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (exp_ret - rf) / vol if vol > 0 else np.nan
    return exp_ret, vol, sharpe

def solve_qp_max_sharpe(mu, cov, rf, bounds, w_sum=1.0):
    from scipy.optimize import minimize
    n = len(mu)

    def neg_sharpe(w):
        _, _, s = portfolio_metrics(w, mu, cov, rf)
        return -s

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - w_sum}]
    bnds = bounds if bounds is not None else tuple((0.0, 1.0) for _ in range(n))
    x0 = np.full(n, 1.0/n)
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter":1000})
    return res.x, res

def solve_qp_min_vol(mu, cov, target_ret, bounds, w_sum=1.0):
    from scipy.optimize import minimize
    n = len(mu)
    def vol(w): return np.sqrt(w @ cov @ w)

    cons = [
        {"type":"eq", "fun": lambda w: np.sum(w) - w_sum},
        {"type":"eq", "fun": lambda w: w @ mu - target_ret},
    ]
    bnds = bounds if bounds is not None else tuple((0.0, 1.0) for _ in range(n))
    x0 = np.full(n, 1.0/n)
    res = minimize(vol, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter":1000})
    return res.x, res

def solve_qp_min_vol_unconstrained(cov, bounds, w_sum=1.0):
    from scipy.optimize import minimize
    n = cov.shape[0]
    def vol(w): return np.sqrt(w @ cov @ w)
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - w_sum}]
    bnds = bounds if bounds is not None else tuple((0.0, 1.0) for _ in range(n))
    x0 = np.full(n, 1.0/n)
    res = minimize(vol, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter":1000})
    return res.x, res

def efficient_frontier(mu, cov, bounds, rf, points=30, w_sum=1.0):
    """Generate efficient frontier as a DataFrame with Return, Volatility, Sharpe."""
    w_min, _ = solve_qp_min_vol_unconstrained(cov, bounds, w_sum=w_sum)
    r_min = float(w_min @ mu)
    r_max = float(np.max(mu))  # simple upper bound without shorting
    rs = np.linspace(r_min, r_max, points)
    vols, sharpes = [], []
    for r in rs:
        try:
            w, _ = solve_qp_min_vol(mu, cov, target_ret=r, bounds=bounds, w_sum=w_sum)
            _, v, s = portfolio_metrics(w, mu, cov, rf)
            vols.append(v); sharpes.append(s)
        except Exception:
            vols.append(np.nan); sharpes.append(np.nan)
    ef = pd.DataFrame({"Return": rs, "Volatility": vols, "Sharpe": sharpes}).dropna()
    return ef

def nice_percent(x): return f"{x:.2%}"

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Data")
data_source = st.sidebar.radio("Provide data via:", ["Upload CSV", "Download with tickers (yfinance)"])
freq = st.sidebar.selectbox("Data frequency", ["Daily","Weekly","Monthly","Quarterly","Yearly"], index=2)
rf = st.sidebar.number_input("Risk-free rate (annualized)", value=0.0, step=0.005, format="%.4f")
long_only = st.sidebar.checkbox("Long-only (no shorting)", value=True)
max_w = st.sidebar.slider("Max weight per asset", 0.05, 1.0, 1.0, 0.05)
w_sum = 1.0  # fully invested

st.title("📈 Portfolio Optimizer (Self-Contained)")

# ---------------------------
# Load Data
# ---------------------------
price_like = True
ret_df = None

if data_source == "Upload CSV":
    st.write("Upload **Prices** (columns=tickers, rows=dates) or **Returns** (decimals) CSV.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        raw = pd.read_csv(file)
        if "Date" in raw.columns:
            raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
            raw = raw.dropna(subset=["Date"]).set_index("Date")
        else:
            try:
                raw.iloc[:,0] = pd.to_datetime(raw.iloc[:,0], errors="coerce")
                raw = raw.dropna(subset=[raw.columns[0]]).set_index(raw.columns[0])
            except Exception:
                raw.index = pd.to_datetime(raw.index, errors="coerce")
        sample = raw.select_dtypes(include=[np.number]).head(10)
        price_like = (sample.abs() > 2.0).any().any()
        ret_df = to_returns(raw, price_like=price_like)
else:
    st.write("Enter tickers separated by commas. Example: AAPL, MSFT, NVDA, TLT, GLD")
    tickers = st.text_input("Tickers", "AAPL, MSFT, NVDA, TLT, GLD")
    start = st.date_input("Start date", datetime(2018,1,1))
    end = st.date_input("End date", datetime.today())
    if st.button("Download Prices", disabled=not YF_OK):
        if not YF_OK:
            st.error("`yfinance` not available. Add it to requirements.txt or use CSV upload.")
        else:
            tks = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            data = yf.download(tks, start=start, end=end, auto_adjust=True, progress=False)["Close"]
            data = data.dropna(how="all")
            ret_df = to_returns(data, price_like=True)

# Sample fallback
if ret_df is None:
    st.info("No data yet — using sample monthly prices for a demo.")
    dates = pd.date_range("2021-01-31", periods=48, freq="M")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame({
        "ASSET_A": 100*np.cumprod(1+rng.normal(0.01,0.05, len(dates))),
        "ASSET_B":  80*np.cumprod(1+rng.normal(0.008,0.04, len(dates))),
        "ASSET_C": 120*np.cumprod(1+rng.normal(0.012,0.06, len(dates))),
        "ASSET_D":  95*np.cumprod(1+rng.normal(0.006,0.03, len(dates))),
    }, index=dates)
    ret_df = to_returns(prices, price_like=True)
    freq = "Monthly"

# ---------------------------
# Compute stats
# ---------------------------
mu, cov, periods = annualize(ret_df, freq)
assets = list(mu.index)
n = len(assets)
bounds = tuple(((0.0, min(max_w,1.0)) if long_only else (-min(max_w,1.0), min(max_w,1.0))) for _ in range(n))

st.subheader("Inputs & Detected Settings")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets", n)
c2.metric("Obs (rows)", ret_df.shape[0])
c3.metric("Frequency", freq)
c4.metric("Risk-free", nice_percent(rf))

with st.expander("Annualized stats"):
    st.dataframe(pd.DataFrame({"Expected Return": mu}).style.format("{:.2%}"))
    st.write("Covariance Matrix (annualized):")
    st.dataframe(pd.DataFrame(cov, index=assets, columns=assets).style.format("{:.4f}"))

# ---------------------------
# Optimization
# ---------------------------
st.subheader("Optimization")
objective = st.selectbox("Objective", ["Max Sharpe", "Min Volatility", "Target Return (Min Volatility)"])
target_ret = None
if objective == "Target Return (Min Volatility)":
    target_ret = st.slider("Target return (annualized)", float(mu.min()), float(mu.max()), float(mu.mean()))

if objective == "Max Sharpe":
    weights, _ = solve_qp_max_sharpe(mu.values, cov.values, rf, bounds, w_sum=w_sum)
elif objective == "Min Volatility":
    weights, _ = solve_qp_min_vol_unconstrained(cov.values, bounds, w_sum=w_sum)
else:
    weights, _ = solve_qp_min_vol(mu.values, cov.values, target_ret, bounds, w_sum=w_sum)

w_series = pd.Series(weights, index=assets)
er, vol, sh = portfolio_metrics(weights, mu.values, cov.values, rf)

c1, c2, c3 = st.columns(3)
c1.metric("Expected Return", nice_percent(er))
c2.metric("Volatility", nice_percent(vol))
c3.metric("Sharpe", f"{sh:.2f}")

# Weights pie
fig1, ax1 = plt.subplots()
ax1.pie(w_series.clip(lower=0).values, labels=w_series.index, autopct="%1.1f%%", startangle=90)
ax1.axis("equal")
st.pyplot(fig1)

# Efficient Frontier
st.subheader("Efficient Frontier")
ef = efficient_frontier(mu.values, cov.values, bounds, rf, points=40, w_sum=w_sum)
fig2, ax2 = plt.subplots()
ax2.plot(ef["Volatility"], ef["Return"])
ax2.scatter([vol], [er], marker="*", s=200)
ax2.set_xlabel("Volatility (σ)")
ax2.set_ylabel("Expected Return (μ)")
ax2.set_title("Efficient Frontier (long-only constrained)" if long_only else "Efficient Frontier")
st.pyplot(fig2)

# Weights table + download
st.subheader("Optimized Weights")
w_tbl = pd.DataFrame({"Weight": w_series, "Expected Return": mu})
st.dataframe(w_tbl.style.format({"Weight":"{:.2%}", "Expected Return":"{:.2%}"}))
st.download_button("Download Weights (CSV)", w_tbl.to_csv().encode("utf-8"),
                   file_name="optimized_weights.csv", mime="text/csv")

st.caption("Tip: Uncheck long-only and set max weight (e.g., 0.3) to allow ±30% weights.")
