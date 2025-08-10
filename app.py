# app.py
import io
import math
import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting (matplotlib) & yfinance
import matplotlib.pyplot as plt
from datetime import datetime
try:
    import yfinance as yf  # optional; app works without it
    YF_OK = True
except Exception:
    YF_OK = False

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def to_returns(df: pd.DataFrame, price_like=True):
    """Return dataframe of periodic returns from prices or pass-through if already returns."""
    df = df.sort_index()
    if price_like:
        r = df.pct_change().dropna(how="all")
    else:
        r = df.copy()
    # drop non-numeric cols
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
    # Maximize (mu - rf)·w / sqrt(w^T Σ w)
    # Equivalent to maximize (mu-rf)·w subject to w^T Σ w = 1 (then scale).
    # Easier: use scipy.minimize on negative Sharpe with constraints.
    from scipy.optimize import minimize

    n = len(mu)
    mu_excess = mu - rf

    def neg_sharpe(w):
        r, v, s = portfolio_metrics(w, mu, cov, rf)
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

def efficient_frontier(mu, cov, bounds, rf, points=30, w_sum
