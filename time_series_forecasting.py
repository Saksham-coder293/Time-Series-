"""
Time Series Forecasting Walkthrough
===================================
Dataset : daily_sales.csv  (columns: date, sales, promo)
Goal    : forecast the next 28 days of sales

Steps
  1. Load and look at the data
  2. Decompose into trend / weekly season / residual
  3. Test stationarity (ADF) and difference the series
  4. Read ACF and PACF to choose model orders
  5. Split in time order (train / test)
  6. Baseline forecast (seasonal naive)
  7. SARIMA forecast
  8. Lag-feature machine learning forecast
  9. Compare models and check residuals

Run:  pip install pandas numpy matplotlib statsmodels scikit-learn
      python time_series_forecasting.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

HORIZON = 28   # days to forecast
SEASON = 7     # weekly pattern in daily data


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
# parse_dates + index_col turns "date" into a proper time index.
# asfreq("D") makes the spacing explicit (lags assume evenly spaced data).
df = pd.read_csv("daily_sales.csv", parse_dates=["date"], index_col="date").asfreq("D")
y = df["sales"]
print(df.head(), "\n")
print(f"Rows: {len(df)}, missing values: {int(y.isna().sum())}\n")

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(y, lw=1)
ax.set(title="Daily sales", ylabel="Sales")
fig.tight_layout()
fig.savefig("01_raw_series.png")


# ---------------------------------------------------------------------------
# 2. DECOMPOSITION
# ---------------------------------------------------------------------------
# Splits the series into trend + seasonal + residual.
# Weekend peaks look the same size at any level, so we use an additive model.
# (If swings grew with the level, use model="multiplicative".)
dec = seasonal_decompose(y, model="additive", period=SEASON)
fig = dec.plot()
fig.set_size_inches(11, 7)
fig.tight_layout()
fig.savefig("02_decomposition.png")


# ---------------------------------------------------------------------------
# 3. STATIONARITY
# ---------------------------------------------------------------------------
# ADF test: null hypothesis = "the series is NOT stationary".
# p-value < 0.05 -> reject the null -> series is stationary.
def adf_report(series, name):
    stat, p, *_ = adfuller(series.dropna())
    verdict = "stationary" if p < 0.05 else "NOT stationary"
    print(f"ADF on {name:<28} p-value = {p:.4f} -> {verdict}")

adf_report(y, "raw sales")
y_d1 = y.diff()                   # d = 1: removes the trend
y_d1_s7 = y_d1.diff(SEASON)       # D = 1, s = 7: removes the weekly cycle
adf_report(y_d1, "after d=1")
adf_report(y_d1_s7, "after d=1 and D=1 (lag 7)")
print()

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for ax, s, title in zip(axes, [y, y_d1, y_d1_s7],
                        ["Raw", "Differenced once (trend removed)",
                         "Then seasonal difference (weekly cycle removed)"]):
    ax.plot(s, lw=0.8)
    ax.set_title(title, fontsize=10)
fig.tight_layout()
fig.savefig("03_differencing.png")


# ---------------------------------------------------------------------------
# 4. ACF / PACF  (use them on the stationary version)
# ---------------------------------------------------------------------------
# Rules of thumb:
#   PACF cuts off after lag p -> AR(p)        ACF cuts off after lag q -> MA(q)
#   Spikes at lag 7, 14, 21   -> seasonal terms with s = 7
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
plot_acf(y.dropna(), lags=28, ax=axes[0, 0], title="ACF - raw")
plot_pacf(y.dropna(), lags=28, ax=axes[0, 1], title="PACF - raw", method="ywm")
plot_acf(y_d1_s7.dropna(), lags=28, ax=axes[1, 0], title="ACF - after differencing")
plot_pacf(y_d1_s7.dropna(), lags=28, ax=axes[1, 1], title="PACF - after differencing", method="ywm")
fig.tight_layout()
fig.savefig("04_acf_pacf.png")


# ---------------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT  (always in time order, never shuffled)
# ---------------------------------------------------------------------------
train, test = y.iloc[:-HORIZON], y.iloc[-HORIZON:]
promo_train, promo_test = df["promo"].iloc[:-HORIZON], df["promo"].iloc[-HORIZON:]
print(f"Train: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} days)")
print(f"Test : {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} days)\n")


def score(actual, pred):
    err = actual - pred
    return {"MAE": err.abs().mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "MAPE %": (err.abs() / actual).mean() * 100}


# ---------------------------------------------------------------------------
# 6. BASELINE: SEASONAL NAIVE
# ---------------------------------------------------------------------------
# "Next Monday will equal last Monday." Any real model should beat this.
last_week = train.iloc[-SEASON:].values
snaive = pd.Series(np.tile(last_week, HORIZON // SEASON + 1)[:HORIZON], index=test.index)


# ---------------------------------------------------------------------------
# 7. SARIMA  (with promo as an external regressor)
# ---------------------------------------------------------------------------
# order          = (p, d, q)         -> d=1 removes the trend
# seasonal_order = (P, D, Q, s)      -> D=1, s=7 removes the weekly cycle
# exog           = known-in-advance features (promo calendar)
sarima = SARIMAX(train, exog=promo_train,
                 order=(1, 1, 1), seasonal_order=(1, 1, 1, SEASON)).fit(disp=False)
print(sarima.summary().tables[1], "\n")
print(f"AIC = {sarima.aic:.1f}   (lower is better; use it to compare candidate orders)\n")

fc = sarima.get_forecast(steps=HORIZON, exog=promo_test)
sarima_pred = fc.predicted_mean
conf = fc.conf_int(alpha=0.05)


# ---------------------------------------------------------------------------
# 8. LAG-FEATURE ML MODEL  (the "lags as columns" approach)
# ---------------------------------------------------------------------------
# Lags 1, 7, 14, 28 are chosen from the ACF/PACF. Rolling stats use shift(1)
# first so the current value never leaks into its own features.
feat = pd.DataFrame({"y": y, "promo": df["promo"]})
for k in (1, 2, 7, 14, 28):
    feat[f"lag_{k}"] = y.shift(k)
feat["roll7_mean"] = y.shift(1).rolling(7).mean()
feat["dow"] = feat.index.dayofweek
feat = feat.dropna()

X_cols = [c for c in feat.columns if c != "y"]
tr, te = feat.loc[:train.index[-1]], feat.loc[test.index[0]:]
gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
gbm.fit(tr[X_cols], tr["y"])

# Recursive multi-step forecast: predict one day, feed it back as a lag.
hist = y.loc[:train.index[-1]].copy()
gbm_vals = []
for d in test.index:
    row = {"promo": promo_test.loc[d]}
    for k in (1, 2, 7, 14, 28):
        row[f"lag_{k}"] = hist.iloc[-k]
    row["roll7_mean"] = hist.iloc[-7:].mean()
    row["dow"] = d.dayofweek
    p = gbm.predict(pd.DataFrame([row])[X_cols])[0]
    gbm_vals.append(p)
    hist.loc[d] = p
gbm_pred = pd.Series(gbm_vals, index=test.index)


# ---------------------------------------------------------------------------
# 9. COMPARE MODELS
# ---------------------------------------------------------------------------
results = pd.DataFrame({
    "Seasonal naive": score(test, snaive),
    "SARIMA": score(test, sarima_pred),
    "Gradient boosting (lags)": score(test, gbm_pred),
}).T.round(2)
print("Test-set accuracy (lower is better)")
print(results, "\n")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(train.iloc[-60:], color="gray", label="Train (last 60 days)")
ax.plot(test, color="black", lw=2, label="Actual")
ax.plot(snaive, "--", label="Seasonal naive")
ax.plot(sarima_pred, label="SARIMA")
ax.plot(gbm_pred, label="Gradient boosting")
ax.fill_between(conf.index, conf.iloc[:, 0], conf.iloc[:, 1], alpha=0.15, label="SARIMA 95% interval")
ax.legend(ncol=3, fontsize=8)
ax.set(title=f"{HORIZON}-day forecast comparison", ylabel="Sales")
fig.tight_layout()
fig.savefig("05_forecast_comparison.png")

# Residual check: if the model captured the structure, residuals look like noise.
resid = sarima.resid.iloc[SEASON + 1:]
lb = acorr_ljungbox(resid, lags=[10], return_df=True)
print(f"Ljung-Box p-value on SARIMA residuals = {lb['lb_pvalue'].iloc[0]:.3f} "
      "(> 0.05 means no leftover autocorrelation)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(resid, lw=0.8)
axes[0].set_title("SARIMA residuals")
plot_acf(resid, lags=28, ax=axes[1], title="Residual ACF (should be inside the band)")
fig.tight_layout()
fig.savefig("06_residuals.png")

print("\nSaved figures: 01_raw_series.png ... 06_residuals.png")
