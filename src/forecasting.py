"""Daily revenue forecasting with a chronological train/test split."""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

TRAIN_FRAC = 0.8
SEASON = 7


def daily_revenue(df: pd.DataFrame) -> pd.Series:
    """Revenue per calendar day (0 on days with no sales). The last, partial day is dropped."""
    s = df.set_index("InvoiceDate")["TotalPrice"].resample("D").sum()
    return s.iloc[:-1] if df["InvoiceDate"].max().hour or df["InvoiceDate"].max().minute else s


def _metrics(actual, pred):
    err = actual - pred
    nz = actual > 0
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": float(np.mean(np.abs(err[nz] / actual[nz])) * 100)}


def run_forecasting(series: pd.Series, train_frac: float = TRAIN_FRAC) -> dict:
    n_train = int(len(series) * train_frac)
    train, test = series.iloc[:n_train], series.iloc[n_train:]
    h = len(test)
    preds = {}
    preds["Naive (last value)"] = np.repeat(train.iloc[-1], h)
    last_season = train.iloc[-SEASON:].to_numpy()
    preds["Seasonal naive (7 days)"] = np.tile(last_season, h // SEASON + 1)[:h]
    hw = ExponentialSmoothing(train + 1, trend="add", damped_trend=True, seasonal="add",
                              seasonal_periods=SEASON).fit()
    preds["Holt-Winters (additive, damped trend)"] = np.clip(hw.forecast(h).to_numpy() - 1, 0, None)
    sx = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, SEASON)).fit(disp=False)
    preds["SARIMA(1,1,1)(1,0,1,7)"] = np.clip(sx.forecast(h).to_numpy(), 0, None)
    actual = test.to_numpy()
    out = {"n_total": len(series), "n_train": n_train, "n_test": h,
           "train_start": str(train.index[0].date()), "train_end": str(train.index[-1].date()),
           "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date()),
           "zero_days": int((series == 0).sum()),
           "zero_days_by_weekday": {d: int(((series == 0) & (series.index.day_name() == d)).sum()) for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
           "train_mean": float(train.mean()), "test_mean": float(test.mean()),
           "models": {k: _metrics(actual, v) for k, v in preds.items()},
           "dates": [str(d.date()) for d in series.index], "actual": series.tolist(),
           "pred": {k: v.tolist() for k, v in preds.items()}}
    out["best"] = min((k for k in out["models"] if not k.startswith(("Naive", "Seasonal naive"))), key=lambda k: out["models"][k]["rmse"])
    return out
