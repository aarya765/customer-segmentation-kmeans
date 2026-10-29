"""Supervised tasks built on the same transactions:
   * Classification: will a customer churn (no purchase in the next 90 days)?
   * Regression:     how much will a customer spend in the next 90 days?

Leakage-free design: features use only transactions up to a cutoff date;
targets use the 90 days after the cutoff.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary", "Tenure", "AvgOrderValue"]
HORIZON_DAYS = 90
SEED = 42
_log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build_customer_table(df: pd.DataFrame, horizon_days: int = HORIZON_DAYS):
    """One row per customer seen before the cutoff, with future targets."""
    cutoff = df["InvoiceDate"].max().normalize() - pd.Timedelta(days=horizon_days)
    snapshot = cutoff + pd.Timedelta(days=1)
    past, future = df[df["InvoiceDate"] < snapshot], df[df["InvoiceDate"] >= snapshot]
    g = past.groupby("CustomerID")
    t = pd.DataFrame({
        "Recency": g["InvoiceDate"].apply(lambda s: (snapshot - s.max()).days),
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["TotalPrice"].sum(),
        "Tenure": g["InvoiceDate"].apply(lambda s: (snapshot - s.min()).days),
    })
    t["AvgOrderValue"] = t["Monetary"] / t["Frequency"]
    t["FutureSpend"] = future.groupby("CustomerID")["TotalPrice"].sum().reindex(t.index).fillna(0.0)
    t["Churned"] = (t["FutureSpend"] == 0).astype(int)
    return t, cutoff


def run_classification(t: pd.DataFrame) -> dict:
    X, y = t[FEATURES], t["Churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    models = {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(_log, StandardScaler(),
                                             LogisticRegression(class_weight="balanced", max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                random_state=SEED, n_jobs=-1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {"counts": y.value_counts().sort_index().to_dict(), "churn_rate": float(y.mean()),
           "n_train": len(Xtr), "n_test": len(Xte), "models": {}, "roc": {}}
    fitted = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        proba = m.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba) if len(set(proba)) > 1 else 0.5
        cvs = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
        out["models"][name] = {
            "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0), "f1": f1_score(yte, pred, zero_division=0),
            "roc_auc": auc, "cv_auc_mean": float(cvs.mean()), "cv_auc_std": float(cvs.std()),
            "confusion": confusion_matrix(yte, pred).tolist()}
        fpr, tpr, _ = roc_curve(yte, proba)
        out["roc"][name] = (fpr.tolist(), tpr.tolist())
        fitted[name] = m
    rf = fitted["Random Forest"]
    out["importance"] = dict(zip(FEATURES, rf.feature_importances_.tolist()))
    out["best"] = max((k for k in out["models"] if not k.startswith("Baseline")), key=lambda k: out["models"][k]["roc_auc"])
    return out


def run_regression(t: pd.DataFrame) -> dict:
    X, y = t[FEATURES], t["FutureSpend"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    def fit_predict(name):
        if name == "Baseline (mean)":
            return DummyRegressor(strategy="mean").fit(Xtr, ytr).predict(Xte)
        if name == "Linear Regression":
            m = make_pipeline(_log, StandardScaler(), LinearRegression()).fit(Xtr, np.log1p(ytr))
            return np.clip(np.expm1(m.predict(Xte)), 0, None)
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1).fit(Xtr, np.log1p(ytr))
        return np.clip(np.expm1(m.predict(Xte)), 0, None)

    out = {"n_train": len(Xtr), "n_test": len(Xte), "models": {}, "pred": {}, "actual": yte.tolist()}
    for name in ["Baseline (mean)", "Linear Regression", "Random Forest (log target)"]:
        p = fit_predict(name)
        nz = yte > 0
        out["models"][name] = {"mae": mean_absolute_error(yte, p), "rmse": float(np.sqrt(mean_squared_error(yte, p))),
                               "r2": r2_score(yte, p),
                               "median_ae": float(np.median(np.abs(yte - p))),
                               "r2_log": r2_score(np.log1p(yte), np.log1p(p)),
                               "mae_log": mean_absolute_error(np.log1p(yte), np.log1p(p))}
        out["pred"][name] = p.tolist()
    out["best"] = max((k for k in out["models"] if not k.startswith("Baseline")), key=lambda k: out["models"][k]["r2"])
    return out
