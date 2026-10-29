"""End-to-end pipeline: clean -> RFM -> K-Means -> churn classification -> spend regression
-> revenue forecasting. Saves figures to reports/ and all metrics to reports/model_metrics.json.
Usage:  python run_pipeline.py [path/to/Online Retail.xlsx|csv] [--k 4]
Falls back to the sample dataset if the raw file is not found.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.cleaning import clean, load_raw, save_clean
from src.clustering import (add_pca, evaluate_k, fit_kmeans, name_segments, prepare_features,
                            segment_summary)
from src.forecasting import daily_revenue, run_forecasting
from src.rfm import build_rfm
from src.supervised import FEATURES, build_customer_table, run_classification, run_regression

p = argparse.ArgumentParser()
p.add_argument("path", nargs="?", default="data/raw/Online Retail.xlsx")
p.add_argument("--k", type=int, default=4)
args = p.parse_args()

src = Path(args.path)
if not src.exists():
    print(f"{src} not found -> using sample data")
    src = Path("data/sample/sample_online_retail.csv")
Path("reports").mkdir(exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

print(f"Loading {src} ...")
df, report = clean(load_raw(src))
for k, v in report.items():
    print(f"  {k}: {v:,}")
save_clean(df)
M = {"cleaning": report, "revenue_total": float(df["TotalPrice"].sum()),
     "avg_order_value": float(df.groupby("InvoiceNo")["TotalPrice"].sum().mean())}

# ---------------- 1. Segmentation (clustering) ----------------
rfm = build_rfm(df)
X = prepare_features(rfm)
metrics = evaluate_k(X)
seg = add_pca(X, name_segments(rfm, fit_kmeans(X, args.k)))
summary = segment_summary(seg)
print("\n", metrics.round(3).to_string(index=False), "\n\n", summary.to_string())
seg.to_csv("data/processed/rfm_segments.csv")
summary.to_csv("reports/segment_summary.csv")
M["rfm_describe"] = rfm.describe().round(2).to_dict()
M["rfm_corr_spearman"] = rfm.corr(method="spearman").round(3).to_dict()
M["k_metrics"] = metrics.round(4).to_dict("records")
M["k_selected"] = args.k
M["segments"] = summary.reset_index().to_dict("records")
M["country_revenue_top"] = df.groupby("Country")["TotalPrice"].sum().nlargest(5).round(0).to_dict()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(metrics.k, metrics.inertia, marker="o"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(metrics.k, metrics.silhouette, marker="o", color="tab:green"); ax[1].set(title="Silhouette score", xlabel="k")
fig.tight_layout(); fig.savefig("reports/cluster_selection.png", dpi=150); plt.close(fig)
plt.figure(figsize=(7, 5)); sns.scatterplot(data=seg, x="PC1", y="PC2", hue="Segment", s=18, palette="Set2")
plt.title(f"Customer segments (K-Means, k={args.k}, PCA view)"); plt.tight_layout()
plt.savefig("reports/segments_pca.png", dpi=150); plt.close()

# ---------------- 2. Classification: churn in the next 90 days ----------------
table, cutoff = build_customer_table(df)
clf = run_classification(table)
M["classification"] = {k: v for k, v in clf.items() if k != "roc"} | {"cutoff": str(cutoff.date())}
print(f"\nChurn classification (cutoff {cutoff.date()}): counts {clf['counts']}, best = {clf['best']}")
for name, m in clf["models"].items():
    print(f"  {name}: acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cm = np.array(clf["models"][clf["best"]]["confusion"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0], xticklabels=["Active", "Churned"], yticklabels=["Active", "Churned"])
ax[0].set(title=f"Confusion matrix: {clf['best']}", xlabel="Predicted", ylabel="Actual")
for name, (fpr, tpr) in clf["roc"].items():
    ax[1].plot(fpr, tpr, label=f"{name} (AUC {clf['models'][name]['roc_auc']:.2f})")
ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].set(title="ROC curves (test set)", xlabel="False positive rate", ylabel="True positive rate"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("reports/churn_classification.png", dpi=150); plt.close(fig)

# ---------------- 3. Regression: spend in the next 90 days ----------------
reg = run_regression(table)
M["regression"] = {k: v for k, v in reg.items() if k not in ("pred", "actual")}
print("\nSpend regression:")
for name, m in reg["models"].items():
    print(f"  {name}: MAE={m['mae']:.0f} RMSE={m['rmse']:.0f} R2={m['r2']:.3f} R2(log)={m['r2_log']:.3f}")
act, prd = np.array(reg["actual"]), np.array(reg["pred"][reg["best"]])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(np.log1p(act), np.log1p(prd), s=8, alpha=.5); lim = [0, max(np.log1p(act).max(), np.log1p(prd).max())]
ax[0].plot(lim, lim, "r--", lw=1); ax[0].set(title=f"Actual vs predicted ({reg['best']}, log scale)", xlabel="log(1 + actual spend)", ylabel="log(1 + predicted spend)")
ax[1].hist(np.log1p(prd) - np.log1p(act), bins=40); ax[1].set(title="Residuals (log scale)", xlabel="predicted - actual", ylabel="customers")
fig.tight_layout(); fig.savefig("reports/spend_regression.png", dpi=150); plt.close(fig)

# ---------------- 4. Time-series forecasting: daily revenue ----------------
series = daily_revenue(df)
fc = run_forecasting(series)
M["forecasting"] = {k: v for k, v in fc.items() if k not in ("dates", "actual", "pred")}
print(f"\nForecasting daily revenue (train {fc['train_start']}..{fc['train_end']}, test {fc['test_start']}..{fc['test_end']}), best = {fc['best']}")
for name, m in fc["models"].items():
    print(f"  {name}: MAE={m['mae']:.0f} RMSE={m['rmse']:.0f} MAPE={m['mape']:.1f}%")
import pandas as pd
idx = pd.to_datetime(fc["dates"]); n0 = fc["n_train"]
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(idx[:n0], fc["actual"][:n0], color="gray", lw=1, label="Train (actual)")
ax.plot(idx[n0:], fc["actual"][n0:], color="black", lw=1.4, label="Test (actual)")
ax.plot(idx[n0:], fc["pred"][fc["best"]], color="tab:red", lw=1.6, label=f"Forecast: {fc['best']}")
ax.plot(idx[n0:], fc["pred"]["Seasonal naive (7 days)"], color="tab:blue", lw=1, ls="--", label="Seasonal naive baseline")
ax.set(title="Daily revenue: actual vs forecast", xlabel="Date", ylabel="Revenue (GBP)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("reports/revenue_forecast.png", dpi=150); plt.close(fig)

json.dump(M, open("reports/model_metrics.json", "w"), indent=2, default=float)
print("\nSaved figures to reports/, metrics to reports/model_metrics.json")
