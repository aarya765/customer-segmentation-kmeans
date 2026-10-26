"""End-to-end pipeline (no dashboard): clean -> RFM -> K-Means -> save outputs.
Usage:  python run_pipeline.py [path/to/Online Retail.xlsx] [--k 4]
Falls back to the sample dataset if the raw file is not found.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.cleaning import load_raw, clean, save_clean
from src.rfm import build_rfm
from src.clustering import (prepare_features, evaluate_k, fit_kmeans,
                            name_segments, add_pca, segment_summary)

p = argparse.ArgumentParser()
p.add_argument("path", nargs="?", default="data/raw/Online Retail.xlsx")
p.add_argument("--k", type=int, default=4)
args = p.parse_args()

src = Path(args.path)
if not src.exists():
    print(f"{src} not found -> using sample data")
    src = Path("data/sample/sample_online_retail.csv")

print(f"Loading {src} ...")
df, report = clean(load_raw(src))
for k, v in report.items():
    print(f"  {k}: {v:,}")
save_clean(df)

rfm = build_rfm(df)
X = prepare_features(rfm)
metrics = evaluate_k(X)
print("\n", metrics.round(3).to_string(index=False))

seg = add_pca(X, name_segments(rfm, fit_kmeans(X, args.k)))
summary = segment_summary(seg)
print("\n", summary.to_string())

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)
seg.to_csv("data/processed/rfm_segments.csv")
summary.to_csv("reports/segment_summary.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(metrics.k, metrics.inertia, marker="o"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(metrics.k, metrics.silhouette, marker="o", color="tab:green"); ax[1].set(title="Silhouette score", xlabel="k")
fig.tight_layout(); fig.savefig("reports/cluster_selection.png", dpi=150); plt.close(fig)

plt.figure(figsize=(7, 5))
sns.scatterplot(data=seg, x="PC1", y="PC2", hue="Segment", s=18, palette="Set2")
plt.title(f"Customer segments (K-Means, k={args.k}, PCA view)")
plt.tight_layout(); plt.savefig("reports/segments_pca.png", dpi=150)
print("\nSaved: data/processed/rfm_segments.csv, reports/*.png, reports/segment_summary.csv")
