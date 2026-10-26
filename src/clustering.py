"""K-Means segmentation on RFM features."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]


def prepare_features(rfm: pd.DataFrame) -> np.ndarray:
    """RFM is heavily right-skewed -> log transform, then standardise."""
    logged = np.log1p(rfm[FEATURES])
    return StandardScaler().fit_transform(logged)


def evaluate_k(X: np.ndarray, k_range=range(2, 11), seed=42) -> pd.DataFrame:
    """Inertia (elbow) and silhouette score for each k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int, seed=42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


_NAMES = {
    2: ["Active High-Value", "Inactive Low-Value"],
    3: ["Champions", "Occasional Buyers", "Lost / Inactive"],
    4: ["Champions", "Loyal Customers", "Occasional Buyers", "At-Risk / Lost"],
    5: ["Champions", "Loyal Customers", "Potential Loyalists",
        "Occasional Buyers", "At-Risk / Lost"],
}
_BASE = ["Champions", "Loyal Customers", "Potential Loyalists", "Promising",
         "Occasional Buyers", "Needs Attention", "At-Risk"]


def name_segments(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Add Cluster + Segment columns. Clusters are ranked by a simple score
    (high Frequency + high Monetary + low Recency) and given business names."""
    out = rfm.copy()
    out["Cluster"] = labels
    means = out.groupby("Cluster")[FEATURES].mean()
    score = means["Frequency"].rank() + means["Monetary"].rank() - means["Recency"].rank()
    order = score.sort_values(ascending=False).index.tolist()
    k = len(order)
    names = _NAMES.get(k) or (_BASE[: k - 1] + ["Lost / Inactive"])
    out["Segment"] = out["Cluster"].map(dict(zip(order, names)))
    return out


def add_pca(X: np.ndarray, seg: pd.DataFrame) -> pd.DataFrame:
    """2-D PCA projection for plotting clusters."""
    comp = PCA(n_components=2, random_state=42).fit_transform(X)
    seg = seg.copy()
    seg["PC1"], seg["PC2"] = comp[:, 0], comp[:, 1]
    return seg


def segment_summary(seg: pd.DataFrame) -> pd.DataFrame:
    g = seg.groupby("Segment").agg(
        Customers=("Recency", "size"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum"),
    )
    g["Revenue_Share_%"] = 100 * g["Total_Revenue"] / g["Total_Revenue"].sum()
    return g.sort_values("Total_Revenue", ascending=False).round(2)
