"""Streamlit dashboard: E-Commerce Customer Segmentation (K-Means on RFM).
Run:  streamlit run app/dashboard.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.cleaning import clean, load_raw
from src.clustering import (add_pca, evaluate_k, fit_kmeans, name_segments,
                            prepare_features, segment_summary)
from src.rfm import build_rfm

RAW = ROOT / "data" / "raw" / "Online Retail.xlsx"
SAMPLE = ROOT / "data" / "sample" / "sample_online_retail.csv"
CACHED = ROOT / "data" / "processed" / "clean_transactions.csv"

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning transactions...")
def get_clean(kind: str, payload: bytes | None = None, name: str | None = None):
    if kind == "upload":
        df, report = clean(load_raw(io.BytesIO(payload), name=name))
    elif kind == "raw":
        if CACHED.exists() and CACHED.stat().st_mtime > RAW.stat().st_mtime:
            df = pd.read_csv(CACHED, parse_dates=["InvoiceDate"])   # fast path after 1st run
            return df, {"clean_rows": len(df), "customers": df["CustomerID"].nunique()}
        df, report = clean(load_raw(RAW))
        CACHED.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHED, index=False)
    else:
        df, report = clean(load_raw(SAMPLE))
    return df, report


@st.cache_data(show_spinner="Running K-Means...")
def run_model(df: pd.DataFrame, k: int):
    rfm = build_rfm(df)
    X = prepare_features(rfm)
    seg = add_pca(X, name_segments(rfm, fit_kmeans(X, k)))
    return seg, evaluate_k(X)


# ---------------- sidebar ----------------
st.sidebar.title("⚙️ Settings")
options = (["UCI Online Retail (data/raw)"] if RAW.exists() else []) \
    + ["Upload file", "Demo sample (synthetic)"]
choice = st.sidebar.radio("Data source", options)

if choice.startswith("UCI"):
    df, report = get_clean("raw")
elif choice == "Upload file":
    up = st.sidebar.file_uploader("Online Retail .xlsx / .csv", type=["xlsx", "csv"])
    if up is None:
        st.info("Upload the UCI Online Retail file in the sidebar, or switch to the demo sample.")
        st.stop()
    df, report = get_clean("upload", up.getvalue(), up.name)
else:
    st.sidebar.caption("Synthetic data with the same columns as UCI Online Retail.")
    df, report = get_clean("sample")

k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)
seg, metrics = run_model(df, k)
best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
st.sidebar.caption(f"Highest silhouette score at k = {best_k}")

# ---------------- header ----------------
st.title("🛍️ E-Commerce Customer Segmentation")
st.caption("RFM analysis + K-Means clustering · ADS Mini Project · Group 17")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{seg.shape[0]:,}")
c2.metric("Transactions", f"{len(df):,}")
c3.metric("Total revenue", f"£{df['TotalPrice'].sum():,.0f}")
c4.metric("Avg order value", f"£{df.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📐 Choosing k", "🎯 Segments", "🔎 Customers"])

# ---------------- overview ----------------
with tab1:
    left, right = st.columns(2)
    monthly = (df.set_index("InvoiceDate")["TotalPrice"].resample("MS").sum().reset_index())
    left.plotly_chart(px.line(monthly, x="InvoiceDate", y="TotalPrice", markers=True,
                              title="Monthly revenue"), width="stretch")
    country = df.groupby("Country")["TotalPrice"].sum().nlargest(10).reset_index()
    right.plotly_chart(px.bar(country, x="TotalPrice", y="Country", orientation="h",
                              title="Top 10 countries by revenue")
                       .update_layout(yaxis={"categoryorder": "total ascending"}),
                       width="stretch")
    d1, d2, d3 = st.columns(3)
    for col, feat in zip((d1, d2, d3), ("Recency", "Frequency", "Monetary")):
        col.plotly_chart(px.histogram(seg, x=feat, nbins=40, title=f"{feat} distribution",
                                      log_y=True), width="stretch")
    with st.expander("Data cleaning report"):
        st.json({k_: int(v) for k_, v in report.items()})

# ---------------- choosing k ----------------
with tab2:
    a, b = st.columns(2)
    a.plotly_chart(px.line(metrics, x="k", y="inertia", markers=True,
                           title="Elbow method (inertia)"), width="stretch")
    b.plotly_chart(px.line(metrics, x="k", y="silhouette", markers=True,
                           title="Silhouette score (higher is better)"), width="stretch")
    st.write("Pick k where the elbow flattens **and** the silhouette score is still reasonable. "
             "Business interpretability also matters: 4–5 segments are easy to act on.")
    st.dataframe(metrics.round(3), hide_index=True)

# ---------------- segments ----------------
with tab3:
    summary = segment_summary(seg)
    st.subheader("Segment summary")
    st.dataframe(summary, width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.pie(summary.reset_index(), names="Segment", values="Customers",
                          title="Customers per segment", hole=0.4), width="stretch")
    b.plotly_chart(px.pie(summary.reset_index(), names="Segment", values="Total_Revenue",
                          title="Revenue share per segment", hole=0.4), width="stretch")

    st.plotly_chart(px.scatter(seg.reset_index(), x="PC1", y="PC2", color="Segment",
                               hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
                               title="Clusters in 2-D (PCA of scaled RFM)", opacity=0.7),
                    width="stretch")
    st.plotly_chart(px.scatter_3d(seg.reset_index(), x="Recency", y="Frequency", z="Monetary",
                                  color="Segment", log_x=False, opacity=0.7,
                                  title="Clusters in RFM space"), width="stretch")

    prof = summary.reset_index().melt(id_vars="Segment",
                                      value_vars=["Avg_Recency", "Avg_Frequency", "Avg_Monetary"])
    st.plotly_chart(px.bar(prof, x="Segment", y="value", color="Segment", facet_col="variable",
                           title="Average RFM profile per segment")
                    .update_yaxes(matches=None, showticklabels=True), width="stretch")

    st.subheader("Suggested actions")
    actions = {
        "Champions": "Reward them, early access to new products, ask for reviews/referrals.",
        "Loyal Customers": "Upsell higher-value items, loyalty programme, bundle offers.",
        "Potential Loyalists": "Nudge a repeat purchase with personalised recommendations.",
        "Promising": "Onboarding emails and small incentives to build the habit.",
        "Occasional Buyers": "Seasonal campaigns and discounts to raise purchase frequency.",
        "Needs Attention": "Limited-time offers, remind them of items viewed or bought.",
        "At-Risk / Lost": "Win-back campaign with a strong discount; stop heavy spend if no response.",
        "At-Risk": "Win-back campaign with a personalised offer.",
        "Lost / Inactive": "Low-cost win-back email; otherwise deprioritise.",
        "Active High-Value": "Protect with VIP treatment and loyalty rewards.",
        "Inactive Low-Value": "Low-cost reactivation campaign.",
    }
    for s in summary.index:
        st.markdown(f"- **{s}**: {actions.get(s, 'Review manually.')}")

# ---------------- customers ----------------
with tab4:
    chosen = st.multiselect("Filter by segment", sorted(seg["Segment"].unique()),
                            default=sorted(seg["Segment"].unique()))
    view = seg[seg["Segment"].isin(chosen)][["Recency", "Frequency", "Monetary", "Segment"]]
    st.dataframe(view, width="stretch")
    st.download_button("⬇️ Download segmented customers (CSV)",
                       view.to_csv().encode(), "customer_segments.csv", "text/csv")
    cid = st.number_input("Look up a CustomerID", min_value=0, step=1, value=0)
    if cid in seg.index:
        st.write(seg.loc[cid, ["Recency", "Frequency", "Monetary", "Segment"]])
    elif cid:
        st.warning("CustomerID not found.")
