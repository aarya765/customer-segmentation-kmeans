"""Streamlit dashboard: E-Commerce Customer Segmentation (K-Means on RFM).
Run:  streamlit run app/dashboard.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cleaning import clean, load_raw
from src.clustering import (add_pca, evaluate_k, fit_kmeans, name_segments,
                            prepare_features, segment_summary)
from src.forecasting import daily_revenue, run_forecasting
from src.rfm import build_rfm
from src.supervised import build_customer_table, run_classification, run_regression

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


@st.cache_data(show_spinner="Training churn and spend models...")
def run_supervised(df: pd.DataFrame):
    table, cutoff = build_customer_table(df)
    return table, cutoff, run_classification(table), run_regression(table)


@st.cache_data(show_spinner="Fitting forecasting models...")
def run_fc(df: pd.DataFrame):
    return run_forecasting(daily_revenue(df))


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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📊 Overview", "📐 Choosing k", "🎯 Segments", "🔎 Customers",
     "🧪 Churn (classification)", "💷 Spend (regression)", "📈 Forecast"])

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


# ---------------- classification: churn ----------------
with tab5:
    table, cutoff, clf, reg = run_supervised(df)
    st.subheader("Will a customer buy again in the next 90 days?")
    st.caption(f"Features use transactions up to {cutoff.date()} only; the label is whether the customer "
               "made no purchase in the following 90 days (churned = 1). No future information leaks into the features.")
    n0, n1 = clf["counts"].get(0, 0), clf["counts"].get(1, 0)
    a, b, c_ = st.columns(3)
    a.metric("Active customers", f"{n0:,}")
    b.metric("Churned customers", f"{n1:,}")
    c_.metric("Churn rate", f"{clf['churn_rate']:.1%}")
    st.info("Class balance check: the classes are mildly imbalanced, so models use class_weight='balanced', the split is "
            "stratified, and accuracy is not used alone (precision, recall, F1 and ROC-AUC are reported).")
    rows = [{"Model": k, **{m_: round(v[m_], 3) for m_ in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
             "CV AUC (5-fold)": f"{v['cv_auc_mean']:.3f} ± {v['cv_auc_std']:.3f}"} for k, v in clf["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    left, right = st.columns(2)
    cm = clf["models"][clf["best"]]["confusion"]
    left.plotly_chart(px.imshow(cm, text_auto=True, x=["Active", "Churned"], y=["Active", "Churned"],
                                labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues",
                                title=f"Confusion matrix: {clf['best']}"), width="stretch")
    roc = go.Figure()
    for k_, (fpr, tpr) in clf["roc"].items():
        roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{k_} (AUC {clf['models'][k_]['roc_auc']:.2f})")
    roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance")
    roc.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate")
    right.plotly_chart(roc, width="stretch")
    imp = pd.Series(clf["importance"]).sort_values().reset_index()
    imp.columns = ["Feature", "Importance"]
    st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Random Forest feature importance"),
                    width="stretch")

# ---------------- regression: next-90-day spend ----------------
with tab6:
    st.subheader("How much will a customer spend in the next 90 days?")
    st.caption("Same leakage-free design as the churn model. Target: spend in the 90 days after the cutoff "
               "(zero for customers who do not buy). Spend is heavy-tailed, so models are trained on log(1 + spend).")
    rows = [{"Model": k, "MAE (£)": round(v["mae"], 1), "RMSE (£)": round(v["rmse"], 1), "R²": round(v["r2"], 3),
             "Median AE (£)": round(v["median_ae"], 1), "R² (log scale)": round(v["r2_log"], 3)} for k, v in reg["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Best model: {reg['best']}. R² on the £ scale is modest because a few very large customers dominate the error.")
    act = np.array(reg["actual"]); prd = np.array(reg["pred"][reg["best"]])
    l, r = st.columns(2)
    l.plotly_chart(px.scatter(x=np.log1p(act), y=np.log1p(prd), opacity=0.5,
                              labels=dict(x="log(1 + actual spend)", y="log(1 + predicted spend)"),
                              title="Actual vs predicted (log scale)"), width="stretch")
    r.plotly_chart(px.histogram(x=np.log1p(prd) - np.log1p(act), nbins=40, labels=dict(x="predicted - actual (log)"),
                                title="Residuals (log scale)"), width="stretch")

# ---------------- time series: revenue forecast ----------------
with tab7:
    fc = run_fc(df)
    st.subheader("Daily revenue forecast")
    st.caption(f"Chronological split: train {fc['train_start']} to {fc['train_end']} ({fc['n_train']} days), "
               f"test {fc['test_start']} to {fc['test_end']} ({fc['n_test']} days). MAPE is computed on days with sales.")
    rows = [{"Method": k, "MAE (£)": round(v["mae"]), "RMSE (£)": round(v["rmse"]), "MAPE (%)": round(v["mape"], 1)}
            for k, v in fc["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pick = st.selectbox("Forecast to plot", list(fc["pred"].keys()), index=list(fc["pred"].keys()).index(fc["best"]))
    dates = pd.to_datetime(fc["dates"]); n0 = fc["n_train"]
    fig = go.Figure()
    fig.add_scatter(x=dates[:n0], y=fc["actual"][:n0], name="Train (actual)", line=dict(color="gray"))
    fig.add_scatter(x=dates[n0:], y=fc["actual"][n0:], name="Test (actual)", line=dict(color="black"))
    fig.add_scatter(x=dates[n0:], y=fc["pred"][pick], name=f"Forecast: {pick}", line=dict(color="crimson"))
    fig.update_layout(title="Daily revenue: actual vs forecast", yaxis_title="Revenue (£)")
    st.plotly_chart(fig, width="stretch")
