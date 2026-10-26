"""RFM (Recency, Frequency, Monetary) feature engineering."""
import pandas as pd


def build_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """One row per customer.
    Recency   = days since last purchase (smaller = more recent)
    Frequency = number of distinct invoices
    Monetary  = total money spent
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    return rfm.round({"Monetary": 2})
