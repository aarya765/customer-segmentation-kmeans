"""Load and clean the UCI Online Retail transactions."""
from pathlib import Path
import pandas as pd

COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity",
           "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


def load_raw(source, name=None) -> pd.DataFrame:
    """Read the raw file. `source` can be a path or a file-like object
    (e.g. a Streamlit upload); `name` is used to detect the extension."""
    name = str(name or getattr(source, "name", source))
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(source, engine="openpyxl")
    else:
        df = pd.read_csv(source, encoding="ISO-8859-1")
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df[COLUMNS]


def clean(df: pd.DataFrame):
    """Return (clean_df, report). `report` lists how many rows each step left."""
    report = {"raw_rows": len(df)}
    df = df.copy()

    df = df.dropna(subset=["CustomerID"])
    report["after_dropping_missing_customer_id"] = len(df)

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]          # cancelled orders
    report["after_dropping_cancellations"] = len(df)

    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    report["after_dropping_nonpositive_qty_price"] = len(df)

    df = df.drop_duplicates()
    report["after_dropping_duplicates"] = len(df)

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    report["clean_rows"] = len(df)
    report["customers"] = df["CustomerID"].nunique()
    return df.reset_index(drop=True), report


def save_clean(df: pd.DataFrame, path="data/processed/clean_transactions.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
