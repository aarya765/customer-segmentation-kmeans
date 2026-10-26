"""Create a small synthetic dataset with the same columns as UCI Online Retail.
Used for quick tests, demos and Streamlit deployment (the real file is gitignored).
Run:  python -m src.generate_sample_data
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/sample/sample_online_retail.csv")
START, END = pd.Timestamp("2010-12-01"), pd.Timestamp("2011-12-09")
# (customers, orders range, price scale, min/max days since last order)
ARCHETYPES = [(120, (14, 30), 3.0, 1, 30),
              (300, (5, 12), 1.8, 5, 90),
              (600, (1, 3), 1.0, 20, 300),
              (400, (1, 2), 0.8, 200, 370)]
COUNTRIES = ["United Kingdom"] * 14 + ["Germany", "France", "EIRE", "Spain", "Netherlands"]


def generate(seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, cust, inv = [], 12346, 536365
    for n_cust, (o_lo, o_hi), scale, r_lo, r_hi in ARCHETYPES:
        for _ in range(n_cust):
            cust += int(rng.integers(1, 4))
            country = str(rng.choice(COUNTRIES))
            last = END - pd.Timedelta(days=int(rng.integers(r_lo, r_hi)))
            for _ in range(int(rng.integers(o_lo, o_hi + 1))):
                when = max(START, last - pd.Timedelta(days=int(rng.integers(0, 330)),
                                                      hours=int(rng.integers(0, 9))))
                inv += 1
                for _ in range(int(rng.integers(1, 6))):
                    rows.append([str(inv), str(rng.integers(10000, 10400)), "SAMPLE ITEM",
                                 int(rng.integers(1, 12)), when,
                                 round(float(rng.gamma(2, 1.2 * scale)) + 0.5, 2), cust, country])
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    # a little noise so the cleaning step has something to do
    canc = df.sample(frac=0.02, random_state=1).copy()
    canc["InvoiceNo"] = "C" + canc["InvoiceNo"]; canc["Quantity"] *= -1
    anon = df.sample(frac=0.02, random_state=2).copy(); anon["CustomerID"] = np.nan
    return pd.concat([df, canc, anon]).sort_values("InvoiceDate").reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = generate(); d.to_csv(OUT, index=False)
    print(f"Wrote {len(d):,} rows to {OUT}")
