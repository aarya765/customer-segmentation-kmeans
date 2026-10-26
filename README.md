# E-Commerce Customer Segmentation Using K-Means Clustering

ADS Mini Project · Semester VII · Group 17

Groups customers of an online retailer by purchasing behaviour (**Recency, Frequency, Monetary value**) using **K-Means**, so a business can identify champions, loyal customers, occasional buyers and customers at risk of churning. Results are explored in an interactive **Streamlit** dashboard.

## Dataset
[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) (541,909 transactions, UK online retailer, CC BY 4.0).
Download `Online Retail.xlsx` and place it in `data/raw/`. It is git-ignored because it is large.
A small synthetic file (`data/sample/sample_online_retail.csv`) with the same columns is included so the project runs without the download.

## Workflow
1. **Cleaning** (`src/cleaning.py`): drop missing CustomerID, cancelled invoices, non-positive quantity/price, duplicates; add `TotalPrice`
2. **RFM features** (`src/rfm.py`): Recency (days since last order), Frequency (distinct invoices), Monetary (total spend)
3. **Scaling**: `log1p` (RFM is right-skewed) + `StandardScaler`
4. **Choosing k**: elbow method + silhouette score (`src/clustering.py`)
5. **K-Means** and rule-based segment naming (Champions, Loyal, Occasional, At-Risk/Lost, ...)
6. **Dashboard** (`app/dashboard.py`)

## Project structure
```
customer-segmentation/
├── app/dashboard.py            # Streamlit app
├── data/{raw,processed,sample}/
├── notebooks/01_eda_and_rfm.ipynb
├── reports/                    # generated charts + segment_summary.csv
├── src/                        # cleaning, rfm, clustering, sample-data generator
├── run_pipeline.py             # end-to-end script (no dashboard)
├── requirements.txt
└── README.md
```

## Setup
```powershell
git clone https://github.com/<your-username>/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run
```powershell
python run_pipeline.py                 # clean -> RFM -> K-Means, saves charts to reports/
streamlit run app/dashboard.py         # interactive dashboard
jupyter notebook notebooks/            # step-by-step EDA
```
`run_pipeline.py` options: `python run_pipeline.py "path/to/file.xlsx" --k 5`

## Results
Add screenshots of the dashboard to `reports/` and describe your segment findings here
(e.g. "Champions are ~X% of customers but generate ~Y% of revenue").

## Team (Group 17)
- Name 1
- Name 2
- Name 3

## Tech stack
Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit · Matplotlib/Seaborn
