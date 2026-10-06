# 🛒 SmartCart AI — Customer Segmentation Dashboard

A Streamlit dashboard that turns the SmartCart customer-segmentation notebook (`Smart_card.ipynb`) into an interactive analytics app.

## Problem Statement
E-commerce companies often treat all customers the same. This project groups customers by purchasing behavior, spending and engagement so that **targeted marketing strategies** can be designed for each segment.

## Features
- Upload a CSV, or use the built-in demo dataset automatically
- Dataset summary and visual processing pipeline
- Interactive charts (segment sizes, income vs spending, spending and recency distributions)
- K-Means and Agglomerative clustering with adjustable k (2–10)
- Silhouette and Davies-Bouldin evaluation with automatic model recommendation
- 3D PCA cluster visualization
- Data-driven segment names and marketing recommendations
- Customer Explorer (assigns a new customer to a segment using the trained pipeline)
- CSV export of clustered customers, segment profiles and recommendations

## ML Pipeline
Raw data → median imputation → outlier removal → feature engineering (Age, Customer_Tenure, Total_Spending, Total_Children, Education_Level, Living_With_Partner) → scaling (StandardScaler) → PCA (3 components) → K-Means / Agglomerative (ward) → evaluation → profiling.

**Leakage fix:** `Response` and `AcceptedCmp1–5` are dropped before scaling, PCA and clustering.

## Tech Stack
Python, Streamlit, pandas, NumPy, scikit-learn, Plotly

## Project Structure
```
smartcart-customer-segmentation/
├── app.py
├── requirements.txt
├── README.md
├── Smart_card.ipynb        (original notebook, untouched)
└── .streamlit/config.toml
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
Push to GitHub → share.streamlit.io → New app → select repo, branch `main`, main file `app.py` → Deploy.

## Expected CSV Columns
Required: `Income, Year_Birth, Recency, NumWebPurchases, NumStorePurchases` and at least one `Mnt*` column. Optional: `Dt_Customer, Kidhome, Teenhome, Education, Marital_Status, NumWebVisitsMonth`.

## Future Improvements
Automatic k selection (elbow/silhouette sweep), DBSCAN/GMM comparison, RFM scoring, campaign-uplift analysis.
