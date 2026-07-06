# SmartCart Customer Segmentation

My second end-to-end ML project — an unsupervised clustering system that segments e-commerce customers into behavioral groups from raw, messy transactional data, replacing SmartCart's generic one-size-fits-all marketing.

Built the full pipeline from scratch and then debugged it independently: fixing a target-leakage bug, replacing deprecated pandas patterns, and validating cluster count choice with two separate metrics before committing to a final model.

## Problem Statement
SmartCart, a multi-country e-commerce platform, used identical marketing and engagement strategies for all customers — with no understanding of distinct behavior patterns. This led to inefficient marketing, missed high-value customer retention, and delayed churn detection. Goal: group customers into meaningful segments using unsupervised ML, based on purchasing behavior, engagement, and loyalty indicators.

## Dataset
- 2240 customers, 22 raw features (demographics, spend by category, purchase channel, engagement, feedback)
- Missing values in Income (handled via median imputation)
- No labels — fully unsupervised problem

## Approach
- **EDA** — pairplots on Income, Recency, Response, Age, Total Spending, Total Children to spot outliers and relationships; correlation heatmap across engineered features
- **Preprocessing** — median imputation on Income; outlier removal (Age < 90, Income < 600k)
- **Feature engineering** — Age (from Year_Birth), Customer Tenure Days (from Dt_Customer), Total Spending (sum of 6 category spends), Total Children (Kidhome + Teenhome), simplified Education tiers (Undergraduate/Graduate/Postgraduate), simplified Marital status (Partner/Alone)
- **Encoding** — One-Hot Encoding on Education and Living_With
- **Scaling** — StandardScaler on all features
- **Dimensionality reduction** — PCA to 3 components for clustering and 3D visualization
- **Cluster count selection** — Elbow Method (KneeLocator) cross-checked against Silhouette Score across k=2–10, rather than picking k on a single metric
- **Modeling** — trained and compared K-Means and Agglomerative Clustering (Ward linkage), k=4
- **Cluster profiling** — mean feature summary per cluster, visualized via count plot and Income-vs-Spending scatter by cluster

## Key Fixes & Learnings
This project went through a real debugging pass, not just a first-draft run:
- **Target leakage fixed** — `Response` (a campaign-outcome column) was initially flowing into the scaled/PCA'd feature matrix used for clustering. Removed before PCA/scaling since it's an outcome label, not a segmentation feature.
- **Deprecated pandas patterns removed** — `.replace()` for category remapping (Education, Marital_Status) replaced with `.map()`.
- **Misleading metric print fixed** — silhouette score loop was printing the *last* k's score instead of the best one; now reports the actual best k and its score.
- **Two-metric validation** — cluster count chosen using Elbow + Silhouette together, not a single heuristic.
- **Model comparison** — K-Means and Agglomerative both run and visually compared in 3D PCA space before selecting cluster assignments for profiling.

## Tech Stack
Python, pandas, scikit-learn, seaborn, matplotlib, kneed

## Project Structure
```
Smart_card.ipynb              # full pipeline: EDA → preprocessing → feature engineering → encoding → scaling → PCA → clustering → profiling
smartcart_customers.csv       # dataset (gitignored)
README.md
```

## How to Run
```
git clone https://github.com/Devanshyelne/smartcart-customer-segmentation.git
cd smartcart-customer-segmentation
pip install pandas scikit-learn seaborn matplotlib kneed
jupyter notebook Smart_card.ipynb
```

## Future Work
- Streamlit dashboard for interactive cluster exploration
- Quantitative comparison (silhouette/Davies-Bouldin) between K-Means and Agglomerative instead of visual-only comparison
- Cluster-based marketing recommendations per segment
- Automated cluster naming/labeling based on dominant traits

Built by Devansh Yelne — 2nd year AI & ML student, learning ML by building and breaking things properly.
