"""SmartCart AI - Customer Segmentation & Marketing Intelligence."""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartCart AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

CORE = [
    "Income",
    "Year_Birth",
    "Recency",
    "NumWebPurchases",
    "NumStorePurchases",
]

MNT = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

LEAKAGE = [
    "Response",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]

FEATURES = [
    "Income",
    "Age",
    "Recency",
    "Total_Spending",
    "Total_Children",
    "Customer_Tenure",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Education_Level",
    "Living_With_Partner",
]

PROFILE = [
    "Income",
    "Recency",
    "Total_Spending",
    "Total_Children",
    "Age",
    "Customer_Tenure",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
]

EDU = {
    "Basic": 0,
    "2n Cycle": 0,
    "Graduation": 1,
    "Master": 2,
    "PhD": 2,
}

PALETTE = [
    "#3B82F6",
    "#8B5CF6",
    "#14B8A6",
    "#F59E0B",
    "#EF4444",
    "#EC4899",
    "#06B6D4",
    "#10B981",
    "#F97316",
    "#A855F7",
]

SEGMENTS = {
    "High-Value": {
        "name": "💎 High-Value Customers",
        "description": (
            "Customers with strong purchasing power and high spending."
        ),
        "strategy": (
            "Offer loyalty rewards, premium products and exclusive "
            "early-access campaigns."
        ),
        "action": "Offer premium loyalty benefits.",
    },
    "Growth": {
        "name": "📈 Growth Customers",
        "description": (
            "Customers with moderate spending and strong potential "
            "to increase their spending."
        ),
        "strategy": (
            "Use bundles, product recommendations and cross-selling."
        ),
        "action": "Push bundles and cross-sell offers.",
    },
    "At-Risk": {
        "name": "⚠️ At-Risk Customers",
        "description": (
            "Customers who have not purchased recently."
        ),
        "strategy": (
            "Use personalized discounts, reminder campaigns and "
            "re-engagement offers."
        ),
        "action": "Send a personalized win-back discount.",
    },
    "Regular": {
        "name": "🛍️ Regular Customers",
        "description": (
            "Customers with steady, lower-value shopping behavior."
        ),
        "strategy": (
            "Use seasonal promotions, loyalty points and low-cost "
            "upsell nudges."
        ),
        "action": "Run seasonal promotions and points programs.",
    },
}


# ============================================================
# ERROR TYPE
# ============================================================

class DataError(Exception):
    """Application-specific data error."""
    pass


# ============================================================
# DEMO DATA
# ============================================================

@st.cache_data(show_spinner=False)
def demo_data(n=1200, seed=42):
    """Generate SmartCart-style demonstration customer data."""

    rng = np.random.default_rng(seed)

    groups = rng.choice(
        4,
        n,
        p=[0.22, 0.28, 0.30, 0.20],
    )

    income_mean = np.array(
        [78000, 55000, 36000, 46000]
    )

    spend_mean = np.array(
        [1400, 650, 160, 320]
    )

    recency_mean = np.array(
        [22, 30, 38, 78]
    )

    children_mean = np.array(
        [0.3, 1.2, 1.5, 1.0]
    )

    web_mean = np.array(
        [5, 6, 3, 2]
    )

    store_mean = np.array(
        [8, 6, 4, 3]
    )

    visit_mean = np.array(
        [4, 6, 7, 5]
    )

    spending = np.clip(
        rng.normal(
            spend_mean[groups],
            spend_mean[groups] * 0.25,
        ),
        10,
        None,
    )

    category_spending = (
        spending[:, None]
        * rng.dirichlet(
            [6, 1, 4, 1.5, 1, 1.5],
            n,
        )
    ).round().astype(int)

    age = np.clip(
        rng.normal(48, 11, n),
        24,
        80,
    ).round().astype(int)

    dates = (
        pd.Timestamp("2012-07-30")
        + pd.to_timedelta(
            rng.integers(0, 700, n),
            unit="D",
        )
    )

    df = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),

            "Year_Birth": 2014 - age,

            "Education": rng.choice(
                [
                    "Graduation",
                    "PhD",
                    "Master",
                    "Basic",
                    "2n Cycle",
                ],
                n,
                p=[
                    0.50,
                    0.20,
                    0.17,
                    0.03,
                    0.10,
                ],
            ),

            "Marital_Status": rng.choice(
                [
                    "Married",
                    "Together",
                    "Single",
                    "Divorced",
                    "Widow",
                ],
                n,
                p=[
                    0.38,
                    0.26,
                    0.22,
                    0.11,
                    0.03,
                ],
            ),

            "Income": np.clip(
                rng.normal(
                    income_mean[groups],
                    8000,
                ),
                8000,
                None,
            ).round(),

            "Kidhome": rng.binomial(
                2,
                children_mean[groups] / 4,
            ),

            "Teenhome": rng.binomial(
                2,
                children_mean[groups] / 4,
            ),

            "Dt_Customer": dates.strftime(
                "%d-%m-%Y"
            ),

            "Recency": np.clip(
                rng.normal(
                    recency_mean[groups],
                    12,
                ),
                0,
                99,
            ).round().astype(int),
        }
    )

    for column_name, values in zip(
        MNT,
        category_spending.T,
    ):
        df[column_name] = values

    df["NumWebPurchases"] = np.clip(
        rng.poisson(
            web_mean[groups]
        ),
        0,
        27,
    )

    df["NumStorePurchases"] = np.clip(
        rng.poisson(
            store_mean[groups]
        ),
        0,
        13,
    )

    df["NumWebVisitsMonth"] = np.clip(
        rng.poisson(
            visit_mean[groups]
        ),
        0,
        20,
    )

    df["AcceptedCmp1"] = rng.binomial(
        1,
        0.07,
        n,
    )

    df["Response"] = rng.binomial(
        1,
        0.15,
        n,
    )

    missing_indices = rng.choice(
        n,
        20,
        replace=False,
    )

    df.loc[
        missing_indices,
        "Income",
    ] = np.nan

    return df


# ============================================================
# READ CSV
# ============================================================

def read_csv(file):
    """Read CSV/TXT with automatic separator detection."""

    try:
        df = pd.read_csv(
            file,
            sep=None,
            engine="python",
        )
    except Exception as exc:
        raise DataError(
            f"The uploaded file could not be read: {exc}"
        )

    if df.empty:
        raise DataError(
            "The uploaded file is empty."
        )

    return df


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(raw):
    """Clean data and create ML-ready features."""

    df = raw.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    spending_columns = [
        column
        for column in MNT
        if column in df.columns
    ]

    missing_core = [
        column
        for column in CORE
        if column not in df.columns
    ]

    if missing_core or not spending_columns:

        missing = list(missing_core)

        if not spending_columns:
            missing.append(
                "at least one Mnt* spending column"
            )

        raise DataError(
            "Your dataset does not contain enough compatible "
            "customer features for segmentation. "
            f"Missing: {', '.join(missing)}."
        )

    # Remove target leakage
    leakage_columns = [
        column
        for column in LEAKAGE
        if column in df.columns
    ]

    df = df.drop(
        columns=leakage_columns,
        errors="ignore",
    )

    numeric_columns = [
        column
        for column in (
            CORE
            + spending_columns
            + [
                "Kidhome",
                "Teenhome",
                "NumWebVisitsMonth",
            ]
        )
        if column in df.columns
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    missing_count = int(
        df[numeric_columns]
        .isna()
        .sum()
        .sum()
    )

    reference_year = pd.Timestamp.now().year

    if "Dt_Customer" in df.columns:

        dates = pd.to_datetime(
            df["Dt_Customer"],
            dayfirst=True,
            errors="coerce",
        )

        if dates.notna().any():

            df["Customer_Tenure"] = (
                dates.max() - dates
            ).dt.days

            reference_year = int(
                dates.max().year
            )

            if "Customer_Tenure" not in numeric_columns:
                numeric_columns.append(
                    "Customer_Tenure"
                )

    # Median imputation
    for column in numeric_columns:

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0

        df[column] = (
            df[column]
            .fillna(median_value)
            .fillna(0)
        )

    # Feature engineering

    df["Age"] = (
        reference_year
        - df["Year_Birth"]
    )

    df["Total_Spending"] = df[
        spending_columns
    ].sum(axis=1)

    child_columns = [
        column
        for column in [
            "Kidhome",
            "Teenhome",
        ]
        if column in df.columns
    ]

    if child_columns:

        df["Total_Children"] = df[
            child_columns
        ].sum(axis=1)

    else:

        df["Total_Children"] = 0

    if "Education" in df.columns:

        df["Education_Level"] = (
            df["Education"]
            .map(EDU)
            .fillna(1)
        )

    else:

        df["Education_Level"] = 1

    if "Marital_Status" in df.columns:

        df["Living_With_Partner"] = (
            df["Marital_Status"]
            .isin(
                [
                    "Married",
                    "Together",
                ]
            )
            .astype(int)
        )

    else:

        df["Living_With_Partner"] = 0

    # Outlier filtering
    rows_before = len(df)

    df = df[
        (df["Age"] <= 90)
        & (df["Age"] >= 18)
        & (df["Income"] <= 600000)
        & (df["Income"] >= 0)
    ].reset_index(drop=True)

    features = [
        column
        for column in FEATURES
        if column in df.columns
    ]

    info = {
        "rows_raw": len(raw),
        "rows_clean": len(df),
        "outliers": rows_before - len(df),
        "imputed": missing_count,
        "leak": leakage_columns,
        "features": features,
    }

    return (
        df,
        features,
        info,
    )


# ============================================================
# SCALING + PCA
# ============================================================

@st.cache_data(show_spinner=False)
def build_space(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=3,
        random_state=42,
    )

    Z = pca.fit_transform(
        X_scaled
    )

    return (
        scaler,
        pca,
        Z,
    )


# ============================================================
# CLUSTERING
# ============================================================

@st.cache_data(show_spinner=False)
def run_models(Z, k):

    kmeans = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42,
    )

    kmeans.fit(Z)

    km_labels = kmeans.labels_

    agglomerative = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )

    ag_labels = (
        agglomerative
        .fit_predict(Z)
    )

    return {
        "km": kmeans,
        "km_labels": km_labels,
        "ag_labels": ag_labels,

        "km_sil": silhouette_score(
            Z,
            km_labels,
        ),

        "km_db": davies_bouldin_score(
            Z,
            km_labels,
        ),

        "ag_sil": silhouette_score(
            Z,
            ag_labels,
        ),

        "ag_db": davies_bouldin_score(
            Z,
            ag_labels,
        ),
    }


# ============================================================
# CLUSTER PROFILE
# ============================================================

def profile(
    df,
    labels,
):

    temp = df.copy()

    temp["Cluster"] = labels

    available_columns = [
        column
        for column in PROFILE
        if column in temp.columns
    ]

    result = (
        temp
        .groupby("Cluster")[available_columns]
        .mean()
    )

    result.insert(
        0,
        "Customers",
        temp
        .groupby("Cluster")
        .size(),
    )

    return result


# ============================================================
# AUTOMATIC SEGMENT NAMES
# ============================================================

def segment_labels(
    profile_data
):

    def z_score(series):

        standard_deviation = series.std(
            ddof=0
        )

        if standard_deviation == 0:
            return series - series.mean()

        return (
            series - series.mean()
        ) / standard_deviation

    value_score = (
        z_score(
            profile_data["Income"]
        )
        + z_score(
            profile_data["Total_Spending"]
        )
    )

    keys = {
        value_score.idxmax(): "High-Value"
    }

    remaining = [
        cluster
        for cluster in profile_data.index
        if cluster not in keys
    ]

    if remaining:

        at_risk_cluster = (
            profile_data
            .loc[
                remaining,
                "Recency",
            ]
            .idxmax()
        )

        keys[
            at_risk_cluster
        ] = "At-Risk"

        median_spending = (
            profile_data[
                "Total_Spending"
            ]
            .median()
        )

        for cluster in remaining:

            if cluster in keys:
                continue

            cluster_spending = (
                profile_data
                .loc[
                    cluster,
                    "Total_Spending",
                ]
            )

            if cluster_spending >= median_spending:
                keys[cluster] = "Growth"
            else:
                keys[cluster] = "Regular"

    names = {}

    for cluster in profile_data.index:

        names[cluster] = SEGMENTS[
            keys[cluster]
        ]["name"]

    return (
        keys,
        names,
    )


# ============================================================
# PLOTLY DARK STYLING
# ============================================================

def style_plot(
    fig,
    height=None,
):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#CBD5E1"
        ),
        title=dict(
            font=dict(
                color="#F8FAFC"
            )
        ),
        legend=dict(
            font=dict(
                color="#CBD5E1"
            )
        ),
        margin=dict(
            l=10,
            r=10,
            t=60,
            b=10,
        ),
    )

    if height is not None:
        fig.update_layout(
            height=height
        )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "⚙️ Dashboard Controls"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=[
            "csv",
            "txt",
        ],
    )

    cluster_count = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=4,
    )

    selected_model = st.radio(
        "Clustering model",
        [
            "K-Means",
            "Agglomerative Clustering",
        ],
    )

    show_raw_data = st.checkbox(
        "Show raw data"
    )

    show_preprocessing = st.checkbox(
        "Show preprocessing details"
    )

    st.divider()

    st.caption(
        "No CSV uploaded → the built-in demo "
        "dataset is used automatically."
    )


# ============================================================
# HERO
# ============================================================

st.title(
    "🛒 SmartCart AI"
)

st.subheader(
    "Customer Segmentation & Marketing Intelligence"
)

st.write(
    "Use machine learning to discover meaningful customer "
    "groups from purchasing behavior, spending patterns "
    "and engagement."
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

try:

    # Uploaded file gets priority
    if uploaded_file is not None:

        raw_data = read_csv(
            uploaded_file
        )

        source = (
            f"Uploaded file: "
            f"{uploaded_file.name}"
        )

    # Optional local CSV
    elif os.path.exists(
        "smartcart_customers.csv"
    ):

        raw_data = pd.read_csv(
            "smartcart_customers.csv"
        )

        source = (
            "Local dataset: "
            "smartcart_customers.csv"
        )

    # Demo data
    else:

        raw_data = demo_data()

        source = (
            "Built-in demo dataset"
        )

    clean_data, features, info = (
        preprocess(
            raw_data
        )
    )

    minimum_rows = max(
        30,
        5 * cluster_count,
    )

    if len(clean_data) < minimum_rows:

        raise DataError(
            f"Only {len(clean_data)} usable rows "
            f"remain after cleaning. "
            f"At least {minimum_rows} are recommended "
            f"for {cluster_count} clusters."
        )

    X = clean_data[
        features
    ].astype(float)

    scaler, pca, Z = build_space(
        X
    )

    results = run_models(
        Z,
        cluster_count,
    )

except DataError as error:

    st.error(
        str(error)
    )

    st.stop()

except Exception as error:

    st.error(
        f"Application error: {error}"
    )

    st.stop()


# ============================================================
# CURRENT MODEL
# ============================================================

using_kmeans = (
    selected_model == "K-Means"
)

labels = (
    results["km_labels"]
    if using_kmeans
    else results["ag_labels"]
)


# ============================================================
# SEGMENTS
# ============================================================

profile_data = profile(
    clean_data,
    labels,
)

segment_keys, segment_names = (
    segment_labels(
        profile_data
    )
)

kmeans_profile = profile(
    clean_data,
    results["km_labels"],
)

kmeans_keys, kmeans_names = (
    segment_labels(
        kmeans_profile
    )
)

clean_data["Cluster"] = labels

clean_data["Segment"] = (
    clean_data[
        "Cluster"
    ]
    .map(segment_names)
)


# ============================================================
# KPI SECTION
# ============================================================

best_silhouette = max(
    results["km_sil"],
    results["ag_sil"],
)

kpi1, kpi2, kpi3, kpi4 = (
    st.columns(4)
)

with kpi1:
    st.metric(
        "Total Customers",
        f"{len(clean_data):,}",
    )

with kpi2:
    st.metric(
        "Features Used",
        len(features),
    )

with kpi3:
    st.metric(
        "Number of Segments",
        cluster_count,
    )

with kpi4:
    st.metric(
        "Best Silhouette Score",
        f"{best_silhouette:.3f}",
    )


st.write("")


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📁 Dataset",
        "📊 Overview",
        "🤖 Segmentation",
        "👥 Customer Profiles",
        "🔎 Customer Explorer",
        "⬇️ Export",
    ]
)


# ============================================================
# DATASET TAB
# ============================================================

with tabs[0]:

    st.header(
        "📁 Dataset"
    )

    st.caption(
        f"Source: {source}"
    )

    m1, m2, m3, m4, m5 = (
        st.columns(5)
    )

    with m1:
        st.metric(
            "Rows",
            f"{len(raw_data):,}",
        )

    with m2:
        st.metric(
            "Columns",
            raw_data.shape[1],
        )

    with m3:
        st.metric(
            "Missing Values",
            f"{int(
                raw_data.isna()
                .sum()
                .sum()
            ):,}",
        )

    with m4:
        st.metric(
            "Numerical Columns",
            raw_data.select_dtypes(
                include="number"
            ).shape[1],
        )

    with m5:
        st.metric(
            "Categorical Columns",
            (
                raw_data.shape[1]
                -
                raw_data.select_dtypes(
                    include="number"
                ).shape[1]
            ),
        )

    st.write("")

    if show_raw_data:

        st.dataframe(
            raw_data,
            use_container_width=True,
        )

    else:

        st.dataframe(
            raw_data.head(10),
            use_container_width=True,
        )

    st.write("")

    st.subheader(
        "Data Processing Pipeline"
    )

    st.info(
        "Raw Data  ➜  "
        "Missing Value Handling  ➜  "
        "Outlier Removal  ➜  "
        "Feature Engineering  ➜  "
        "Encoding  ➜  "
        "Scaling  ➜  "
        "PCA  ➜  "
        "Clustering"
    )

    if show_preprocessing:

        with st.expander(
            "Preprocessing Details",
            expanded=True,
        ):

            st.write(
                f"**Raw rows:** "
                f"{info['rows_raw']:,}"
            )

            st.write(
                f"**Rows after cleaning:** "
                f"{info['rows_clean']:,}"
            )

            st.write(
                f"**Outliers removed:** "
                f"{info['outliers']:,}"
            )

            st.write(
                f"**Missing values imputed:** "
                f"{info['imputed']:,}"
            )

            if info["leak"]:

                st.write(
                    "**Leakage columns removed:** "
                    + ", ".join(
                        info["leak"]
                    )
                )

            else:

                st.write(
                    "**Leakage columns removed:** None"
                )

            st.write(
                "**Features used:** "
                + ", ".join(
                    features
                )
            )

            st.write(
                "**Scaling:** StandardScaler"
            )

            st.write(
                "**Dimensionality Reduction:** "
                "PCA with 3 components"
            )


# ============================================================
# OVERVIEW TAB
# ============================================================

with tabs[1]:

    st.header(
        "📊 Customer Overview"
    )

    left, right = st.columns(2)

    segment_counts = (
        clean_data
        .groupby("Segment")
        .size()
        .reset_index(
            name="Customers"
        )
    )

    distribution_fig = px.bar(
        segment_counts,
        x="Segment",
        y="Customers",
        color="Segment",
        color_discrete_sequence=PALETTE,
        title="Customer Distribution by Segment",
    )

    distribution_fig.update_layout(
        showlegend=False
    )

    left.plotly_chart(
        style_plot(
            distribution_fig
        ),
        use_container_width=True,
    )

    sample_data = clean_data.sample(
        min(
            len(clean_data),
            2000,
        ),
        random_state=1,
    )

    income_fig = px.scatter(
        sample_data,
        x="Income",
        y="Total_Spending",
        color="Segment",
        color_discrete_sequence=PALETTE,
        opacity=0.75,
        title="Income vs Total Spending",
        hover_data=[
            "Age",
            "Recency",
            "NumWebPurchases",
            "NumStorePurchases",
        ],
    )

    right.plotly_chart(
        style_plot(
            income_fig
        ),
        use_container_width=True,
    )

    left, right = st.columns(2)

    spending_fig = px.histogram(
        clean_data,
        x="Total_Spending",
        nbins=40,
        title="Spending Distribution",
    )

    left.plotly_chart(
        style_plot(
            spending_fig
        ),
        use_container_width=True,
    )

    recency_fig = px.histogram(
        clean_data,
        x="Recency",
        nbins=40,
        title="Recency Distribution",
    )

    right.plotly_chart(
        style_plot(
            recency_fig
        ),
        use_container_width=True,
    )


# ============================================================
# SEGMENTATION TAB
# ============================================================

with tabs[2]:

    st.header(
        "🤖 Model Evaluation"
    )

    if using_kmeans:

        silhouette = results[
            "km_sil"
        ]

        davies_bouldin = results[
            "km_db"
        ]

    else:

        silhouette = results[
            "ag_sil"
        ]

        davies_bouldin = results[
            "ag_db"
        ]

    e1, e2 = st.columns(2)

    with e1:
        st.metric(
            f"Silhouette Score ({selected_model})",
            f"{silhouette:.3f}",
        )

    with e2:
        st.metric(
            f"Davies-Bouldin Score ({selected_model})",
            f"{davies_bouldin:.3f}",
        )

    st.info(
        "Silhouette Score: higher is better. "
        "Davies-Bouldin Score: lower is better."
    )

    comparison = pd.DataFrame(
        {
            "Model": [
                "K-Means",
                "Agglomerative",
            ],
            "Silhouette": [
                round(
                    results["km_sil"],
                    4,
                ),
                round(
                    results["ag_sil"],
                    4,
                ),
            ],
            "Davies-Bouldin": [
                round(
                    results["km_db"],
                    4,
                ),
                round(
                    results["ag_db"],
                    4,
                ),
            ],
        }
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    if (
        results["km_sil"]
        >= results["ag_sil"]
    ):

        best_model = "K-Means"

    else:

        best_model = (
            "Agglomerative Clustering"
        )

    st.success(
        f"🏆 Recommended Model: {best_model} "
        f"(highest silhouette score for "
        f"k = {cluster_count})"
    )

    st.write("")

    st.subheader(
        "🌐 3D PCA Cluster Visualization"
    )

    pca_data = pd.DataFrame(
        Z,
        columns=[
            "PC1",
            "PC2",
            "PC3",
        ],
    )

    pca_data["Segment"] = (
        clean_data[
            "Segment"
        ].values
    )

    pca_fig = px.scatter_3d(
        pca_data,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Segment",
        color_discrete_sequence=PALETTE,
        opacity=0.82,
        title="3D PCA Customer Segmentation",
    )

    pca_fig.update_traces(
        marker_size=4
    )

    st.plotly_chart(
        style_plot(
            pca_fig,
            height=650,
        ),
        use_container_width=True,
    )

    explained_variance = (
        pca
        .explained_variance_ratio_
        .sum()
        * 100
    )

    st.info(
        f"PCA explained variance: "
        f"{explained_variance:.1f}% "
        f"(3 components) · "
        f"Model shown: {selected_model}"
    )


# ============================================================
# CUSTOMER PROFILES TAB
# ============================================================

with tabs[3]:

    st.header(
        "👥 Customer Profiles"
    )

    displayed_profile = (
        profile_data
        .rename(
            index=segment_names
        )
    )

    displayed_profile.index.name = (
        "Segment"
    )

    st.dataframe(
        displayed_profile.round(1),
        use_container_width=True,
    )

    st.write("")

    st.header(
        "🎯 Marketing Recommendations"
    )

    recommendation_columns = (
        st.columns(2)
    )

    recommendations = []

    for index, cluster in enumerate(
        profile_data.index
    ):

        segment_type = segment_keys[
            cluster
        ]

        segment_info = SEGMENTS[
            segment_type
        ]

        row = profile_data.loc[
            cluster
        ]

        if (
            row["NumWebPurchases"]
            >
            row["NumStorePurchases"]
        ):

            preferred_channel = "web"

        else:

            preferred_channel = (
                "in-store"
            )

        metadata = (
            f"{int(row['Customers']):,} customers · "
            f"Avg income ₹{row['Income']:,.0f} · "
            f"Avg spend ₹{row['Total_Spending']:,.0f} · "
            f"Recency {row['Recency']:.0f} days · "
            f"Prefers {preferred_channel} purchases"
        )

        recommendations.append(
            {
                "Segment": segment_info[
                    "name"
                ],
                "Customers": int(
                    row["Customers"]
                ),
                "Description": segment_info[
                    "description"
                ],
                "Recommended Strategy": (
                    segment_info["strategy"]
                    + " "
                    + f"Prioritise the "
                    f"{preferred_channel} channel."
                ),
            }
        )

        with recommendation_columns[
            index % 2
        ]:

            with st.container(
                border=True
            ):

                st.subheader(
                    segment_info["name"]
                )

                st.write(
                    segment_info[
                        "description"
                    ]
                )

                st.markdown(
                    "**Recommendation:** "
                    + segment_info[
                        "strategy"
                    ]
                )

                st.caption(
                    f"Prioritise the "
                    f"**{preferred_channel}** channel."
                )

                st.caption(
                    metadata
                )


# ============================================================
# CUSTOMER EXPLORER TAB
# ============================================================

with tabs[4]:

    st.header(
        "🔎 Customer Explorer"
    )

    st.write(
        "Enter customer characteristics and predict "
        "their likely segment using the trained "
        "K-Means model."
    )

    left, right = st.columns(2)

    with left:

        income_input = st.number_input(
            "Income",
            min_value=0.0,
            max_value=600000.0,
            value=float(
                X["Income"].median()
            ),
            step=1000.0,
        )

        age_input = st.slider(
            "Age",
            min_value=18,
            max_value=90,
            value=int(
                np.clip(
                    round(
                        X["Age"].median()
                    ),
                    18,
                    90,
                )
            ),
        )

        spending_input = st.number_input(
            "Total Spending",
            min_value=0.0,
            max_value=100000.0,
            value=float(
                X[
                    "Total_Spending"
                ].median()
            ),
            step=50.0,
        )

    with right:

        recency_input = st.slider(
            "Recency (days)",
            min_value=0,
            max_value=max(
                100,
                int(
                    X["Recency"].max()
                ),
            ),
            value=int(
                np.clip(
                    round(
                        X["Recency"].median()
                    ),
                    0,
                    100,
                )
            ),
        )

        web_input = st.slider(
            "Web Purchases",
            min_value=0,
            max_value=30,
            value=int(
                np.clip(
                    round(
                        X[
                            "NumWebPurchases"
                        ].median()
                    ),
                    0,
                    30,
                )
            ),
        )

        store_input = st.slider(
            "Store Purchases",
            min_value=0,
            max_value=30,
            value=int(
                np.clip(
                    round(
                        X[
                            "NumStorePurchases"
                        ].median()
                    ),
                    0,
                    30,
                )
            ),
        )

        children_input = st.slider(
            "Total Children",
            min_value=0,
            max_value=5,
            value=int(
                np.clip(
                    round(
                        X[
                            "Total_Children"
                        ].median()
                    ),
                    0,
                    5,
                )
            ),
        )

    customer_row = (
        X.median()
        .to_frame()
        .T
    )

    customer_row[
        "Income"
    ] = float(
        income_input
    )

    customer_row[
        "Age"
    ] = float(
        age_input
    )

    customer_row[
        "Total_Spending"
    ] = float(
        spending_input
    )

    customer_row[
        "Recency"
    ] = float(
        recency_input
    )

    customer_row[
        "NumWebPurchases"
    ] = float(
        web_input
    )

    customer_row[
        "NumStorePurchases"
    ] = float(
        store_input
    )

    customer_row[
        "Total_Children"
    ] = float(
        children_input
    )

    transformed_customer = (
        pca.transform(
            scaler.transform(
                customer_row[
                    features
                ]
            )
        )
    )

    predicted_cluster = int(
        results["km"].predict(
            transformed_customer
        )[0]
    )

    predicted_segment = (
        kmeans_names[
            predicted_cluster
        ]
    )

    predicted_type = (
        kmeans_keys[
            predicted_cluster
        ]
    )

    predicted_details = SEGMENTS[
        predicted_type
    ]

    st.write("")

    st.success(
        f"Predicted Segment: "
        f"{predicted_segment}"
    )

    with st.container(
        border=True
    ):

        st.subheader(
            predicted_details["name"]
        )

        st.write(
            predicted_details[
                "description"
            ]
        )

        st.markdown(
            "**Recommended Action:** "
            + predicted_details[
                "action"
            ]
        )


# ============================================================
# EXPORT TAB
# ============================================================

with tabs[5]:

    st.header(
        "⬇️ Export Results"
    )

    export_columns = [
        "Cluster",
        "Segment",
    ]

    export_columns += [
        column
        for column in features
        if column in clean_data.columns
    ]

    export_data = clean_data[
        [
            column
            for column in ["ID"]
            if column in clean_data.columns
        ]
        + export_columns
    ]

    d1, d2, d3 = st.columns(3)

    with d1:

        st.download_button(
            label="⬇️ Clustered Customers CSV",
            data=(
                export_data
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            ),
            file_name=(
                "clustered_customers.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with d2:

        st.download_button(
            label="⬇️ Segment Profiles CSV",
            data=(
                displayed_profile
                .round(2)
                .to_csv()
                .encode("utf-8")
            ),
            file_name=(
                "segment_profiles.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with d3:

        st.download_button(
            label="⬇️ Recommendations CSV",
            data=(
                pd.DataFrame(
                    recommendations
                )
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            ),
            file_name=(
                "recommendations.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    st.write("")

    st.subheader(
        "Preview"
    )

    st.dataframe(
        export_data.head(20),
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛒 SmartCart AI · Customer Segmentation · "
    "K-Means + Agglomerative Clustering"
)