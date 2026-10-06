"""SmartCart AI - Customer Segmentation & Marketing Intelligence."""

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
# DARK THEME / CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0B1220;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #080F1C;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #E2E8F0 !important;
    }

    /* Hero */
    .hero-box {
        background: linear-gradient(
            135deg,
            #020617 0%,
            #0F2445 55%,
            #123563 100%
        );
        padding: 30px 34px;
        border-radius: 18px;
        margin-bottom: 22px;
        border: 1px solid #1E3A5F;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.30);
    }

    .hero-title {
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #93B4E8;
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .hero-description {
        color: #CBD5E1;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* KPI boxes */
    .metric-box {
        background: linear-gradient(
            145deg,
            #111827,
            #0F172A
        );
        border: 1px solid #1E293B;
        border-left: 5px solid #3B82F6;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 100px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 0.76rem;
        font-weight: 600;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section cards */
    .segment-card {
        background: linear-gradient(
            145deg,
            #111827,
            #0F172A
        );
        border: 1px solid #1E293B;
        border-top: 4px solid #3B82F6;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
    }

    .segment-title {
        color: #F8FAFC;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .segment-description {
        color: #CBD5E1;
        line-height: 1.5;
        margin-bottom: 10px;
    }

    .segment-recommendation {
        color: #E2E8F0;
        line-height: 1.5;
        margin-bottom: 10px;
    }

    .segment-meta {
        color: #94A3B8;
        font-size: 0.84rem;
    }

    /* Horizontal separator */
    hr {
        border-color: #1E293B !important;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1E293B;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #111827;
        border: 1px dashed #334155;
        border-radius: 10px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
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

BAD_DATA = (
    "Your dataset does not contain enough compatible customer "
    "features for segmentation."
)

SEGMENTS = {
    "High-Value": (
        "💎 High-Value Customers",
        "Customers with strong purchasing power and high spending.",
        "Offer loyalty rewards, premium products and exclusive early-access campaigns.",
        "Offer premium loyalty benefits.",
    ),
    "Growth": (
        "📈 Growth Customers",
        "Customers with moderate spending and potential to spend more.",
        "Use bundles, product recommendations and cross-selling.",
        "Push bundles and cross-sell offers.",
    ),
    "At-Risk": (
        "⚠️ At-Risk Customers",
        "Customers who have not purchased recently.",
        "Use personalized discounts, reminder campaigns and re-engagement offers.",
        "Send a personalized win-back discount.",
    ),
    "Regular": (
        "🛍️ Regular Customers",
        "Customers with steady, lower-value shopping behavior.",
        "Use seasonal promotions, loyalty points and low-cost upsell nudges.",
        "Run seasonal promotions and points programs.",
    ),
}


# ============================================================
# ERROR CLASS
# ============================================================

class DataError(Exception):
    pass


# ============================================================
# DEMO DATA
# ============================================================

@st.cache_data(show_spinner=False)
def demo_data(n=1200, seed=42):
    """Generate SmartCart-style demo customer data."""

    rng = np.random.default_rng(seed)

    groups = rng.choice(
        4,
        n,
        p=[0.22, 0.28, 0.30, 0.20],
    )

    income_mean = np.array(
        [78000, 55000, 36000, 46000]
    )

    spending_mean = np.array(
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

    visits_mean = np.array(
        [4, 6, 7, 5]
    )

    spending = np.clip(
        rng.normal(
            spending_mean[groups],
            spending_mean[groups] * 0.25,
        ),
        10,
        None,
    )

    mnt = (
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

    for name, values in zip(
        MNT,
        mnt.T,
    ):
        df[name] = values

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
            visits_mean[groups]
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

    # Add some missing Income values
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
    try:
        df = pd.read_csv(
            file,
            sep=None,
            engine="python",
        )
    except Exception:
        raise DataError(
            "The file could not be read as a CSV. "
            "Please upload a valid CSV file."
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

    df = raw.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    spending_columns = [
        c
        for c in MNT
        if c in df.columns
    ]

    missing = [
        c
        for c in CORE
        if c not in df.columns
    ]

    if missing or not spending_columns:

        needed = missing.copy()

        if not spending_columns:
            needed.append(
                "at least one Mnt* spending column"
            )

        raise DataError(
            f"{BAD_DATA} Missing: "
            f"{', '.join(needed)}."
        )

    # Remove leakage columns
    leakage_columns = [
        c
        for c in LEAKAGE
        if c in df.columns
    ]

    df = df.drop(
        columns=leakage_columns,
        errors="ignore",
    )

    numeric_columns = [
        c
        for c in (
            CORE
            + spending_columns
            + [
                "Kidhome",
                "Teenhome",
                "NumWebVisitsMonth",
            ]
        )
        if c in df.columns
    ]

    # Convert numeric values
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

    # Customer tenure
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

            numeric_columns.append(
                "Customer_Tenure"
            )

    # Median imputation
    for column in numeric_columns:

        df[column] = (
            df[column]
            .fillna(
                df[column].median()
            )
            .fillna(0)
        )

    # Age
    df["Age"] = (
        reference_year
        - df["Year_Birth"]
    )

    # Total spending
    df["Total_Spending"] = df[
        spending_columns
    ].sum(axis=1)

    # Total children
    child_columns = [
        c
        for c in [
            "Kidhome",
            "Teenhome",
        ]
        if c in df.columns
    ]

    if child_columns:

        df["Total_Children"] = df[
            child_columns
        ].sum(axis=1)

    else:

        df["Total_Children"] = 0

    # Education encoding
    if "Education" in df.columns:

        df["Education_Level"] = (
            df["Education"]
            .map(EDU)
            .fillna(1)
        )

    # Partner encoding
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

    # Remove outliers
    before_cleaning = len(df)

    df = df[
        (df["Age"] <= 90)
        & (df["Income"] <= 600000)
    ].reset_index(drop=True)

    features = [
        column
        for column in FEATURES
        if column in df.columns
    ]

    info = {
        "rows_raw": len(raw),
        "rows_clean": len(df),
        "outliers": (
            before_cleaning
            - len(df)
        ),
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
# PCA + SCALING
# ============================================================

@st.cache_data(show_spinner=False)
def build_space(X):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=3,
        random_state=42,
    )

    Z = pca.fit_transform(
        scaled
    )

    return (
        scaler,
        pca,
        Z,
    )


# ============================================================
# MODELS
# ============================================================

@st.cache_data(show_spinner=False)
def run_models(Z, k):

    kmeans = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42,
    )

    kmeans.fit(Z)

    kmeans_labels = (
        kmeans.labels_
    )

    agglomerative = (
        AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )
        .fit_predict(Z)
    )

    return {
        "km": kmeans,
        "km_labels": kmeans_labels,
        "ag_labels": agglomerative,

        "km_sil": silhouette_score(
            Z,
            kmeans_labels,
        ),

        "km_db": davies_bouldin_score(
            Z,
            kmeans_labels,
        ),

        "ag_sil": silhouette_score(
            Z,
            agglomerative,
        ),

        "ag_db": davies_bouldin_score(
            Z,
            agglomerative,
        ),
    }


# ============================================================
# PROFILE
# ============================================================

def profile(
    df,
    labels,
):

    temp = df.copy()

    temp["Cluster"] = labels

    columns = [
        c
        for c in PROFILE
        if c in temp.columns
    ]

    profile_data = (
        temp
        .groupby("Cluster")[columns]
        .mean()
    )

    profile_data.insert(
        0,
        "Customers",
        temp
        .groupby("Cluster")
        .size(),
    )

    return profile_data


# ============================================================
# SEGMENT NAMING
# ============================================================

def segment_labels(
    profile_data
):

    def z_score(series):

        std = series.std(
            ddof=0
        )

        if std == 0:
            return (
                series
                - series.mean()
            )

        return (
            series
            - series.mean()
        ) / std

    value_score = (
        z_score(
            profile_data["Income"]
        )
        + z_score(
            profile_data[
                "Total_Spending"
            ]
        )
    )

    keys = {
        value_score.idxmax():
            "High-Value"
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

            if (
                profile_data
                .loc[
                    cluster,
                    "Total_Spending",
                ]
                >= median_spending
            ):

                keys[
                    cluster
                ] = "Growth"

            else:

                keys[
                    cluster
                ] = "Regular"

    counts = (
        pd.Series(keys)
        .value_counts()
    )

    seen = {}
    names = {}

    for cluster in profile_data.index:

        segment_type = keys[
            cluster
        ]

        base_name = SEGMENTS[
            segment_type
        ][0]

        if counts[
            segment_type
        ] > 1:

            seen[
                segment_type
            ] = (
                seen.get(
                    segment_type,
                    0,
                )
                + 1
            )

            names[
                cluster
            ] = (
                f"{base_name} "
                f"{chr(
                    64
                    + seen[
                        segment_type
                    ]
                )}"
            )

        else:

            names[
                cluster
            ] = base_name

    return (
        keys,
        names,
    )


# ============================================================
# PLOTLY DARK THEME
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
            t=55,
            b=10,
        ),
    )

    if height:
        fig.update_layout(
            height=height
        )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Dashboard Controls"
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=[
            "csv",
            "txt",
        ],
    )

    st.markdown("")

    cluster_count = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=4,
    )

    st.markdown("")

    selected_model = st.radio(
        "Clustering model",
        [
            "K-Means",
            "Agglomerative Clustering",
        ],
    )

    st.markdown("")

    show_raw_data = st.checkbox(
        "Show raw data"
    )

    show_preprocessing = st.checkbox(
        "Show preprocessing details"
    )

    st.markdown("---")

    st.caption(
        "No CSV uploaded → the built-in "
        "demo dataset is used automatically."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-box">

        <div class="hero-title">
            🛒 SmartCart AI
        </div>

        <div class="hero-subtitle">
            Customer Segmentation & Marketing Intelligence
        </div>

        <div class="hero-description">
            Use machine learning to discover meaningful
            customer groups from purchasing behavior,
            spending patterns and engagement.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    if uploaded_file is not None:

        raw_data = read_csv(
            uploaded_file
        )

        source = (
            f"Uploaded file: "
            f"{uploaded_file.name}"
        )

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

    if len(clean_data) < max(
        30,
        5 * cluster_count,
    ):

        raise DataError(
            f"Only {len(clean_data)} "
            f"usable rows remain after "
            f"cleaning. Reduce the number "
            f"of clusters or upload more data."
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
        f"Unable to process the dataset: {error}"
    )

    st.stop()


# ============================================================
# SELECT CLUSTER LABELS
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
# BUILD SEGMENT PROFILES
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
    ].map(
        segment_names
    )
)

color_map = {
    name: PALETTE[
        i % len(PALETTE)
    ]
    for i, name in enumerate(
        sorted(
            segment_names.values()
        )
    )
}


# ============================================================
# KPI SECTION
# ============================================================

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-value">
                {len(clean_data):,}
            </div>
            <div class="metric-label">
                Total Customers
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-value">
                {len(features)}
            </div>
            <div class="metric-label">
                Features Used
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-value">
                {cluster_count}
            </div>
            <div class="metric-label">
                Number of Segments
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:

    best_silhouette = max(
        results["km_sil"],
        results["ag_sil"],
    )

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-value">
                {best_silhouette:.3f}
            </div>
            <div class="metric-label">
                Best Silhouette Score
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("")


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

    st.subheader(
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
            f"{len(raw_data):,}"
        )

    with m2:
        st.metric(
            "Columns",
            raw_data.shape[1]
        )

    with m3:
        st.metric(
            "Missing Values",
            f"{int(
                raw_data.isna()
                .sum()
                .sum()
            ):,}"
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

    st.markdown("")

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

    st.markdown("")

    st.subheader(
        "Data Processing Pipeline"
    )

    pipeline = [
        "Raw Data",
        "Missing Values",
        "Outlier Removal",
        "Feature Engineering",
        "Encoding",
        "Scaling",
        "PCA",
        "Clustering",
    ]

    st.info(
        " ➜ ".join(pipeline)
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
                f"**Clean rows:** "
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

            leakage_text = (
                ", ".join(info["leak"])
                if info["leak"]
                else "None"
            )

            st.write(
                f"**Leakage columns removed:** "
                f"{leakage_text}"
            )

            st.write(
                f"**Features used:** "
                f"{', '.join(features)}"
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

    st.subheader(
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
        color_discrete_map=color_map,
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
        color_discrete_map=color_map,
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

    st.subheader(
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
        f"🏆 Recommended Model: "
        f"{best_model} "
        f"(highest silhouette score "
        f"for k = {cluster_count})"
    )

    st.markdown("")

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
        clean_data["Segment"]
        .values
    )

    pca_fig = px.scatter_3d(
        pca_data,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Segment",
        color_discrete_map=color_map,
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
        f"Model shown: "
        f"{selected_model}"
    )


# ============================================================
# CUSTOMER PROFILES TAB
# ============================================================

with tabs[3]:

    st.subheader(
        "👥 Segment Profiles"
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

    st.markdown("")

    st.subheader(
        "🎯 Marketing Recommendations"
    )

    recommendation_columns = (
        st.columns(2)
    )

    recommendations = []

    for index, cluster in enumerate(
        profile_data.index
    ):

        segment_type = (
            segment_keys[
                cluster
            ]
        )

        (
            display_name,
            description,
            strategy,
            action,
        ) = SEGMENTS[
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
            f"{int(row['Customers']):,} customers "
            f"· avg income "
            f"{row['Income']:,.0f} "
            f"· avg spend "
            f"{row['Total_Spending']:,.0f} "
            f"· recency "
            f"{row['Recency']:.0f} days "
            f"· prefers "
            f"{preferred_channel} purchases"
        )

        recommendations.append(
            {
                "Segment": segment_names[
                    cluster
                ],
                "Customers": int(
                    row["Customers"]
                ),
                "Description": description,
                "Recommended Strategy": (
                    f"{strategy} "
                    f"Prioritise the "
                    f"{preferred_channel} "
                    f"channel."
                ),
            }
        )

        with recommendation_columns[
            index % 2
        ]:

            st.markdown(
                f"""
                <div class="segment-card">

                    <div class="segment-title">
                        {display_name}
                    </div>

                    <div class="segment-description">
                        {description}
                    </div>

                    <div class="segment-recommendation">
                        <strong>Recommendation:</strong>
                        {strategy}
                        Prioritise the
                        {preferred_channel}
                        channel.
                    </div>

                    <div class="segment-meta">
                        {metadata}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# CUSTOMER EXPLORER TAB
# ============================================================

with tabs[4]:

    st.subheader(
        "🔎 Customer Explorer"
    )

    st.caption(
        "Enter customer characteristics to predict "
        "their segment using the trained K-Means model."
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

        spending_input = (
            st.number_input(
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

    # Start from median values
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

    segment_details = SEGMENTS[
        predicted_type
    ]

    st.markdown("")

    st.success(
        f"Predicted Segment: "
        f"{predicted_segment}"
    )

    st.markdown(
        f"""
        <div class="segment-card">

            <div class="segment-title">
                {predicted_segment}
            </div>

            <div class="segment-description">
                {segment_details[1]}
            </div>

            <div class="segment-recommendation">
                <strong>Recommended Action:</strong>
                {segment_details[3]}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EXPORT TAB
# ============================================================

with tabs[5]:

    st.subheader(
        "⬇️ Download Results"
    )

    output_columns = [
        "Cluster",
        "Segment",
    ] + [
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
        + output_columns
    ]

    d1, d2, d3 = st.columns(3)

    with d1:

        st.download_button(
            "⬇️ Clustered Customers CSV",
            export_data
            .to_csv(
                index=False
            )
            .encode("utf-8"),
            "clustered_customers.csv",
            "text/csv",
            use_container_width=True,
        )

    with d2:

        st.download_button(
            "⬇️ Segment Profiles CSV",
            displayed_profile
            .round(2)
            .to_csv()
            .encode("utf-8"),
            "segment_profiles.csv",
            "text/csv",
            use_container_width=True,
        )

    with d3:

        st.download_button(
            "⬇️ Recommendations CSV",
            pd.DataFrame(
                recommendations
            )
            .to_csv(
                index=False
            )
            .encode("utf-8"),
            "recommendations.csv",
            "text/csv",
            use_container_width=True,
        )

    st.markdown("")

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

st.markdown(
    """
    <div class="footer">
        🛒 SmartCart AI · Customer Segmentation ·
        K-Means + Agglomerative Clustering
    </div>
    """,
    unsafe_allow_html=True,
)