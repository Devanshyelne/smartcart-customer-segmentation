"""SmartCart AI - Customer Segmentation Dashboard."""

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
# DARK UI CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Main application ---------- */

    .stApp {
        background: #0B1220;
        color: #F8FAFC;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #080F1C;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #E2E8F0 !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #94A3B8 !important;
    }

    /* ---------- Hero ---------- */

    .hero {
        background:
            radial-gradient(circle at top right, rgba(47,111,237,0.30), transparent 40%),
            linear-gradient(135deg, #020617 0%, #0F2445 55%, #102E59 100%);
        padding: 30px 34px;
        border-radius: 18px;
        margin-bottom: 20px;
        border: 1px solid #1E3A5F;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.30);
    }

    .hero h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 2.4rem;
        font-weight: 750;
    }

    .hero h3 {
        color: #8FB4F3 !important;
        margin: 7px 0 12px;
        font-weight: 500;
    }

    .hero p {
        color: #CBD5E1 !important;
        margin: 0;
        font-size: 1rem;
    }

    /* ---------- KPI cards ---------- */

    .kpi {
        background: linear-gradient(145deg, #111827, #0F172A);
        border: 1px solid #1E293B;
        border-left: 5px solid #2F6FED;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.22);
    }

    .kpi .v {
        font-size: 2rem;
        font-weight: 750;
        color: #F8FAFC;
        line-height: 1.1;
    }

    .kpi .l {
        color: #94A3B8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 7px;
    }

    /* ---------- Custom cards ---------- */

    .card {
        background: linear-gradient(145deg, #111827, #0F172A);
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.22);
        margin-bottom: 15px;
        border-top: 4px solid #2F6FED;
    }

    .card h4 {
        margin: 0 0 8px;
        color: #F8FAFC;
        font-size: 1.1rem;
    }

    .card p {
        margin: 5px 0;
        color: #CBD5E1;
        line-height: 1.5;
    }

    .card .meta {
        color: #94A3B8;
        font-size: 0.84rem;
    }

    /* ---------- Pipeline ---------- */

    .pipe {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin: 10px 0 18px;
    }

    .pipe .s {
        background: #172033;
        color: #E2E8F0;
        border: 1px solid #29364B;
        padding: 7px 14px;
        border-radius: 20px;
        font-size: 0.84rem;
    }

    .pipe .a {
        color: #60A5FA;
        font-weight: 700;
    }

    /* ---------- Tabs ---------- */

    button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA !important;
    }

    /* ---------- General headings ---------- */

    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
    }

    /* ---------- Text ---------- */

    p, span, label {
        color: #CBD5E1;
    }

    /* ---------- Dataframes ---------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid #1E293B;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---------- Metric widgets ---------- */

    div[data-testid="stMetric"] {
        background: #111827;
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

    /* ---------- Expanders ---------- */

    details {
        background: #111827 !important;
        border: 1px solid #1E293B !important;
        border-radius: 10px !important;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    /* ---------- File uploader ---------- */

    section[data-testid="stFileUploaderDropzone"] {
        background: #111827;
        border: 1px dashed #334155;
    }

    /* ---------- Alerts ---------- */

    div[data-testid="stAlert"] {
        border-radius: 10px;
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
    "Your dataset does not contain enough compatible customer features "
    "for segmentation."
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
# CUSTOM ERROR
# ============================================================

class DataError(Exception):
    pass


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner=False)
def demo_data(n=1200, seed=42):
    """Synthetic SmartCart-style customer dataset."""

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

    visits_mean = np.array(
        [4, 6, 7, 5]
    )

    spend = np.clip(
        rng.normal(
            spend_mean[groups],
            spend_mean[groups] * 0.25,
        ),
        10,
        None,
    )

    mnt = (
        spend[:, None]
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
                p=[0.5, 0.2, 0.17, 0.03, 0.10],
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
                p=[0.38, 0.26, 0.22, 0.11, 0.03],
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

    for name, col in zip(MNT, mnt.T):
        df[name] = col

    df["NumWebPurchases"] = np.clip(
        rng.poisson(web_mean[groups]),
        0,
        27,
    )

    df["NumStorePurchases"] = np.clip(
        rng.poisson(store_mean[groups]),
        0,
        13,
    )

    df["NumWebVisitsMonth"] = np.clip(
        rng.poisson(visits_mean[groups]),
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

    df.loc[
        rng.choice(n, 20, replace=False),
        "Income",
    ] = np.nan

    return df


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


def preprocess(raw):

    df = raw.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    available_mnt = [
        c for c in MNT
        if c in df.columns
    ]

    missing = [
        c for c in CORE
        if c not in df.columns
    ]

    if missing or not available_mnt:

        need = (
            missing
            + (
                []
                if available_mnt
                else ["at least one Mnt* spending column"]
            )
        )

        raise DataError(
            f"{BAD_DATA} Missing: {', '.join(need)}."
        )

    leak = [
        c for c in LEAKAGE
        if c in df.columns
    ]

    # Remove target leakage
    df = df.drop(
        columns=leak,
        errors="ignore",
    )

    numeric_columns = [
        c
        for c in (
            CORE
            + available_mnt
            + [
                "Kidhome",
                "Teenhome",
                "NumWebVisitsMonth",
            ]
        )
        if c in df.columns
    ]

    for c in numeric_columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    n_missing = int(
        df[numeric_columns]
        .isna()
        .sum()
        .sum()
    )

    ref_year = pd.Timestamp.now().year

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

            ref_year = int(
                dates.max().year
            )

            numeric_columns.append(
                "Customer_Tenure"
            )

    # Median imputation
    for c in numeric_columns:
        df[c] = (
            df[c]
            .fillna(df[c].median())
            .fillna(0)
        )

    # Feature engineering
    df["Age"] = (
        ref_year - df["Year_Birth"]
    )

    df["Total_Spending"] = df[
        available_mnt
    ].sum(axis=1)

    children_columns = [
        c
        for c in (
            "Kidhome",
            "Teenhome",
        )
        if c in df.columns
    ]

    if children_columns:
        df["Total_Children"] = df[
            children_columns
        ].sum(axis=1)
    else:
        df["Total_Children"] = 0

    if "Education" in df.columns:
        df["Education_Level"] = (
            df["Education"]
            .map(EDU)
            .fillna(1)
        )

    if "Marital_Status" in df.columns:
        df["Living_With_Partner"] = (
            df["Marital_Status"]
            .isin(
                ["Married", "Together"]
            )
            .astype(int)
        )

    # Outlier removal
    n_before = len(df)

    df = df[
        (df["Age"] <= 90)
        & (df["Income"] <= 600000)
    ].reset_index(drop=True)

    features = [
        c
        for c in FEATURES
        if c in df.columns
    ]

    info = {
        "rows_raw": len(raw),
        "rows_clean": len(df),
        "outliers": n_before - len(df),
        "imputed": n_missing,
        "leak": leak,
        "features": features,
    }

    return df, features, info


# ============================================================
# MACHINE LEARNING
# ============================================================

@st.cache_data(show_spinner=False)
def build_space(X):

    scaler = StandardScaler().fit(X)

    pca = PCA(
        n_components=3,
        random_state=42,
    ).fit(
        scaler.transform(X)
    )

    Z = pca.transform(
        scaler.transform(X)
    )

    return scaler, pca, Z


@st.cache_data(show_spinner=False)
def run_models(Z, k):

    km = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42,
    ).fit(Z)

    ag = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    ).fit_predict(Z)

    return {
        "km": km,
        "km_labels": km.labels_,
        "ag_labels": ag,

        "km_sil": silhouette_score(
            Z,
            km.labels_,
        ),

        "km_db": davies_bouldin_score(
            Z,
            km.labels_,
        ),

        "ag_sil": silhouette_score(
            Z,
            ag,
        ),

        "ag_db": davies_bouldin_score(
            Z,
            ag,
        ),
    }


def profile(df, labels):

    d = df.assign(
        Cluster=labels
    )

    columns = [
        c
        for c in PROFILE
        if c in d.columns
    ]

    p = d.groupby(
        "Cluster"
    )[columns].mean()

    p.insert(
        0,
        "Customers",
        d.groupby("Cluster").size(),
    )

    return p


def segment_labels(p):

    """Generate business-friendly labels."""

    def z(series):

        std = series.std(ddof=0)

        if std == 0:
            return series - series.mean()

        return (
            series - series.mean()
        ) / std

    value = (
        z(p["Income"])
        + z(p["Total_Spending"])
    )

    keys = {
        value.idxmax(): "High-Value"
    }

    remaining = [
        c
        for c in p.index
        if c not in keys
    ]

    if remaining:

        # Highest recency = at-risk
        at_risk_cluster = (
            p.loc[
                remaining,
                "Recency"
            ].idxmax()
        )

        keys[
            at_risk_cluster
        ] = "At-Risk"

        median_spend = (
            p["Total_Spending"]
            .median()
        )

        for c in remaining:

            if c not in keys:

                if (
                    p.loc[
                        c,
                        "Total_Spending"
                    ]
                    >= median_spend
                ):
                    keys[c] = "Growth"
                else:
                    keys[c] = "Regular"

    counts = pd.Series(
        keys
    ).value_counts()

    seen = {}
    names = {}

    for cluster in p.index:

        key = keys[cluster]

        base = SEGMENTS[key][0]

        if counts[key] > 1:

            seen[key] = (
                seen.get(key, 0) + 1
            )

            names[cluster] = (
                f"{base} "
                f"{chr(64 + seen[key])}"
            )

        else:
            names[cluster] = base

    return keys, names


# ============================================================
# UI HELPERS
# ============================================================

def kpi(col, value, label):

    col.markdown(
        f"""
        <div class="kpi">
            <div class="v">{value}</div>
            <div class="l">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dark_plot(fig, height=None):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#CBD5E1"
        ),
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=10,
        ),
        legend=dict(
            font=dict(
                color="#CBD5E1"
            )
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

    st.markdown(
        "---"
    )

    upload = st.file_uploader(
        "Upload customer CSV",
        type=["csv", "txt"],
    )

    st.markdown("")

    k = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=4,
    )

    st.markdown("")

    model_name = st.radio(
        "Clustering model",
        [
            "K-Means",
            "Agglomerative Clustering",
        ],
    )

    st.markdown("")

    show_raw = st.checkbox(
        "Show raw data"
    )

    show_prep = st.checkbox(
        "Show preprocessing details"
    )

    st.markdown("---")

    st.caption(
        "No CSV uploaded → "
        "the built-in demo dataset is used automatically."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🛒 SmartCart AI</h1>

        <h3>
            Customer Segmentation & Marketing Intelligence
        </h3>

        <p>
            Use machine learning to discover meaningful
            customer groups from purchasing behavior,
            spending patterns and engagement.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD + PREPROCESS + TRAIN
# ============================================================

try:

    if upload is not None:

        raw = read_csv(upload)

        source = (
            f"Uploaded file: {upload.name}"
        )

    else:

        raw = demo_data()

        source = "Built-in demo dataset"

    df, features, info = preprocess(
        raw
    )

    if len(df) < max(30, 5 * k):

        raise DataError(
            f"Only {len(df)} usable rows "
            f"after cleaning - not enough "
            f"to form {k} segments. "
            f"Upload more data or reduce "
            f"the number of clusters."
        )

    X = df[features].astype(float)

    scaler, pca, Z = build_space(X)

    results = run_models(
        Z,
        k,
    )

except DataError as error:

    st.error(
        str(error)
    )

    st.stop()

except Exception:

    st.error(
        BAD_DATA
    )

    st.stop()


# ============================================================
# SELECT MODEL
# ============================================================

use_kmeans = (
    model_name == "K-Means"
)

labels = (
    results["km_labels"]
    if use_kmeans
    else results["ag_labels"]
)


# ============================================================
# SEGMENT PROFILES
# ============================================================

profile_df = profile(
    df,
    labels,
)

segment_keys, segment_names = (
    segment_labels(profile_df)
)

km_profile = profile(
    df,
    results["km_labels"],
)

km_keys, km_names = (
    segment_labels(km_profile)
)


df["Cluster"] = labels

df["Segment"] = (
    df["Cluster"]
    .map(segment_names)
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
# KPI ROW
# ============================================================

c1, c2, c3, c4 = st.columns(4)

kpi(
    c1,
    f"{len(df):,}",
    "Total Customers",
)

kpi(
    c2,
    len(features),
    "Features Used",
)

kpi(
    c3,
    k,
    "Number of Segments",
)

kpi(
    c4,
    f"{max(
        results['km_sil'],
        results['ag_sil']
    ):.3f}",
    "Best Silhouette Score",
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

    st.caption(
        f"Source: {source}"
    )

    metrics = st.columns(5)

    metrics[0].metric(
        "Rows",
        f"{len(raw):,}",
    )

    metrics[1].metric(
        "Columns",
        raw.shape[1],
    )

    metrics[2].metric(
        "Missing values",
        f"{int(
            raw.isna()
            .sum()
            .sum()
        ):,}",
    )

    metrics[3].metric(
        "Numerical columns",
        raw.select_dtypes(
            include="number"
        ).shape[1],
    )

    metrics[4].metric(
        "Categorical columns",
        raw.shape[1]
        - raw.select_dtypes(
            include="number"
        ).shape[1],
    )

    st.markdown("")

    st.dataframe(
        raw
        if show_raw
        else raw.head(10),
        use_container_width=True,
    )

    st.markdown("")

    st.subheader(
        "Data Processing Pipeline"
    )

    pipeline_steps = [
        "Raw Data",
        "Missing Values",
        "Outlier Removal",
        "Feature Engineering",
        "Encoding",
        "Scaling",
        "PCA",
        "Clustering",
    ]

    st.markdown(
        '<div class="pipe">'
        + '<span class="a">➜</span>'.join(
            f'<span class="s">{step}</span>'
            for step in pipeline_steps
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    if show_prep:

        with st.expander(
            "Preprocessing details",
            expanded=True,
        ):

            st.markdown(
                f"""
                **Rows:** {info['rows_raw']:,}
                raw → {info['rows_clean']:,}
                after cleaning

                **Outliers removed:** {info['outliers']:,}

                **Missing values filled:** {info['imputed']:,}

                **Leakage columns excluded:**
                {', '.join(info['leak'])
                if info['leak']
                else 'None'}

                **Engineered features:**
                Age, Customer Tenure,
                Total Spending,
                Total Children,
                Education Level,
                Living With Partner

                **Features used:**
                {', '.join(features)}

                **Scaling:**
                StandardScaler

                **Dimensionality reduction:**
                PCA → 3 components

                **Final step:**
                Clustering
                """
            )


# ============================================================
# OVERVIEW TAB
# ============================================================

with tabs[1]:

    st.subheader(
        "📊 Customer Overview"
    )

    a, b = st.columns(2)

    counts = (
        df.groupby("Segment")
        .size()
        .reset_index(
            name="Customers"
        )
    )

    fig_distribution = px.bar(
        counts,
        x="Segment",
        y="Customers",
        color="Segment",
        color_discrete_map=color_map,
        title="Customer Distribution by Segment",
    )

    fig_distribution.update_layout(
        showlegend=False
    )

    a.plotly_chart(
        dark_plot(fig_distribution),
        use_container_width=True,
    )

    scatter_data = df.sample(
        min(
            len(df),
            2000,
        ),
        random_state=1,
    )

    fig_income = px.scatter(
        scatter_data,
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

    b.plotly_chart(
        dark_plot(fig_income),
        use_container_width=True,
    )

    a, b = st.columns(2)

    fig_spending = px.histogram(
        df,
        x="Total_Spending",
        nbins=40,
        title="Spending Distribution",
    )

    a.plotly_chart(
        dark_plot(fig_spending),
        use_container_width=True,
    )

    fig_recency = px.histogram(
        df,
        x="Recency",
        nbins=40,
        title="Recency Distribution",
    )

    b.plotly_chart(
        dark_plot(fig_recency),
        use_container_width=True,
    )


# ============================================================
# SEGMENTATION TAB
# ============================================================

with tabs[2]:

    st.subheader(
        "🤖 Model Evaluation"
    )

    silhouette, db_score = (
        (
            results["km_sil"],
            results["km_db"],
        )
        if use_kmeans
        else (
            results["ag_sil"],
            results["ag_db"],
        )
    )

    evaluation_cols = st.columns(2)

    evaluation_cols[0].metric(
        f"Silhouette Score ({model_name})",
        f"{silhouette:.3f}",
    )

    evaluation_cols[1].metric(
        f"Davies-Bouldin Score ({model_name})",
        f"{db_score:.3f}",
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

    best_model = (
        "K-Means"
        if results["km_sil"]
        >= results["ag_sil"]
        else "Agglomerative Clustering"
    )

    st.success(
        f"🏆 Recommended Model: "
        f"{best_model} "
        f"(highest silhouette score for k = {k})"
    )

    st.markdown("")

    st.subheader(
        "🌐 PCA Cluster Visualization"
    )

    principal_components = pd.DataFrame(
        Z,
        columns=[
            "PC1",
            "PC2",
            "PC3",
        ],
    )

    principal_components[
        "Segment"
    ] = df[
        "Segment"
    ].values

    fig_3d = px.scatter_3d(
        principal_components,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Segment",
        color_discrete_map=color_map,
        opacity=0.82,
        title="3D PCA Customer Segmentation",
    )

    fig_3d.update_traces(
        marker_size=4
    )

    st.plotly_chart(
        dark_plot(
            fig_3d,
            height=650,
        ),
        use_container_width=True,
    )

    st.info(
        f"PCA explained variance: "
        f"{pca.explained_variance_ratio_.sum() * 100:.1f}% "
        f"(3 components) · "
        f"Model shown: {model_name}"
    )


# ============================================================
# CUSTOMER PROFILES TAB
# ============================================================

with tabs[3]:

    st.subheader(
        "👥 Segment Profiles"
    )

    shown = profile_df.rename(
        index=segment_names
    )

    shown.index.name = "Segment"

    st.dataframe(
        shown.round(1),
        use_container_width=True,
    )

    st.markdown("")

    st.subheader(
        "🎯 Marketing Recommendations"
    )

    recommendation_columns = st.columns(2)

    recommendations = []

    for i, cluster in enumerate(
        profile_df.index
    ):

        segment_type = (
            segment_keys[cluster]
        )

        label, description, strategy, action = (
            SEGMENTS[segment_type]
        )

        row = profile_df.loc[
            cluster
        ]

        channel = (
            "web"
            if row["NumWebPurchases"]
            > row["NumStorePurchases"]
            else "in-store"
        )

        metadata = (
            f"{int(row['Customers']):,} customers · "
            f"avg income {row['Income']:,.0f} · "
            f"avg spend {row['Total_Spending']:,.0f} · "
            f"recency {row['Recency']:.0f} days · "
            f"prefers {channel} purchases"
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
                    f"{channel} channel."
                ),
            }
        )

        recommendation_columns[
            i % 2
        ].markdown(
            f"""
            <div class="card">

                <h4>
                    {segment_names[cluster]}
                </h4>

                <p>
                    {description}
                </p>

                <p>
                    <b>Recommendation:</b>
                    {strategy}
                    Prioritise the
                    {channel} channel.
                </p>

                <p class="meta">
                    {metadata}
                </p>

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
        "Uses the trained K-Means pipeline: "
        "Scaler → PCA → K-Means."
    )

    def clip_value(
        value,
        low,
        high,
    ):
        return int(
            min(
                max(
                    round(value),
                    low,
                ),
                high,
            )
        )

    x1, x2 = st.columns(2)

    income = x1.number_input(
        "Income",
        min_value=0.0,
        max_value=600000.0,
        value=float(
            X["Income"].median()
        ),
        step=1000.0,
    )

    age = x1.slider(
        "Age",
        min_value=18,
        max_value=90,
        value=clip_value(
            X["Age"].median(),
            18,
            90,
        ),
    )

    spending = x1.number_input(
        "Total Spending",
        min_value=0.0,
        max_value=100000.0,
        value=float(
            X["Total_Spending"].median()
        ),
        step=50.0,
    )

    recency = x2.slider(
        "Recency (days)",
        min_value=0,
        max_value=max(
            100,
            int(
                X["Recency"].max()
            ),
        ),
        value=clip_value(
            X["Recency"].median(),
            0,
            100,
        ),
    )

    web_purchases = x2.slider(
        "Web purchases",
        min_value=0,
        max_value=30,
        value=clip_value(
            X["NumWebPurchases"].median(),
            0,
            30,
        ),
    )

    store_purchases = x2.slider(
        "Store purchases",
        min_value=0,
        max_value=30,
        value=clip_value(
            X["NumStorePurchases"].median(),
            0,
            30,
        ),
    )

    total_children = x2.slider(
        "Total children",
        min_value=0,
        max_value=5,
        value=clip_value(
            X["Total_Children"].median(),
            0,
            5,
        ),
    )

    row = (
        X.median()
        .to_frame()
        .T
    )

    row["Income"] = float(
        income
    )

    row["Age"] = float(
        age
    )

    row["Total_Spending"] = float(
        spending
    )

    row["Recency"] = float(
        recency
    )

    row["NumWebPurchases"] = float(
        web_purchases
    )

    row["NumStorePurchases"] = float(
        store_purchases
    )

    row["Total_Children"] = float(
        total_children
    )

    transformed = pca.transform(
        scaler.transform(
            row[features]
        )
    )

    predicted_cluster = int(
        results["km"].predict(
            transformed
        )[0]
    )

    predicted_segment = km_names[
        predicted_cluster
    ]

    segment_details = SEGMENTS[
        km_keys[predicted_cluster]
    ]

    st.markdown("")

    st.markdown(
        f"""
        <div class="card">

            <p class="meta">
                PREDICTED CUSTOMER SEGMENT
            </p>

            <h4>
                {predicted_segment}
            </h4>

            <p>
                {segment_details[1]}
            </p>

            <p>
                <b>Recommended Action:</b>
                {segment_details[3]}
            </p>

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
        c
        for c in features
        if c in df.columns
    ]

    base = df[
        [
            c
            for c in ["ID"]
            if c in df.columns
        ]
        + output_columns
    ]

    d1, d2, d3 = st.columns(3)

    d1.download_button(
        label="⬇️ Clustered Customers CSV",
        data=base.to_csv(
            index=False
        ).encode(),
        file_name="clustered_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )

    d2.download_button(
        label="⬇️ Segment Profiles CSV",
        data=shown.round(2)
        .to_csv()
        .encode(),
        file_name="segment_profiles.csv",
        mime="text/csv",
        use_container_width=True,
    )

    d3.download_button(
        label="⬇️ Recommendations CSV",
        data=pd.DataFrame(
            recommendations
        )
        .to_csv(index=False)
        .encode(),
        file_name="recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("")

    st.subheader(
        "Preview"
    )

    st.dataframe(
        base.head(20),
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        padding:25px 0 10px 0;
        color:#64748B;
        font-size:0.85rem;
    ">
        🛒 SmartCart AI · Machine Learning Customer Segmentation
        · K-Means + Agglomerative Clustering
    </div>
    """,
    unsafe_allow_html=True,
)