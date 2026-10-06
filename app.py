"""SmartCart AI - Customer Segmentation Dashboard (single-file Streamlit app)."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SmartCart AI", page_icon="🛒", layout="wide")

st.markdown(
    """<style>
.block-container{padding-top:1.5rem;max-width:1250px}
.hero{background:linear-gradient(135deg,#0B1B33,#16315C);padding:28px 32px;border-radius:16px;margin-bottom:18px}
.hero h1{color:#fff !important;margin:0;font-size:2.2rem}
.hero h3{color:#9DB7E8 !important;margin:4px 0 10px;font-weight:500}
.hero p{color:#D6E2F7 !important;margin:0}
.kpi{background:#fff;border-left:5px solid #2F6FED;border-radius:12px;padding:16px 18px;box-shadow:0 2px 8px rgba(11,27,51,.08)}
.kpi .v{font-size:1.9rem;font-weight:700;color:#0B1B33}
.kpi .l{color:#6B7A90;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
.card{background:#fff;border-radius:14px;padding:18px 20px;box-shadow:0 2px 8px rgba(11,27,51,.08);margin-bottom:14px;border-top:4px solid #2F6FED}
.card h4{margin:0 0 6px;color:#0B1B33}
.card p{margin:4px 0;color:#33415C}
.card .meta{color:#6B7A90;font-size:.85rem}
.pipe{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0 16px}
.pipe .s{background:#0B1B33;color:#fff;padding:7px 14px;border-radius:20px;font-size:.85rem}
.pipe .a{color:#2F6FED;font-weight:700}
</style>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- constants
CORE = ["Income", "Year_Birth", "Recency", "NumWebPurchases", "NumStorePurchases"]
MNT = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
LEAKAGE = ["Response"] + [f"AcceptedCmp{i}" for i in range(1, 6)]  # campaign outcomes
FEATURES = ["Income", "Age", "Recency", "Total_Spending", "Total_Children", "Customer_Tenure",
            "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Education_Level",
            "Living_With_Partner"]
PROFILE = ["Income", "Recency", "Total_Spending", "Total_Children", "Age", "Customer_Tenure",
           "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth"]
EDU = {"Basic": 0, "2n Cycle": 0, "Graduation": 1, "Master": 2, "PhD": 2}
PALETTE = ["#2F6FED", "#0B1B33", "#18B8A6", "#F59E0B", "#E5484D", "#8E6CEF", "#64748B", "#10B981",
           "#EC4899", "#A16207"]
BAD_DATA = "Your dataset does not contain enough compatible customer features for segmentation."

SEGMENTS = {
    "High-Value": ("💎 High-Value Customers", "Customers with strong purchasing power and high spending.",
                   "Offer loyalty rewards, premium products and exclusive early-access campaigns.",
                   "Offer premium loyalty benefits."),
    "Growth": ("📈 Growth Customers", "Customers with moderate spending and potential to spend more.",
               "Use bundles, product recommendations and cross-selling.",
               "Push bundles and cross-sell offers."),
    "At-Risk": ("⚠️ At-Risk Customers", "Customers who have not purchased recently (relatively high recency).",
                "Use personalized discounts, reminder campaigns and re-engagement offers.",
                "Send a personalized win-back discount."),
    "Regular": ("🛍️ Regular Customers", "Customers with steady, lower-value shopping behavior.",
                "Use seasonal promotions, loyalty points and low-cost upsell nudges.",
                "Run seasonal promotions and points programs."),
}


class DataError(Exception):
    pass


# ----------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def demo_data(n=1200, seed=42):
    """Synthetic data in the same schema as the original SmartCart dataset."""
    rng = np.random.default_rng(seed)
    g = rng.choice(4, n, p=[0.22, 0.28, 0.30, 0.20])
    inc_m, spend_m = np.array([78000, 55000, 36000, 46000]), np.array([1400, 650, 160, 320])
    rec_m, kids_m = np.array([22, 30, 38, 78]), np.array([0.3, 1.2, 1.5, 1.0])
    web_m, store_m, vis_m = np.array([5, 6, 3, 2]), np.array([8, 6, 4, 3]), np.array([4, 6, 7, 5])
    spend = np.clip(rng.normal(spend_m[g], spend_m[g] * 0.25), 10, None)
    mnt = (spend[:, None] * rng.dirichlet([6, 1, 4, 1.5, 1, 1.5], n)).round().astype(int)
    age = np.clip(rng.normal(48, 11, n), 24, 80).round().astype(int)
    dates = pd.Timestamp("2012-07-30") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    df = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "Year_Birth": 2014 - age,
        "Education": rng.choice(["Graduation", "PhD", "Master", "Basic", "2n Cycle"], n, p=[.5, .2, .17, .03, .1]),
        "Marital_Status": rng.choice(["Married", "Together", "Single", "Divorced", "Widow"], n,
                                     p=[.38, .26, .22, .11, .03]),
        "Income": np.clip(rng.normal(inc_m[g], 8000), 8000, None).round(),
        "Kidhome": rng.binomial(2, kids_m[g] / 4),
        "Teenhome": rng.binomial(2, kids_m[g] / 4),
        "Dt_Customer": dates.strftime("%d-%m-%Y"),
        "Recency": np.clip(rng.normal(rec_m[g], 12), 0, 99).round().astype(int),
    })
    for name, col in zip(MNT, mnt.T):
        df[name] = col
    df["NumWebPurchases"] = np.clip(rng.poisson(web_m[g]), 0, 27)
    df["NumStorePurchases"] = np.clip(rng.poisson(store_m[g]), 0, 13)
    df["NumWebVisitsMonth"] = np.clip(rng.poisson(vis_m[g]), 0, 20)
    df["AcceptedCmp1"] = rng.binomial(1, 0.07, n)
    df["Response"] = rng.binomial(1, 0.15, n)
    df.loc[rng.choice(n, 20, replace=False), "Income"] = np.nan
    return df


def read_csv(file):
    try:
        df = pd.read_csv(file, sep=None, engine="python")  # auto-detects comma / tab
    except Exception:
        raise DataError("The file could not be read as a CSV. Please upload a valid CSV file.")
    if df.empty:
        raise DataError("The uploaded file is empty.")
    return df


def preprocess(raw):
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    mnt = [c for c in MNT if c in df.columns]
    missing = [c for c in CORE if c not in df.columns]
    if missing or not mnt:
        need = missing + ([] if mnt else ["at least one Mnt* spending column"])
        raise DataError(f"{BAD_DATA}  Missing: {', '.join(need)}.")
    leak = [c for c in LEAKAGE if c in df.columns]
    df = df.drop(columns=leak)  # target-leakage fix: Response / campaign columns never reach the model

    num = [c for c in CORE + mnt + ["Kidhome", "Teenhome", "NumWebVisitsMonth"] if c in df.columns]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_missing = int(df[num].isna().sum().sum())

    ref_year = pd.Timestamp.now().year
    if "Dt_Customer" in df.columns:
        d = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
        if d.notna().any():
            df["Customer_Tenure"] = (d.max() - d).dt.days
            ref_year = int(d.max().year)
            num.append("Customer_Tenure")
    for c in num:  # missing-value handling: median imputation
        df[c] = df[c].fillna(df[c].median()).fillna(0)

    df["Age"] = ref_year - df["Year_Birth"]
    df["Total_Spending"] = df[mnt].sum(axis=1)
    kids = [c for c in ("Kidhome", "Teenhome") if c in df.columns]
    df["Total_Children"] = df[kids].sum(axis=1) if kids else 0
    if "Education" in df.columns:
        df["Education_Level"] = df["Education"].map(EDU).fillna(1)
    if "Marital_Status" in df.columns:
        df["Living_With_Partner"] = df["Marital_Status"].isin(["Married", "Together"]).astype(int)

    n_before = len(df)
    df = df[(df["Age"] <= 90) & (df["Income"] <= 600000)].reset_index(drop=True)  # outlier removal
    feats = [c for c in FEATURES if c in df.columns]
    info = {"rows_raw": len(raw), "rows_clean": len(df), "outliers": n_before - len(df),
            "imputed": n_missing, "leak": leak, "features": feats}
    return df, feats, info


# ----------------------------------------------------------------- ML
@st.cache_data(show_spinner=False)
def build_space(X):
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=3, random_state=42).fit(scaler.transform(X))
    return scaler, pca, pca.transform(scaler.transform(X))


@st.cache_data(show_spinner=False)
def run_models(Z, k):
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    return {
        "km": km, "km_labels": km.labels_, "ag_labels": ag,
        "km_sil": silhouette_score(Z, km.labels_), "km_db": davies_bouldin_score(Z, km.labels_),
        "ag_sil": silhouette_score(Z, ag), "ag_db": davies_bouldin_score(Z, ag),
    }


def profile(df, labels):
    d = df.assign(Cluster=labels)
    cols = [c for c in PROFILE if c in d.columns]
    p = d.groupby("Cluster")[cols].mean()
    p.insert(0, "Customers", d.groupby("Cluster").size())
    return p


def segment_labels(p):
    """Derive business names from cluster statistics (never from cluster numbers)."""
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
    value = z(p["Income"]) + z(p["Total_Spending"])
    keys = {value.idxmax(): "High-Value"}
    rest = [c for c in p.index if c not in keys]
    if rest:
        keys[p.loc[rest, "Recency"].idxmax()] = "At-Risk"
        median_spend = p["Total_Spending"].median()
        for c in rest:
            if c not in keys:
                keys[c] = "Growth" if p.loc[c, "Total_Spending"] >= median_spend else "Regular"
    counts = pd.Series(keys).value_counts()
    seen, names = {}, {}
    for c in p.index:
        k = keys[c]
        base = SEGMENTS[k][0]
        if counts[k] > 1:
            seen[k] = seen.get(k, 0) + 1
            names[c] = f"{base} {chr(64 + seen[k])}"
        else:
            names[c] = base
    return keys, names


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    up = st.file_uploader("Upload customer CSV", type=["csv", "txt"])
    k = st.slider("Number of clusters", 2, 10, 4)
    model_name = st.radio("Clustering model", ["K-Means", "Agglomerative Clustering"])
    show_raw = st.checkbox("Show raw data")
    show_prep = st.checkbox("Show preprocessing details")
    st.caption("No CSV uploaded → the built-in demo dataset is used automatically.")

st.markdown(
    """<div class="hero"><h1>🛒 SmartCart AI</h1><h3>Customer Segmentation &amp; Marketing Intelligence</h3>
<p>Use machine learning to discover meaningful customer groups from purchasing behavior,
spending patterns and engagement.</p></div>""",
    unsafe_allow_html=True,
)

try:
    if up is not None:
        raw, source = read_csv(up), f"Uploaded file: {up.name}"
    else:
        raw, source = demo_data(), "Built-in demo dataset"
    df, feats, info = preprocess(raw)
    if len(df) < max(30, 5 * k):
        raise DataError(f"Only {len(df)} usable rows after cleaning - not enough to form {k} segments. "
                        "Upload more data or reduce the number of clusters.")
    X = df[feats].astype(float)
    scaler, pca, Z = build_space(X)
    res = run_models(Z, k)
except DataError as e:
    st.error(str(e))
    st.stop()
except Exception:
    st.error(BAD_DATA)
    st.stop()

use_km = model_name == "K-Means"
labels = res["km_labels"] if use_km else res["ag_labels"]
prof = profile(df, labels)
seg_keys, seg_names = segment_labels(prof)
km_prof = profile(df, res["km_labels"])
km_keys, km_names = segment_labels(km_prof)
df["Cluster"] = labels
df["Segment"] = df["Cluster"].map(seg_names)
cmap = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sorted(seg_names.values()))}

c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{len(df):,}", "Total Customers")
kpi(c2, len(feats), "Features Used")
kpi(c3, k, "Number of Segments")
kpi(c4, f"{max(res['km_sil'], res['ag_sil']):.3f}", "Best Silhouette Score")
st.write("")

tabs = st.tabs(["📁 Dataset", "📊 Overview", "🤖 Segmentation", "👥 Customer Profiles",
                "🔎 Customer Explorer", "⬇️ Export"])

# ----------------------------------------------------------------- dataset
with tabs[0]:
    st.caption(f"Source: {source}")
    m = st.columns(5)
    m[0].metric("Rows", f"{len(raw):,}")
    m[1].metric("Columns", raw.shape[1])
    m[2].metric("Missing values", f"{int(raw.isna().sum().sum()):,}")
    m[3].metric("Numerical columns", raw.select_dtypes(include="number").shape[1])
    m[4].metric("Categorical columns", raw.shape[1] - raw.select_dtypes(include="number").shape[1])
    st.dataframe(raw if show_raw else raw.head(10))
    st.subheader("Data Processing Pipeline")
    steps = ["Raw Data", "Missing Value Handling", "Outlier Removal", "Feature Engineering",
             "Encoding", "Scaling", "PCA", "Clustering"]
    st.markdown('<div class="pipe">' + '<span class="a">➜</span>'.join(f'<span class="s">{s}</span>' for s in steps)
                + "</div>", unsafe_allow_html=True)
    if show_prep:
        with st.expander("Preprocessing details", expanded=True):
            st.markdown(
                f"- **Rows:** {info['rows_raw']:,} raw → {info['rows_clean']:,} after cleaning "
                f"({info['outliers']} outliers removed: Age > 90 or Income > 600,000)\n"
                f"- **Missing values filled (median):** {info['imputed']:,}\n"
                f"- **Leakage columns excluded:** {', '.join(info['leak']) if info['leak'] else 'none present'} "
                "(campaign outcomes are never used for clustering)\n"
                "- **Engineered:** Age, Customer_Tenure, Total_Spending, Total_Children, Education_Level, "
                "Living_With_Partner (only where source columns exist)\n"
                f"- **Features used ({len(feats)}):** {', '.join(feats)}\n"
                "- **Scaling:** StandardScaler → **PCA:** 3 components → **Clustering** on PCA space")

# ----------------------------------------------------------------- overview
with tabs[1]:
    a, b = st.columns(2)
    cnt = df.groupby("Segment").size().reset_index(name="Customers")
    a.plotly_chart(px.bar(cnt, x="Segment", y="Customers", color="Segment", color_discrete_map=cmap,
                          title="Customer distribution by cluster").update_layout(showlegend=False))
    b.plotly_chart(px.scatter(df.sample(min(len(df), 2000), random_state=1), x="Income", y="Total_Spending",
                              color="Segment", color_discrete_map=cmap, opacity=0.7,
                              title="Income vs Total Spending"))
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="Total_Spending", nbins=40, color_discrete_sequence=[PALETTE[0]],
                                title="Spending distribution"))
    b.plotly_chart(px.histogram(df, x="Recency", nbins=40, color_discrete_sequence=[PALETTE[1]],
                                title="Recency distribution (days since last purchase)"))

# ----------------------------------------------------------------- segmentation
with tabs[2]:
    st.subheader("🔬 Model Evaluation")
    sil, db = (res["km_sil"], res["km_db"]) if use_km else (res["ag_sil"], res["ag_db"])
    e = st.columns(2)
    e[0].metric(f"Silhouette Score ({model_name})", f"{sil:.3f}")
    e[1].metric(f"Davies-Bouldin Score ({model_name})", f"{db:.3f}")
    st.markdown("**Silhouette:** higher is better (well-separated clusters). "
                "**Davies-Bouldin:** lower is better (compact, distinct clusters).")
    comp = pd.DataFrame({"Model": ["K-Means", "Agglomerative"],
                         "Silhouette": [round(res["km_sil"], 4), round(res["ag_sil"], 4)],
                         "Davies-Bouldin": [round(res["km_db"], 4), round(res["ag_db"], 4)]})
    st.dataframe(comp, hide_index=True)
    best = "K-Means" if res["km_sil"] >= res["ag_sil"] else "Agglomerative Clustering"
    st.success(f"🏆 Recommended Model: {best} (highest silhouette score for k = {k})")

    st.subheader("🌐 PCA Cluster Visualization (3D)")
    pc = pd.DataFrame(Z, columns=["PC1", "PC2", "PC3"])
    pc["Segment"] = df["Segment"].values
    fig = px.scatter_3d(pc, x="PC1", y="PC2", z="PC3", color="Segment", color_discrete_map=cmap, opacity=0.8)
    fig.update_traces(marker_size=3)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), legend_title_text="")
    st.plotly_chart(fig)
    st.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum() * 100:.1f}% "
            f"(3 components) · Model shown: {model_name}")

# ----------------------------------------------------------------- profiles
with tabs[3]:
    st.subheader("Segment Profiles (cluster averages)")
    shown = prof.rename(index=seg_names)
    shown.index.name = "Segment"
    st.dataframe(shown.round(1))
    st.subheader("🎯 Marketing Recommendations")
    cols = st.columns(2)
    recs = []
    for i, c in enumerate(prof.index):
        label, desc, strategy, _ = SEGMENTS[seg_keys[c]]
        r = prof.loc[c]
        channel = "web" if r["NumWebPurchases"] > r["NumStorePurchases"] else "in-store"
        meta = (f"{int(r['Customers']):,} customers · avg income {r['Income']:,.0f} · "
                f"avg spend {r['Total_Spending']:,.0f} · recency {r['Recency']:.0f} days · "
                f"prefers {channel} purchases")
        recs.append({"Segment": seg_names[c], "Customers": int(r["Customers"]), "Description": desc,
                     "Recommended Strategy": f"{strategy} Prioritise the {channel} channel."})
        cols[i % 2].markdown(
            f'<div class="card"><h4>{seg_names[c]}</h4><p>{desc}</p>'
            f'<p><b>Recommendation:</b> {strategy} Prioritise the {channel} channel.</p>'
            f'<p class="meta">{meta}</p></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- explorer
with tabs[4]:
    st.subheader("🔎 Customer Explorer")
    st.caption("Uses the already-trained K-Means pipeline (scaler → PCA → K-Means). No separate model.")
    clip = lambda v, lo, hi: int(min(max(round(v), lo), hi))
    x1, x2 = st.columns(2)
    inp = {
        "Income": x1.number_input("Income", 0.0, 600000.0, float(X["Income"].median()), 1000.0),
        "Age": x1.slider("Age", 18, 90, clip(X["Age"].median(), 18, 90)),
        "Total_Spending": x1.number_input("Total Spending", 0.0, 100000.0, float(X["Total_Spending"].median()), 50.0),
        "Recency": x2.slider("Recency (days)", 0, max(100, int(X["Recency"].max())), clip(X["Recency"].median(), 0, 100)),
        "NumWebPurchases": x2.slider("Web purchases", 0, 30, clip(X["NumWebPurchases"].median(), 0, 30)),
        "NumStorePurchases": x2.slider("Store purchases", 0, 30, clip(X["NumStorePurchases"].median(), 0, 30)),
        "Total_Children": x2.slider("Total children", 0, 5, clip(X["Total_Children"].median(), 0, 5)),
    }
    row = X.median().to_frame().T  # features not asked for default to dataset median
    for key, val in inp.items():
        row[key] = float(val)
    cl = int(res["km"].predict(pca.transform(scaler.transform(row[feats])))[0])
    seg = SEGMENTS[km_keys[cl]]
    st.markdown(f'<div class="card"><p class="meta">Predicted Segment</p><h4>{km_names[cl]}</h4>'
                f'<p><b>Recommended Action:</b> {seg[3]}</p></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- export
with tabs[5]:
    st.subheader("Download results")
    out_cols = ["Cluster", "Segment"] + [c for c in feats if c in df.columns]
    base = df[[c for c in ("ID",) if c in df.columns] + out_cols]
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Clustered customers CSV", base.to_csv(index=False).encode(), "clustered_customers.csv", "text/csv")
    d2.download_button("⬇️ Segment profiles CSV", shown.round(2).to_csv().encode(), "segment_profiles.csv", "text/csv")
    d3.download_button("⬇️ Recommendations CSV", pd.DataFrame(recs).to_csv(index=False).encode(),
                       "recommendations.csv", "text/csv")
    st.dataframe(base.head(20))