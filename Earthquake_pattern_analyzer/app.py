import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Earthquake Pattern Analysis",
    page_icon="🌍",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/earthquake.csv")

try:
    df = load_data()

except Exception as e:
    st.error(f"Unable to load earthquake.csv\n\n{e}")
    st.stop()

# ==========================================================
# CLEAN DATA
# ==========================================================

df.columns = df.columns.str.lower().str.strip()

required_cols = [
    "latitude",
    "longitude",
    "depth",
    "mag",
    "cluster"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🔎 Filters")

mag_range = st.sidebar.slider(
    "Magnitude Range",
    float(df["mag"].min()),
    float(df["mag"].max()),
    (
        float(df["mag"].min()),
        float(df["mag"].max())
    )
)

depth_range = st.sidebar.slider(
    "Depth Range",
    float(df["depth"].min()),
    float(df["depth"].max()),
    (
        float(df["depth"].min()),
        float(df["depth"].max())
    )
)

# ==========================================================
# FILTER DATA
# ==========================================================

filtered_df = df[
    (df["mag"] >= mag_range[0]) &
    (df["mag"] <= mag_range[1]) &
    (df["depth"] >= depth_range[0]) &
    (df["depth"] <= depth_range[1])
]

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
# 🌍 Earthquake Pattern Analysis & Seismic Hotspot Detection

Discover hidden seismic patterns using **DBSCAN clustering** and geospatial analytics.

This system **does not predict future earthquakes**.

Instead, it identifies:

- 🌋 Earthquake Hotspots
- 🌍 Similar Seismic Regions
- 📊 Earthquake Clusters
- ⚠️ Outlier Events
""")

st.divider()

# ==========================================================
# KPI SECTION
# ==========================================================

total_eq = len(filtered_df)

noise_points = (
    filtered_df["cluster"] == -1
).sum()

clusters_found = len(
    [c for c in filtered_df["cluster"].unique() if c != -1]
)

avg_mag = filtered_df["mag"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🌍 Earthquakes",
    f"{total_eq:,}"
)

c2.metric(
    "📍 Clusters",
    clusters_found
)

c3.metric(
    "⚠️ Outliers",
    f"{noise_points:,}"
)

c4.metric(
    "📊 Avg Magnitude",
    f"{avg_mag:.2f}"
)

st.divider()

# ==========================================================
# MAP
# ==========================================================

st.subheader("🗺️ Global Earthquake Distribution")

map_fig = px.scatter_geo(
    filtered_df,
    lat="latitude",
    lon="longitude",
    color="mag",
    size="mag",
    hover_data=[
        "depth",
        "cluster"
    ],
    projection="natural earth",
    color_continuous_scale="Turbo"
)

map_fig.update_layout(
    height=700,
    margin=dict(
        l=0,
        r=0,
        t=30,
        b=0
    ),
    coloraxis_colorbar=dict(
        title="Magnitude"
    )
)

st.plotly_chart(
    map_fig,
    use_container_width=True
)

st.caption(
    "Color represents earthquake magnitude while marker size indicates earthquake strength."
)

# ==========================================================
# ANALYSIS SECTION
# ==========================================================

left, right = st.columns(2)

with left:

    st.subheader("📈 Magnitude vs Depth")

    scatter_fig = px.scatter(
        filtered_df,
        x="mag",
        y="depth",
        color="cluster",
        hover_data=[
            "latitude",
            "longitude"
        ]
    )

    scatter_fig.update_layout(
        height=500
    )

    st.plotly_chart(
        scatter_fig,
        use_container_width=True
    )

with right:

    st.subheader("📊 Cluster Distribution")

    cluster_counts = (
        filtered_df["cluster"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Count"
    ]

    pie_fig = px.pie(
        cluster_counts,
        names="Cluster",
        values="Count",
        hole=0.5
    )

    pie_fig.update_layout(
        height=500
    )

    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )

st.divider()

# ==========================================================
# CLUSTER SUMMARY
# ==========================================================

st.subheader("📋 Cluster Summary")

summary = (
    filtered_df
    .groupby("cluster")
    .agg(
        Number_of_Earthquakes=("cluster", "count"),
        Average_Magnitude=("mag", "mean"),
        Average_Depth=("depth", "mean"),
        Maximum_Magnitude=("mag", "max")
    )
    .reset_index()
)

st.dataframe(
    summary.round(2),
    use_container_width=True
)

# ==========================================================
# INSIGHTS
# ==========================================================

st.subheader("🧠 Key Insights")

largest_cluster = (
    filtered_df["cluster"]
    .value_counts()
    .idxmax()
)

largest_size = (
    filtered_df["cluster"]
    .value_counts()
    .max()
)

outlier_pct = (
    noise_points /
    len(filtered_df)
) * 100

st.info(
    f"""
🔹 Largest Cluster: {largest_cluster}

🔹 Earthquakes in Largest Cluster: {largest_size}

🔹 Average Magnitude: {avg_mag:.2f}

🔹 Outlier Percentage: {outlier_pct:.2f}%

🔹 Distinct Seismic Patterns Found: {clusters_found}
"""
)

# ==========================================================
# DOWNLOAD
# ==========================================================

st.subheader("⬇️ Export Data")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="earthquake_clusters.csv",
    mime="text/csv"
)

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Built with Streamlit • Plotly • DBSCAN Clustering • Geospatial Analytics"
)