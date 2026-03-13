# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Dengue Outbreak Analytics",
    page_icon="🦟",
    layout="wide"
)

# ------------------------------------------------
# STYLE + NEUTRAL GRADIENT BACKGROUND
# ------------------------------------------------
st.markdown("""
<style>
/* Full-page neutral gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f5f5f5 0%, #ffffff 50%, #f0f0f0 100%);
    background-attachment: fixed;
}

/* Dashboard title and subtitle */
.main-title{
    font-size:42px;
    font-weight:700;
    color:#333333;
}
.subtitle{
    color:#555555;
    margin-bottom:30px;
}

/* Metrics cards styling */
div[data-testid="stMetric"]{
    background-color:rgba(255,255,255,0.95);
    border-radius:12px;
    padding:10px;
    box-shadow:0px 2px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER + MOSQUITO GIF NEXT TO TITLE
# ------------------------------------------------
gif_url = "https://raw.githubusercontent.com/karleaabha-byte/dengue_interactive/refs/heads/main/mosquito.gif"

col1, col2 = st.columns([1, 5])
with col1:
    st.image(gif_url, width=120)
with col2:
    st.markdown('<div class="main-title">Dengue Outbreak Dynamics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Stochastic Analysis and Prediction of Dengue Cases</div>', unsafe_allow_html=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
df = pd.read_csv("clean_dengue_india_regions2.csv")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna()

# ------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------
st.sidebar.header("Controls")
regions = sorted(df["Region"].unique())
region = st.sidebar.selectbox("Select Region", regions)
data = df[df["Region"] == region].sort_values("Year")

simulations = st.sidebar.slider("Monte Carlo Simulations", 50, 1000, 200)
noise_std = st.sidebar.slider("Random Noise Std Dev", 0.01, 0.2, 0.05)

# ------------------------------------------------
# GROWTH RATE
# ------------------------------------------------
data["growth"] = data["Cases"].pct_change()
growth = data["growth"].replace([np.inf, -np.inf], np.nan).dropna()
avg_growth = growth.median() if len(growth) > 0 else 0

# ------------------------------------------------
# LYAPUNOV EXPONENT
# ------------------------------------------------
st.header("Lyapunov Stability Analysis")
st.latex(r"\lambda = mean(\log(1 + growth))")
growth_clean = growth[growth > -0.99]
lyapunov = np.mean(np.log(1 + growth_clean)) if len(growth_clean) > 0 else 0

# ------------------------------------------------
# METRICS
# ------------------------------------------------
c1, c2 = st.columns(2)
c1.metric("Growth Rate 📈", round(avg_growth, 3))
c2.metric("Lyapunov Exponent 🧮", round(lyapunov, 4))

# ------------------------------------------------
# STABILITY CLASSIFICATION
# ------------------------------------------------
if lyapunov < -0.01:
    status = "Declining"
elif -0.01 <= lyapunov <= 0.01:
    status = "Stable"
elif 0.01 < lyapunov <= 0.08:
    status = "Growing"
else:
    status = "Volatile"
st.metric("System Stability ⚖️", status)

# ------------------------------------------------
# YEARWISE CASE GRAPH
# ------------------------------------------------
st.header("Year-wise Dengue Cases")
fig_bar = px.bar(
    data,
    x="Year",
    y="Cases",
    color="Cases",
    color_continuous_scale=px.colors.sequential.Greys,
    labels={"Cases":"Number of Cases","Year":"Year"}
)
fig_bar.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------------
# ROLLING TREND
# ------------------------------------------------
st.header("Smoothed Outbreak Trend")
data["rolling"] = data["Cases"].rolling(3).mean()
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=data["Year"],
    y=data["Cases"],
    mode="lines+markers",
    name="Actual Cases",
    line=dict(color="#6c757d")
))
fig_trend.add_trace(go.Scatter(
    x=data["Year"],
    y=data["rolling"],
    mode="lines",
    name="3-Year Moving Avg",
    line=dict(color="#343a40", width=4)
))
fig_trend.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------------------
# HEATMAP
# ------------------------------------------------
st.header("Dengue Outbreak Heatmap")
heatmap_df = df.pivot_table(
    index="Region",
    columns="Year",
    values="Cases",
    aggfunc="sum"
)
fig_heat = px.imshow(
    heatmap_df,
    color_continuous_scale=px.colors.sequential.Greys,
    aspect="auto",
    labels=dict(x="Year", y="Region", color="Cases")
)
fig_heat.update_layout(height=700, template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------------------------
# MONTE CARLO SIMULATION
# ------------------------------------------------
st.header("Monte Carlo Outbreak Simulation")
st.latex(r"Cases_{t+1}=Cases_t(1+G+\epsilon)")

last_cases = data["Cases"].iloc[-1]
last_year = int(data["Year"].max())
future_years = 5
years = list(range(last_year + 1, last_year + future_years + 1))
paths = []

for s in range(simulations):
    current = last_cases
    path = []
    for y in years:
        noise = np.random.normal(0, noise_std)
        current = current * (1 + avg_growth + noise)
        path.append(current)
    paths.append(path)

paths = np.array(paths)
mean_path = paths.mean(axis=0)
upper = np.percentile(paths, 95, axis=0)
lower = np.percentile(paths, 5, axis=0)

fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(x=years, y=upper, line=dict(width=0), showlegend=False))
fig_sim.add_trace(go.Scatter(
    x=years,
    y=lower,
    fill="tonexty",
    fillcolor="rgba(108,117,125,0.2)",
    line=dict(width=0),
    name="Uncertainty"
))
fig_sim.add_trace(go.Scatter(
    x=years,
    y=mean_path,
    mode="lines+markers",
    line=dict(color="#6c757d", width=4),
    name="Expected Cases"
))
fig_sim.update_layout(
    template="plotly_white",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Year",
    yaxis_title="Predicted Cases"
)
st.plotly_chart(fig_sim, use_container_width=True)

# ------------------------------------------------
# FUTURE PREDICTION
# ------------------------------------------------
st.header("Future Growth Prediction")
future = []
current = last_cases
for i in range(1, 6):
    current = current * (1 + avg_growth)
    future.append({"Year": last_year + i, "Cases": current})

future_df = pd.DataFrame(future)
combined = pd.concat([data[["Year", "Cases"]], future_df])
combined["Type"] = ["Actual"] * len(data) + ["Predicted"] * len(future_df)

fig_pred = px.line(
    combined,
    x="Year",
    y="Cases",
    color="Type",
    markers=True,
    color_discrete_map={"Actual":"#6c757d","Predicted":"#adb5bd"}
)
fig_pred.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_pred, use_container_width=True)

# ------------------------------------------------
# DATASET
# ------------------------------------------------
st.header("Dataset")
st.dataframe(data)
