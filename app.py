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
# STYLE
# ------------------------------------------------
st.markdown("""
<style>
.main-title{
font-size:42px;
font-weight:700;
text-align:center;
}
.subtitle{
text-align:center;
color:gray;
margin-bottom:30px;
}
div[data-testid="stMetric"]{
background-color:white;
border-radius:12px;
padding:10px;
box-shadow:0px 4px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER + GIF ANIMATION
# ------------------------------------------------
st.markdown('<div class="main-title">Dengue Outbreak Dynamics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Stochastic Analysis and Prediction of Dengue Cases</div>', unsafe_allow_html=True)

# Display GIF (place mosquito.gif in the same folder as app.py)
st.image("mosquito.gif", width=150)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
df = pd.read_csv("clean_dengue_india_regions2.csv")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df = df.dropna()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.header("Controls")
regions = sorted(df["Region"].unique())
region = st.sidebar.selectbox("Select Region", regions)
data = df[df["Region"] == region].sort_values("Year")

# Monte Carlo sliders
simulations = st.sidebar.slider("Monte Carlo Simulations", min_value=50, max_value=1000, value=200, step=50)
noise_std = st.sidebar.slider("Random Noise Std Dev", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

# ------------------------------------------------
# GROWTH RATE
# ------------------------------------------------
data["growth"] = data["Cases"].pct_change()
growth = data["growth"].replace([np.inf,-np.inf],np.nan).dropna()
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
c1,c2 = st.columns(2)
c1.metric("Growth Rate 📈", round(avg_growth,3))
c2.metric("Lyapunov Exponent 🧮", round(lyapunov,4))

# ------------------------------------------------
# STABILITY CLASSIFICATION
# ------------------------------------------------
if lyapunov < -0.01:
    status="Declining"
elif -0.01 <= lyapunov <= 0.01:
    status="Stable"
elif 0.01 < lyapunov <= 0.08:
    status="Growing"
else:
    status="Volatile"

st.metric("System Stability ⚖️", status)

# ------------------------------------------------
# YEARWISE CASE GRAPH
# ------------------------------------------------
st.header("Year-wise Dengue Cases")
color_scale = px.colors.sequential.Pinkyl
fig_bar = px.bar(
    data,
    x="Year",
    y="Cases",
    color="Cases",
    color_continuous_scale=color_scale,
    labels={"Cases":"Number of Cases"}
)
fig_bar.update_layout(template="plotly_white")
st.plotly_chart(fig_bar,use_container_width=True)

# ------------------------------------------------
# ROLLING TREND
# ------------------------------------------------
st.header("Smoothed Outbreak Trend")
data["rolling"]=data["Cases"].rolling(3).mean()
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=data["Year"],
    y=data["Cases"],
    mode="lines+markers",
    name="Actual Cases",
    line=dict(color="#ff4da6")
))
fig_trend.add_trace(go.Scatter(
    x=data["Year"],
    y=data["rolling"],
    mode="lines",
    name="3-Year Moving Avg",
    line=dict(color="#7a0177",width=4)
))
fig_trend.update_layout(template="plotly_white")
st.plotly_chart(fig_trend,use_container_width=True)

# ------------------------------------------------
# HEATMAP
# ------------------------------------------------
st.header("Dengue Outbreak Heatmap")
heatmap_df = df.pivot_table(index="Region", columns="Year", values="Cases", aggfunc="sum")
fig_heat = px.imshow(
    heatmap_df,
    color_continuous_scale=color_scale,
    aspect="auto",
    labels=dict(x="Year",y="Region",color="Cases")
)
fig_heat.update_layout(height=700,template="plotly_white")
st.plotly_chart(fig_heat,use_container_width=True)

# ------------------------------------------------
# MONTE CARLO SIMULATION
# ------------------------------------------------
st.header("Monte Carlo Outbreak Simulation")
st.latex(r"Cases_{t+1}=Cases_t(1+G+\epsilon)")

last_cases=data["Cases"].iloc[-1]
last_year=int(data["Year"].max())
future_years=5
years=list(range(last_year+1,last_year+future_years+1))
paths=[]

for s in range(simulations):
    current=last_cases
    path=[]
    for y in years:
        noise=np.random.normal(0,noise_std)
        current=current*(1+avg_growth+noise)
        path.append(current)
    paths.append(path)

paths=np.array(paths)
mean_path=paths.mean(axis=0)
upper=np.percentile(paths,95,axis=0)
lower=np.percentile(paths,5,axis=0)

fig_sim=go.Figure()
fig_sim.add_trace(go.Scatter(x=years,y=upper,line=dict(width=0),showlegend=False))
fig_sim.add_trace(go.Scatter(
    x=years,
    y=lower,
    fill="tonexty",
    fillcolor="rgba(255,0,150,0.2)",
    line=dict(width=0),
    name="Uncertainty"
))
fig_sim.add_trace(go.Scatter(
    x=years,
    y=mean_path,
    mode="lines+markers",
    line=dict(color="#ff4da6",width=4),
    name="Expected Cases"
))
fig_sim.update_layout(template="plotly_white", xaxis_title="Year", yaxis_title="Predicted Cases")
st.plotly_chart(fig_sim,use_container_width=True)

# ------------------------------------------------
# FUTURE PREDICTION
# ------------------------------------------------
st.header("Future Growth Prediction")
st.latex(r"Cases_{t+1}=Cases_t(1+G)")

future=[]
current=last_cases
for i in range(1,6):
    current=current*(1+avg_growth)
    future.append({"Year":last_year+i,"Cases":current})

future_df=pd.DataFrame(future)
combined=pd.concat([data[["Year","Cases"]],future_df])
combined["Type"]=["Actual"]*len(data)+["Predicted"]*len(future_df)

fig_pred=px.line(
    combined,
    x="Year",
    y="Cases",
    color="Type",
    markers=True,
    color_discrete_sequence=["#ff4da6","#ff99cc"]
)
fig_pred.update_layout(template="plotly_white")
st.plotly_chart(fig_pred,use_container_width=True)

# ------------------------------------------------
# DATASET
# ------------------------------------------------
st.header("Dataset")
st.dataframe(data)
