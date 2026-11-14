import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Page Setup
st.set_page_config(page_title="Sales Prediction & Analysis", layout="wide")
st.title("📈 Sales Prediction & Time-Series Analysis")
st.write("Upload historical sales or marketing data to clean records, spot trends, and forecast future revenue using Ridge and Lasso Regression.")

# 2. Sidebar Controls
st.sidebar.header("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

def generate_sample_data():
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
    np.random.seed(42)
    trend = np.linspace(100, 250, 365)
    seasonality = 30 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7)
    noise = np.random.normal(0, 12, 365)
    sales = np.round(trend + seasonality + noise, 2).astype(float)
    sales[15] = np.nan
    sales[100] = np.nan
    return pd.DataFrame({"Date": dates, "Sales": sales})

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(raw_df):,} rows & {len(raw_df.columns)} columns!")
else:
    raw_df = generate_sample_data()
    st.sidebar.info("Using built-in sample data. Upload your CSV above.")

# Auto-detect Date & Revenue/Sales columns
cols = raw_df.columns.tolist()
default_date_idx = cols.index("date") if "date" in cols else 0
default_sales_idx = cols.index("revenue_usd") if "revenue_usd" in cols else (cols.index("Sales") if "Sales" in cols else min(1, len(cols)-1))

date_col = st.sidebar.selectbox("Select Date Column", cols, index=default_date_idx)
sales_col = st.sidebar.selectbox("Select Sales / Revenue Column", cols, index=default_sales_idx)

# Optional Region / Dimension Filter (e.g., East, North, South, West)
dim_col = "dimension" if "dimension" in cols else None
selected_region = "All Regions (Combined)"
if dim_col:
    regions = ["All Regions (Combined)"] + sorted(raw_df[dim_col].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("Filter by Region / Dimension", regions)

st.sidebar.header("2. Model Settings")
model_choice = st.sidebar.selectbox("Choose Regression Model", ["Compare Both", "Ridge Regression", "Lasso Regression"])
alpha = st.sidebar.slider("Regularization Strength (Alpha)", min_value=0.1, max_value=1000.0, value=10.0, step=5.0)
forecast_days = st.sidebar.slider("Days to Forecast Ahead", min_value=7, max_value=90, value=30)

# Detect extra marketing/pricing columns if available
extra_driver_cols = [c for c in cols if c.endswith("_spend_usd") or c in ["price_index", "promo_discount_pct"]]
use_extra_drivers = False
if extra_driver_cols:
    use_extra_drivers = st.sidebar.checkbox(f"Include {len(extra_driver_cols)} Marketing & Pricing Drivers", value=True)

# 3. Data Cleaning & Region Aggregation
work_df = raw_df.copy()
if dim_col and selected_region != "All Regions (Combined)":
    work_df = work_df[work_df[dim_col] == selected_region]

# Parse DD-MM-YYYY or YYYY-MM-DD safely
work_df["clean_date"] = pd.to_datetime(work_df[date_col], dayfirst=True, errors="coerce")
work_df = work_df.dropna(subset=["clean_date"])

# Group by date (sums sales & ad spend across regions if "All Regions" is chosen)
agg_rules = {sales_col: "sum"}
if use_extra_drivers:
    for c in extra_driver_cols:
        agg_rules[c] = "mean" if c in ["price_index", "promo_discount_pct"] else "sum"

df = work_df.groupby("clean_date", as_index=False).agg(agg_rules).sort_values("clean_date")
df = df.rename(columns={"clean_date": "date", sales_col: "sales"})

missing_before = df["sales"].isna().sum()
df = df.set_index("date").asfreq("D")
df = df.interpolate(method="linear").bfill().ffill().reset_index()

with st.expander("🧹 Step 1: Data Cleaning Summary (Click to expand)", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Clean Days", f"{len(df):,}")
    c2.metric("Missing Sales Fixed", int(missing_before))
    c3.metric("Average Daily Revenue", f"${df['sales'].mean():,.2f}")
    st.dataframe(df.head(10), use_container_width=True)

# 4. Trend Identification
st.subheader(f"📊 Step 2: Historical Sales & Trend Analysis ({selected_region})")
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=df["date"], y=df["sales"], mode="lines", name="Daily Sales", line=dict(color="#94a3b8", width=1.5)))

if len(df) >= 14:
    decomp = seasonal_decompose(df.set_index("date")["sales"], model="additive", period=7)
    trend_line = decomp.trend.bfill().ffill()
    fig_trend.add_trace(go.Scatter(x=df["date"], y=trend_line, mode="lines", name="Underlying Trend", line=dict(color="#2563eb", width=3)))

fig_trend.update_layout(xaxis_title="Date", yaxis_title="Sales / Revenue", height=380, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_trend, use_container_width=True)

# 5. Time-Series Feature Engineering
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["time_index"] = np.arange(len(df))
df["lag_1"] = df["sales"].shift(1)
df["lag_7"] = df["sales"].shift(7)
df["rolling_mean_7"] = df["sales"].shift(1).rolling(window=7).mean()

model_df = df.dropna().reset_index(drop=True)
ts_features = ["day_of_week", "month", "day_of_year", "time_index", "lag_1", "lag_7", "rolling_mean_7"]
features = ts_features + (extra_driver_cols if use_extra_drivers else [])

X = model_df[features]
y = model_df["sales"]

# Chronological 80/20 Train-Test Split
split_idx = int(len(model_df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Recent 14-day average of marketing/pricing drivers to project future baseline
recent_driver_defaults = {c: float(model_df[c].iloc[-14:].mean()) for c in extra_driver_cols} if use_extra_drivers else {}

def train_and_forecast(model_type):
    regressor = Ridge(alpha=alpha) if model_type == "Ridge" else Lasso(alpha=alpha, max_iter=20000)
    pipeline = make_pipeline(StandardScaler(), regressor)
    pipeline.fit(X_train, y_train)
    
    test_preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    history = model_df[["date", "sales"]].copy()
    future_dates, future_preds = [], []
    last_date = history["date"].iloc[-1]
    last_idx = model_df["time_index"].iloc[-1]
    
    for step in range(1, forecast_days + 1):
        next_date = last_date + pd.Timedelta(days=step)
        row_dict = {
            "day_of_week": next_date.dayofweek,
            "month": next_date.month,
            "day_of_year": next_date.dayofyear,
            "time_index": last_idx + step,
            "lag_1": history["sales"].iloc[-1],
            "lag_7": history["sales"].iloc[-7],
            "rolling_mean_7": history["sales"].iloc[-7:].mean(),
            **recent_driver_defaults
        }
        next_row = pd.DataFrame([row_dict])[features]
        pred = float(pipeline.predict(next_row)[0])
        future_dates.append(next_date)
        future_preds.append(pred)
        history = pd.concat([history, pd.DataFrame([{"date": next_date, "sales": pred}])], ignore_index=True)
        
    return test_preds, mae, rmse, future_dates, future_preds, regressor.coef_

# 6. Model Results & Forecasting Visualization
st.subheader("🔮 Step 3: Future Sales Forecast & Model Accuracy")
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=model_df["date"], y=model_df["sales"], mode="lines", name="Actual Sales", line=dict(color="#1e293b")))

models_to_run = ["Ridge", "Lasso"] if model_choice == "Compare Both" else [model_choice.split()[0]]
colors = {"Ridge": "#10b981", "Lasso": "#f59e0b"}

metric_cols = st.columns(len(models_to_run) * 2)
coef_data = {}

for idx, m_name in enumerate(models_to_run):
    test_preds, mae, rmse, f_dates, f_preds, coefs = train_and_forecast(m_name)
    coef_data[m_name] = coefs
    
    metric_cols[idx*2].metric(f"{m_name} MAE (Avg Error)", f"${mae:,.2f}")
    metric_cols[idx*2 + 1].metric(f"{m_name} RMSE", f"${rmse:,.2f}")
    
    fig_forecast.add_trace(go.Scatter(
        x=model_df["date"].iloc[split_idx:], y=test_preds, mode="lines",
        name=f"{m_name} Test Fit", line=dict(color=colors[m_name], dash="dot")
    ))
    fig_forecast.add_trace(go.Scatter(
        x=f_dates, y=f_preds, mode="lines",
        name=f"{m_name} Future Forecast", line=dict(color=colors[m_name], width=3, dash="dash")
    ))

fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Sales / Revenue", height=420, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_forecast, use_container_width=True)

# 7. Feature Importance
st.subheader("🧠 Step 4: Top Drivers of Revenue (Feature Weights)")
fig_coef = go.Figure()
for m_name, coefs in coef_data.items():
    fig_coef.add_trace(go.Bar(x=features, y=coefs, name=f"{m_name} Weights", marker_color=colors[m_name]))
fig_coef.update_layout(barmode="group", yaxis_title="Impact on Revenue", height=360, margin=dict(l=20, r=20, t=20, b=80))
st.plotly_chart(fig_coef, use_container_width=True)