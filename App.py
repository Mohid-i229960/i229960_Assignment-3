import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="💰 Finance Forecast App", layout="wide")

# Custom CSS for background and buttons
st.markdown("""
    <style>
        body {
            background-color: #fcf8f3;
        }
        .stButton>button {
            background-color: #ff8c00;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Finance Forecast App using Linear Regression")

# Initialize session state variables
for var in ["data_loaded", "features_ready", "split_done", "model_trained", "model",
            "X_train", "X_test", "y_train", "y_test", "data"]:
    if var not in st.session_state:
        st.session_state[var] = False if "data" not in var else pd.DataFrame()

# Welcome Section
st.markdown("### Unlock the future of financial insights with Machine Learning!")
st.image("https://media.giphy.com/media/SWoSkN6DxTszqIKEqv/giphy.gif", width=500)
st.markdown("👉 Use the sidebar to fetch stock data and build your ML model.")

# Sidebar for Input
st.sidebar.header("📥 Input Data")
ticker = st.sidebar.text_input("Enter Ticker (e.g., MSFT, NFLX)", "MSFT")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2022-01-01"))

if st.sidebar.button("Fetch Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if not data.empty:
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.success("✅ Data fetched successfully!")
        st.dataframe(data.tail())
    else:
        st.error("❌ Could not fetch data. Please check your inputs.")

# Feature Engineering
if st.button("Feature Engineering"):
    if st.session_state.data_loaded:
        df = st.session_state.data.copy()
        if "Adj Close" in df.columns:
            df["Return"] = df["Adj Close"].pct_change()
            df["Lag1"] = df["Return"].shift(1)
            df.dropna(inplace=True)
            st.session_state.data = df
            st.session_state.features_ready = True
            st.line_chart(df["Return"])
            st.success("✅ Features engineered.")
        else:
            st.error("❌ 'Adj Close' not found in columns.")
    else:
        st.warning("⚠️ Please fetch data first.")

# Preprocessing
if st.button("Preprocess Data"):
    if st.session_state.data_loaded:
        df = st.session_state.data.dropna()
        st.session_state.data = df
        st.write(df.describe())
        st.success("✅ Data cleaned.")
    else:
        st.warning("⚠️ Please fetch data first.")

# Train/Test Split
if st.button("Train/Test Split"):
    if st.session_state.features_ready:
        data = st.session_state.data
        X = data[["Lag1"]]
        y = data["Return"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.split_done = True
        fig = px.pie(values=[len(X_train), len(X_test)], names=["Train", "Test"], title="Train/Test Split")
        st.plotly_chart(fig)
        st.success("✅ Split successful.")
    else:
        st.warning("⚠️ Please perform feature engineering first.")

# Model Training
if st.button("Train Model"):
    if st.session_state.split_done:
        model = LinearRegression()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model
        st.session_state.model_trained = True
        st.success("✅ Model trained.")
    else:
        st.warning("⚠️ Perform train/test split first.")

# Evaluation
if st.button("Evaluate Model"):
    if st.session_state.model_trained:
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        mse = mean_squared_error(st.session_state.y_test, y_pred)
        r2 = r2_score(st.session_state.y_test, y_pred)
        st.metric("📉 Mean Squared Error", f"{mse:.6f}")
        st.metric("📈 R² Score", f"{r2:.4f}")
        df_res = pd.DataFrame({"Actual": st.session_state.y_test, "Predicted": y_pred})
        st.line_chart(df_res)
        fig = px.scatter(df_res, x="Actual", y="Predicted", title="📊 Actual vs Predicted")
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ Train model before evaluating.")
