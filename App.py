import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Streamlit app configuration
st.set_page_config(page_title="Finance Forecast", layout="centered")

# Custom Styling
st.markdown("""
    <style>
        .main { background-color: #ffffff; }
        .stButton button {
            background-color: #0066cc;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Finance Forecast App")
st.markdown("### A simple ML app to predict financial returns using Linear Regression.")
st.image("https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif", width=400)



# Initialize session state
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "features_ready" not in st.session_state:
    st.session_state.features_ready = False
if "split_done" not in st.session_state:
    st.session_state.split_done = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model" not in st.session_state:
    st.session_state.model = None

# Sidebar inputs
st.sidebar.header("📥 Input Data")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Fetch Data
if st.sidebar.button("Fetch Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if not data.empty:
        st.success("✅ Data loaded successfully!")
        st.dataframe(data.tail())
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.session_state.features_ready = False
        st.session_state.split_done = False
        st.session_state.model_trained = False
    else:
        st.error("❌ Failed to load data. Check ticker or date range.")

# Feature Engineering
if st.button("Feature Engineering"):
    if st.session_state.data_loaded:
        df = st.session_state.data.copy()
        adj_col = None
        for col in ["Adj Close", "Close"]:
            if col in df.columns:
                adj_col = col
                break

        if adj_col:
            df["Return"] = df[adj_col].pct_change()
            df["Lag1"] = df["Return"].shift(1)
            df.dropna(inplace=True)
            st.session_state.data = df
            st.session_state.features_ready = True
            st.line_chart(df["Return"])
            st.success(f"✅ Features engineered using '{adj_col}'.")
        else:
            st.error(f"❌ Neither 'Adj Close' nor 'Close' found in data. Columns: {df.columns.tolist()}")
    else:
        st.warning("⚠️ Please fetch data first.")

# Preprocessing
if st.button("Preprocessing"):
    if st.session_state.data_loaded:
        df = st.session_state.data.copy()
        df.dropna(inplace=True)
        st.session_state.data = df
        st.write(df.describe())
        st.success("✅ Missing values removed.")
    else:
        st.warning("⚠️ Please fetch data first.")

# Train/Test Split
if st.button("Train/Test Split"):
    if st.session_state.features_ready:
        df = st.session_state.data.copy()
        X = df[["Lag1"]]
        y = df["Return"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.split_done = True
        st.success("✅ Train/test split completed.")
        fig = px.pie(values=[len(X_train), len(X_test)], names=["Train", "Test"], title="Train/Test Split")
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ Perform feature engineering first.")

# Train Model
if st.button("Train Model"):
    if st.session_state.split_done:
        model = LinearRegression()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model
        st.session_state.model_trained = True
        st.success("✅ Model trained successfully.")
    else:
        st.warning("⚠️ Please perform the train/test split first.")

# Evaluate Model
if st.button("Evaluate Model"):
    if st.session_state.model_trained:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.write(f"**Mean Squared Error (MSE):** {mse:.6f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Plot Actual vs Predicted
        results = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
        fig = px.scatter(results, x="Actual", y="Predicted", title="Actual vs Predicted Returns")
        st.plotly_chart(fig)

        # Residuals
        residuals = y_test - preds
        fig2 = px.histogram(residuals, nbins=50, title="Residuals Distribution")
        st.plotly_chart(fig2)
    else:
        st.warning("⚠️ Train the model first.")
