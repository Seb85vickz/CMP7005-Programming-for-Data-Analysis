
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # <--- NEW IMPORT
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# --- Page Configuration ---
st.set_page_config(
    page_title="India Air Quality Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Polishing ---
st.markdown("""
    <style>
    /* 1. Main Background: Light Red */
    .stApp {
        background-color: #FFCCCB;
    }

    /* 2. Sidebar Background: Light Sea Blue */
    [data-testid="stSidebar"] {
        background-color: #ADD8E6;
    }

    /* 3. Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }

    /* 4. Heading Colors */
    h1 {
        color: #2E4053;
    }
    h2 {
        color: #34495E;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading & Caching ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("all_cities_combined 18.csv")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except FileNotFoundError:
        return None

# --- Preprocessing Function ---
def preprocess_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    df = df.dropna(subset=['AQI'])
    return df

# --- Main Application Logic ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [" Home & Data Overview", " Exploratory Data Analysis", " Modelling & Prediction"])

    data = load_data()

    if data is None:
        st.error("Welcome to the AQI Machine Learning Prediction")
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        else:
            st.stop()

    # --- PAGE 1: HOME & DATA OVERVIEW ---
    if page == " Home & Data Overview":
        st.title(" India Air Quality Analysis Project")
        st.write("This interactive application monitors and forecasts air pollution using a real-world dataset from Indian cities (2015-2020).")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", data.shape[0])
        col2.metric("Total Cities", data['City'].nunique())
        col3.metric("Features", data.shape[1])

        st.divider()

        st.subheader(" Dataset Preview")
        with st.expander("View Raw Data"):
            st.dataframe(data.head(100), use_container_width=True)

        st.subheader(" Statistical Summary")
        with st.expander("View Descriptive Statistics"):
            st.dataframe(data.describe(), use_container_width=True)

        st.subheader(" Missing Values Analysis")
        missing_val = data.isnull().sum()
        fig_missing = px.bar(missing_val, x=missing_val.index, y=missing_val.values,
                             title="Missing Values Count per Feature",
                             labels={'y':'Count', 'x':'Feature'}, color=missing_val.values, color_continuous_scale='Reds')
        st.plotly_chart(fig_missing, use_container_width=True)

    # --- PAGE 2: EXPLORATORY DATA ANALYSIS (EDA) ---
    elif page == " Exploratory Data Analysis":
        st.title(" Exploratory Data Analysis")

        st.sidebar.subheader("Filter Settings")
        selected_city = st.sidebar.selectbox("Select City for Analysis", data['City'].unique())

        city_data = data[data['City'] == selected_city]

        st.subheader(f" Air Quality Trends in {selected_city}")
        pollutant = st.selectbox("Select Pollutant to visualize", ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI'])

        fig_line = px.line(city_data, x='Date', y=pollutant, title=f'{pollutant} Levels Over Time in {selected_city}',
                           markers=True)
        fig_line.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader(" Correlation Heatmap")
        st.write("Understand the relationship between different pollutants.")

        numeric_df = data.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Viridis',
                             title="Feature Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Distribution of Pollutants")
        col1, col2 = st.columns(2)

        with col1:
            dist_pollutant = st.selectbox("Select Pollutant for Histogram", numeric_df.columns, index=2)
            fig_hist = px.histogram(data, x=dist_pollutant, nbins=50, title=f"Distribution of {dist_pollutant}",
                                    color_discrete_sequence=['#3366CC'])
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_box = px.box(data, x='AQI_Bucket', y='PM2.5', title="PM2.5 Levels by AQI Category",
                             color='AQI_Bucket')
            st.plotly_chart(fig_box, use_container_width=True)

    # --- PAGE 3: MODELLING & PREDICTION ---
    elif page == " Modelling & Prediction":
        st.title(" AQI Prediction Models")
        st.write("Let's Compare different Machine Learning algorithms to predict Air Quality Index (AQI).")

        # Data Preparation
        df_model = preprocess_data(data.copy())
        features = ['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'SO2', 'O3']
        target = 'AQI'

        X = df_model[features]
        y = df_model[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- MODEL 1: RANDOM FOREST ---
        st.subheader("1. Random Forest Regressor")

        @st.cache_resource
        def train_rf_model(X_train, y_train):
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            return model

        with st.spinner("Training Random Forest..."):
            rf_model = train_rf_model(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_r2 = r2_score(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

        col1, col2 = st.columns(2)
        col1.metric("RF R² Score", f"{rf_r2:.2f}")
        col2.metric("RF RMSE", f"{rf_rmse:.2f}")

        st.divider()

        # --- MODEL 2: XGBOOST ---
        st.subheader("2. XGBoost Regressor")


        st.info("XGBoost Parameters: n_estimators=100, learning_rate=0.05, max_depth=8")

        @st.cache_resource
        def train_xgb_model(X_train, y_train):
            # Using parameters from the reference code
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=8, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)
            return model

        # Add a specific button to trigger XGBoost Training
        xgb_model = None
        if st.button("Train XGBoost Model"):
            with st.spinner("Training XGBoost..."):
                xgb_model = train_xgb_model(X_train, y_train)
                st.session_state['xgb_model'] = xgb_model # Save to session state

                xgb_pred = xgb_model.predict(X_test)
                xgb_r2 = r2_score(y_test, xgb_pred)
                xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

                st.success("XGBoost Trained Successfully!")

                # Show Metrics for XGBoost
                c1, c2 = st.columns(2)
                c1.metric("XGBoost R² Score", f"{xgb_r2:.2f}")
                c2.metric("XGBoost RMSE", f"{xgb_rmse:.2f}")

        # Check if model is already in session state (so it persists after interaction)
        if 'xgb_model' in st.session_state:
            xgb_model = st.session_state['xgb_model']

        st.divider()

        # --- PREDICTION INTERFACE (SHARED) ---
        st.subheader(" Make a Prediction")
        st.write("Adjust sliders to simulate pollutant levels, then choose a model to predict.")

        input_cols = st.columns(3)
        user_inputs = {}

        for i, feature in enumerate(features):
            with input_cols[i % 3]:
                min_val = float(df_model[feature].min())
                max_val = float(df_model[feature].max())
                mean_val = float(df_model[feature].mean())
                user_inputs[feature] = st.slider(f"{feature}", min_val, max_val, mean_val)

        input_df = pd.DataFrame([user_inputs])

        # Prediction Buttons
        p_col1, p_col2 = st.columns(2)

        # 1. RF Prediction
        with p_col1:
            if st.button("Predict with Random Forest"):
                prediction = rf_model.predict(input_df)[0]
                st.markdown(f"### RF Prediction: <span style='color:green'>{prediction:.2f}</span>", unsafe_allow_html=True)

                # AQI Status Logic
                if prediction <= 50: status = "Good"
                elif prediction <= 100: status = "Satisfactory"
                elif prediction <= 200: status = "Moderate"
                elif prediction <= 300: status = "Poor"
                elif prediction <= 400: status = "Very Poor"
                else: status = "Severe"
                st.info(f"Status: **{status}**")

        # 2. XGBoost Prediction
        with p_col2:
            if st.button("Predict with XGBoost"):
                if xgb_model:
                    prediction_xgb = xgb_model.predict(input_df)[0]

                    # Using the styling from the reference code for XGBoost output
                    bucket, color = "Severe", "darkred"
                    if prediction_xgb <= 50: bucket, color = "Good", "green"
                    elif prediction_xgb <= 100: bucket, color = "Satisfactory", "lightgreen"
                    elif prediction_xgb <= 200: bucket, color = "Moderate", "gold" # 'yellow' can be hard to read
                    elif prediction_xgb <= 300: bucket, color = "Poor", "orange"
                    elif prediction_xgb <= 400: bucket, color = "Very Poor", "red"

                    st.markdown(f"""
                        <div style="background-color: {color}; padding: 10px; border-radius: 10px; text-align: center;">
                            <h3 style="color: white; margin:0; text-shadow: 1px 1px 2px black;">XGB Pred: {prediction_xgb:.1f}</h3>
                            <p style="color: white; margin:0; font-weight:bold;">{bucket}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Please train the XGBoost model first!")

if __name__ == "__main__":
    main()
