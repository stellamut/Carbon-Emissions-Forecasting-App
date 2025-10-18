import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Custom CSS for Theming and Font ---
st.markdown("""
<style>
    /* Font: Futura Family */
    @font-face {
        font-family: 'Futura';
        src: url('https://fonts.cdnfonts.com/s/7213/FuturaLT-Book.woff') format('woff'); /* Or a local path if you host the font */
        font-weight: normal;
        font-style: normal;
    }
    @font-face {
        font-family: 'Futura';
        src: url('https://fonts.cdnfonts.com/s/7213/FuturaLT-Bold.woff') format('woff');
        font-weight: bold;
        font-style: normal;
    }

    body {
        font-family: 'Futura', sans-serif;
        color: #1A431A; /* Dark Green */
        background-color: #FDF9F3; /* A very light cream/off-white */
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Futura', sans-serif;
        color: #1A431A; /* Dark Green for headings */
    }

    .stButton>button {
        background-color: #FFBF00; /* Golden Yellow for buttons */
        color: #FFFFFF; /* White text on buttons */
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #e6a800; /* Slightly darker yellow on hover */
        color: #FFFFFF;
    }

    /* Streamlit Metric Styling */
    [data-testid="stMetric"] > div {
        background-color: #FFDAB9; /* Peach background for metrics */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        color: #1A431A; /* Dark Green text for metrics */
    }
    [data-testid="stMetricLabel"] {
        color: #1A431A; /* Dark Green */
        font-size: 1.1em;
        font-weight: bold;
    }
    [data-testid="stMetricValue"] {
        color: #1A431A; /* Dark Green */
        font-size: 2.5em;
        font-weight: bold;
    }
    [data-testid="stMetricDelta"] {
        color: #9ACD32; /* Lime Green for positive delta */
        /* Use inverse color for negative delta by default */
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1aumjbt { /* Streamlit's sidebar classes */
        background-color: #FDF9F3; /* Light cream for sidebar */
    }
    .sidebar .sidebar-content {
        background-color: #FDF9F3; /* Light cream for sidebar content */
    }

    /* Info/Warning blocks */
    .stAlert {
        background-color: #FDF9F3; /* Use a neutral light background */
        border-left: 5px solid #FFBF00; /* Golden Yellow border for info */
        color: #1A431A; /* Dark Green text */
    }
    .stAlert > div > span { /* For the icon */
        color: #FFBF00 !important; /* Golden Yellow icon */
    }

    /* General text color for all other elements */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        color: #1A431A; /* Dark Green */
    }
    
</style>
""", unsafe_allow_html=True)

# --- End of Custom CSS ---


# Load the saved assets (rest of your app.py)
@st.cache_resource
def load_assets():
    model = joblib.load('xgb_co2_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = ['GDP', 'Elec_Cons_PC', 'Population', 'Renewable_Share', 'Coal_Share', 'CO2_Emissions_Lag1']
    return model, scaler, features

model, scaler, FEATURES = load_assets()

st.title("🌍 SDG 13: Proactive Carbon Emission Forecaster")
st.subheader("Simulate future CO₂ emissions based on key socio-economic indicators.")

# ... (rest of your Streamlit app code as designed before) ...
# Example:
with st.sidebar:
    st.header("Policy Levers")
    st.write("Adjust the projected values for the next year to simulate impact.")
    
    input_data = {
        'GDP': st.number_input("Projected GDP (US$ Trillions)", value=2.0, min_value=0.1),
        'Elec_Cons_PC': st.number_input("Elec. Consumption (kWh/capita)", value=5000, min_value=100),
        'Population': st.number_input("Projected Population (Millions)", value=300, min_value=10),
        'Renewable_Share': st.slider("Renewable Share in Energy (%)", value=15.0, min_value=0.0, max_value=100.0, step=0.1),
        'Coal_Share': st.slider("Coal Share in Energy (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.1),
        'CO2_Emissions_Lag1': st.number_input("Previous Year's CO₂ Emissions (kt)", value=3000000, min_value=0)
    }

if st.button("Forecast Emissions"):
    input_df = pd.DataFrame([input_data])
    X_pred = input_df[FEATURES]
    X_pred_scaled = scaler.transform(X_pred)
    predicted_co2 = model.predict(X_pred_scaled)[0]

    st.markdown("---")
    st.metric(
        label="Projected CO₂ Emissions (kt)",
        value=f"{predicted_co2:,.0f} kt",
        delta=f"{(predicted_co2 - input_data['CO2_Emissions_Lag1']):,.0f} kt Change",
        delta_color="inverse"
    )

    st.markdown("## 🔍 Policy Impact Analysis (Feature Importance)")
    st.write("This shows which factors the model weighted most heavily for the prediction:")
    
    importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.bar_chart(importance_df.set_index('Feature'))
    
    st.info(
        "**Interpretation:** If 'Coal_Share' has high importance, policies targeting the reduction of coal use will likely have the biggest impact on future emissions."
    )
