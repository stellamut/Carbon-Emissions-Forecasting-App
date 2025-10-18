# ---------------------------------------------
# STEP 1: Data Acquisition & Wrangling (using pandas)
# ---------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

# Target and Feature Indicator Codes (based on World Bank Data)
INDICATORS = {
    'EN.ATM.CO2E.KT': 'CO2_Emissions',  # TARGET (Y)
    'NY.GDP.MKTP.CD': 'GDP',
    'EG.USE.ELEC.KH.PC': 'Elec_Cons_PC',
    'SP.POP.TOTL': 'Population',
    'EG.FEC.RNEW.ZS': 'Renewable_Share',
    'EG.USE.COMM.CL.ZS': 'Coal_Share'
}

def load_and_clean_data(file_path):
    # *Simplified placeholder for reading World Bank Data*
    # (In reality, you would load the downloaded WDI files or use wbgapi)
    df = pd.read_csv(file_path) 
    df.rename(columns={'Year': 'Time', 'Country Name': 'Country'}, inplace=True)
    df.dropna(subset=list(INDICATORS.values()), inplace=True)
    return df

# ---------------------------------------------
# STEP 2: Feature Engineering (Creating Time-Series Features)
# ---------------------------------------------
def feature_engineer(df, target_col='CO2_Emissions', lag=1):
    # Create the 'lag' of the Target variable (e.g., CO2 emission from previous year)
    df[f'{target_col}_Lag{lag}'] = df.groupby('Country')[target_col].shift(lag)
    df.dropna(inplace=True)
    return df

# ---------------------------------------------
# STEP 3: Model Training (Supervised Regression)
# ---------------------------------------------
def train_forecaster(df):
    features = [col for col in df.columns if col not in ['Country', 'Time', 'CO2_Emissions']]
    target = 'CO2_Emissions'
    
    X = df[features]
    y = df[target]

    # Split for Supervised Learning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling numerical features is essential for stability, although less critical for Tree Models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and Train XGBoost Regressor
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Evaluation (Optional: Calculate RMSE/MAE on test set here)
    # from sklearn.metrics import mean_squared_error
    # rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled)))
    
    # Save Model and Scaler for Streamlit Deployment
    joblib.dump(model, 'xgb_co2_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Model Training Complete. Model and Scaler saved.")
    return model, scaler, features

# Example call (assuming you have a 'wdi_data.csv' file)
# df_clean = load_and_clean_data('wdi_data.csv')
# df_engineered = feature_engineer(df_clean)
# model, scaler, features = train_forecaster(df_engineered)