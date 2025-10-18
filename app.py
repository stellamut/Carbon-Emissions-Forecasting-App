# ====================================================
# SDG 13: Climate Action – Forecast Carbon Emissions
# Theme: Machine Learning Meets the UN SDGs
# ====================================================

# 📦 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

# ----------------------------------------------------
# 🧩 1. Load Dataset
# ----------------------------------------------------
# Example dataset: CO2 emissions (kt) from World Bank (can replace with UN SDG data)
url = "https://raw.githubusercontent.com/datasets/co2-fossil-global/master/global.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
df.head()

# ----------------------------------------------------
# 🧹 2. Data Preprocessing
# ----------------------------------------------------
# Rename columns for convenience
df.rename(columns={'Year': 'year', 'Total': 'co2_emissions'}, inplace=True)

# Drop missing or invalid rows
df = df.dropna(subset=['year', 'co2_emissions'])

# Keep only numeric features for regression
df = df[df['co2_emissions'] > 0]

# Create lag features (previous year’s emissions)
df['lag1'] = df['co2_emissions'].shift(1)
df['lag2'] = df['co2_emissions'].shift(2)
df = df.dropna()

# Define features (X) and target (y)
X = df[['lag1', 'lag2']]
y = df['co2_emissions']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# 🤖 3. Train Regression Models
# ----------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

# Display results
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Metrics:\n")
print(results_df)

# ----------------------------------------------------
# 📈 4. Visualization
# ----------------------------------------------------
best_model = models["Random Forest"]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_best)
plt.xlabel("Actual CO2 Emissions (kt)")
plt.ylabel("Predicted CO2 Emissions (kt)")
plt.title("Predicted vs Actual CO2 Emissions")
plt.grid(True)
plt.show()

# ----------------------------------------------------
# 🔍 5. (Optional) Unsupervised Learning Extension
# Cluster Years by Emission Levels
# ----------------------------------------------------
# Cluster by emission magnitudes
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['co2_emissions']])

plt.figure(figsize=(8,5))
sns.scatterplot(x='year', y='co2_emissions', hue='cluster', data=df, palette='viridis')
plt.title("Clustering of Years by Emission Levels")
plt.xlabel("Year")
plt.ylabel("Global CO2 Emissions (kt)")
plt.show()

# ----------------------------------------------------
# 🌱 6. Ethical Reflection (Output Summary)
# ----------------------------------------------------
print("\n--- Ethical Reflection ---")
print("Potential Bias: Dataset focuses on global emissions; missing regional granularity may bias policy insights.")
print("Fairness: Ensure representation of all regions and socio-economic contexts in data collection.")
print("Sustainability: Accurate forecasting aids in setting emission targets and monitoring progress toward SDG 13.")
