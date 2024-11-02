import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load and preprocess dataset
file_path = "IPL_Data_2015.csv"
data = pd.read_csv(file_path)

# Select features and target
X = data.drop(columns=["Sold Price", "Sl.NO.", "PLAYING ROLE"])
y = data["Sold Price"]

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Support Vector Regressor": SVR(),
}

# Store model metrics for comparison
model_metrics = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mse(y_test, predictions, squared=False)  # Calculate RMSE
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    model_metrics.append({"Model": name, "RMSE": rmse, "MAE": mae, "R² Score": r2})

# Find the best model based on RMSE
best_model_data = min(model_metrics, key=lambda x: x["RMSE"])
best_model_name = best_model_data["Model"]
best_model = models[best_model_name]

# Streamlit App Interface
st.title("IPL Auction Price Predictor")
st.markdown("## Predict the auction price for IPL players based on their performance stats")

# Team images for background
team_images = {
    "Chennai Super Kings": "CSK.png",
    "Delhi Capitals": "DC.png",
    "Gujarat Titans": "GT.png",
    "Mumbai Indians": "MI.png",
    "Rajasthan Royals": "RR.png",
    "Royal Challengers Bangalore": "RCB.png",
    "Sunrisers Hyderabad": "SRH.png",
}

# Create a selection for the carousel index
carousel_index = st.slider("Select Team", 0, len(team_images) - 1, 0)

# Show the selected team logo
team_list = list(team_images.keys())
selected_team = team_list[carousel_index]
st.image(team_images[selected_team], caption=selected_team, use_column_width=True)

# Player role selection
player_role = st.selectbox("Select Player Role", ["Batsman", "Bowler", "All-Rounder"])

# Define input fields based on player role
batsman_fields = [
    "BAT",
    "BAT*SR",
    "BAT*RUN-S",
    "BAT*T-RUNS",
    "BAT*ODI-RUNS",
    "T-RUNS",
    "SIXERS",
    "AVE",
]
bowler_fields = [
    "BOW",
    "BOW*ECO",
    "BOW*SR-BL",
    "BOW*WK-I",
    "BOW*WK-O",
    "T-WKTS",
    "RUNS-C",
    "WKTS",
    "ECON",
]
all_rounder_fields = batsman_fields + bowler_fields

# Initialize empty dictionary for input data
input_data = {feature: 0.0 for feature in X.columns}

# Show relevant input fields based on player role
if player_role == "Batsman":
    st.sidebar.header("Batsman Stats Input")
    for field in batsman_fields:
        input_data[field] = st.sidebar.number_input(f"Enter {field}", min_value=0.0)
elif player_role == "Bowler":
    st.sidebar.header("Bowler Stats Input")
    for field in bowler_fields:
        input_data[field] = st.sidebar.number_input(f"Enter {field}", min_value=0.0)
else:  # All-Rounder
    st.sidebar.header("All-Rounder Stats Input")
    for field in all_rounder_fields:
        input_data[field] = st.sidebar.number_input(f"Enter {field}", min_value=0.0)

# Additional common fields
st.sidebar.header("Additional Player Details")
input_data["ALL*SR-B"] = st.sidebar.number_input("ALL*SR-B", min_value=0.0)
input_data["ALL*SR-BL"] = st.sidebar.number_input("ALL*SR-BL", min_value=0.0)
input_data["ALL*ECON"] = st.sidebar.number_input("ALL*ECON", min_value=0.0)
input_data["Base Price"] = st.sidebar.number_input("Base Price", min_value=0.0)
input_data["Year"] = st.sidebar.number_input("Year", min_value=2015, step=1)

# Prediction button
if st.button("Predict Auction Price"):
    # Prepare the input data
    input_df = pd.DataFrame([input_data])

    # Impute and scale input data
    input_df = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    input_scaled = scaler.transform(input_df)

    # Predict using the best model
    predicted_price = best_model.predict(input_scaled)[0]
    st.success(f"Predicted Auction Price: ₹{predicted_price:.2f}")
    st.write(f"Model used: {best_model_name} with RMSE: {best_model_data['RMSE']:.2f}")

    # Show model comparison metrics
    st.subheader("Model Comparison")
    model_df = pd.DataFrame(model_metrics)
    st.write(model_df)

    # Plot RMSE comparison
    st.subheader("RMSE Comparison Across Models")
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="RMSE", data=model_df, ax=ax)
    ax.set_title("RMSE Comparison Across Models")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Model")
    st.pyplot(fig)

    # Scatter plot of predictions vs actual values for the best model
    st.subheader(f"{best_model_name} Predictions vs Actual Values")
    best_model_predictions = best_model.predict(X_test)
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=best_model_predictions, ax=ax)
    ax.set_title(f"{best_model_name} Predictions vs Actual Values")
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    st.pyplot(fig)

    # Display metrics for the best model
    st.write(f"### Best Model ({best_model_name}) Metrics")
    st.write(f"- RMSE: {best_model_data['RMSE']:.2f}")
    st.write(f"- MAE: {best_model_data['MAE']:.2f}")
    st.write(f"- R² Score: {best_model_data['R² Score']:.2f}")
