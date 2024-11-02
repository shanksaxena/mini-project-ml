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

# Streamlit App Interface with Improved IPL Theme Colors
st.markdown(
    """
    <style>
    /* Page Background */
    .stApp {
        background-color: #1e3a8a; /* Deep blue for main background */
        color: #ffffff; /* White text for readability */
        background-image: url("https://path_to_your_trophy_image.png");
        background-size: cover;
        background-attachment: fixed;
    }
    /* Titles */
    .title {
        color: #ffcc00; /* Gold color for main titles */
        text-align: center;
        font-size: 3em;
        font-weight: bold;
    }
    .subtitle {
        color: #e0e7ff; /* Light blue-white for subtitles */
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
    }
    /* Input Fields */
    .stTextInput, .stNumberInput, .stSelectbox {
        color: #1e3a8a; /* Deep blue for text input */
        background-color: #e0e7ff; /* Light blue-white for background */
    }
    /* Buttons */
    .stButton>button {
        background-color: #ffcc00; /* Gold background */
        color: #1e3a8a; /* Blue text */
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #e0e7ff; /* Light blue-white on hover */
        color: #1e3a8a; /* Deep blue text */
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1e3a8a; /* Deep blue for sidebar */
        color: #ffffff; /* White text for readability */
    }
    /* Dataframes */
    .dataframe {
        color: #1e3a8a; /* Deep blue text */
        background-color: #e0e7ff; /* Light blue-white for background */
        border: 1px solid #ffcc00; /* Gold border */
    }
    /* Model Comparison Table */
    .stTable td, .stTable th {
        color: #1e3a8a; /* Deep blue text */
        background-color: #e0e7ff; /* Light blue-white background */
        border: 1px solid #ffcc00; /* Gold border */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page Title and Subheading
st.markdown("<h1 class='title'>IPL Auction Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Predict IPL Player Prices with Performance Stats</h3>", unsafe_allow_html=True)

# Player role selection
player_role = st.selectbox("Select Player Role", ["Batsman", "Bowler", "All-Rounder"])

# Define input fields based on player role
batsman_fields = [
    "BAT", "BAT*SR", "BAT*RUN-S", "BAT*T-RUNS", "BAT*ODI-RUNS", "T-RUNS", "SIXERS", "AVE",
]
bowler_fields = [
    "BOW", "BOW*ECO", "BOW*SR-BL", "BOW*WK-I", "BOW*WK-O", "T-WKTS", "RUNS-C", "WKTS", "ECON",
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
    