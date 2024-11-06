import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    try:
        df = pd.read_csv('IPL_Data_2015.csv')
        print("Available columns:", df.columns.tolist())
        
        df = df.fillna(0)
        df['experience'] = 2015 - df['Year']
        
        # Using the correct column names from IPL dataset
        df['batting_impact'] = df['BAT*SR'] * df['AVE']
        df['runs_per_match'] = df['BAT*RUN-S'] / np.where(df['SIXERS'] > 0, df['SIXERS'], 1)
        df['bowling_impact'] = df['WKTS'] / np.where(df['ECON'] > 0, df['ECON'], 1)
        df['wicket_rate'] = df['WKTS'] / np.where(df['RUNS-C'] > 0, df['RUNS-C'], 1)
        
        # Determine player type
        df['player_type'] = 'Unknown'
        mask_batsman = (df['BAT*RUN-S'] > 0) & (df['WKTS'] == 0)
        mask_bowler = (df['WKTS'] > 0) & (df['BAT*RUN-S'] == 0)
        mask_allrounder = (df['BAT*RUN-S'] > 0) & (df['WKTS'] > 0)
        
        df.loc[mask_batsman, 'player_type'] = 'Batsman'
        df.loc[mask_bowler, 'player_type'] = 'Bowler'
        df.loc[mask_allrounder, 'player_type'] = 'Allrounder'
        
        df['log_price'] = np.log1p(df['Base Price'])
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        print(f"Error details: {str(e)}")
        return None

def train_player_specific_models(df):
    models = {}
    metrics = {}
    scalers = {}
    
    for player_type in ['Batsman', 'Bowler', 'Allrounder']:
        player_data = df[df['player_type'] == player_type]
        
        if player_type == 'Batsman':
            features = ['BAT*SR', 'BAT*RUN-S', 'AVE', 'SIXERS', 'batting_impact', 
                       'runs_per_match', 'experience']
        elif player_type == 'Bowler':
            features = ['BOW*ECO', 'BOW*SR-BL', 'WKTS', 'ECON', 'bowling_impact', 
                       'wicket_rate', 'experience']
        else:  # Allrounder
            features = ['BAT*SR', 'BAT*RUN-S', 'BOW*ECO', 'WKTS', 'batting_impact',
                       'bowling_impact', 'experience']
        
        if len(player_data) > 0:
            X = player_data[features]
            y = player_data['log_price']
            
            if len(player_data) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                scalers[player_type] = scaler
                
                # Initialize all three models
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                dt_model = DecisionTreeRegressor(random_state=42)
                lr_model = LinearRegression()
                
                # Train all models
                rf_model.fit(X_train_scaled, y_train)
                dt_model.fit(X_train_scaled, y_train)
                lr_model.fit(X_train_scaled, y_train)
                
                # Make predictions
                rf_pred = rf_model.predict(X_test_scaled)
                dt_pred = dt_model.predict(X_test_scaled)
                lr_pred = lr_model.predict(X_test_scaled)
                
                # Calculate metrics for each model
                models[player_type] = {
                    'Random Forest': {
                        'model': rf_model,
                        'features': features,
                        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                        'R2': r2_score(y_test, rf_pred)
                    },
                    'Decision Tree': {
                        'model': dt_model,
                        'features': features,
                        'RMSE': np.sqrt(mean_squared_error(y_test, dt_pred)),
                        'R2': r2_score(y_test, dt_pred)
                    },
                    'Linear Regression': {
                        'model': lr_model,
                        'features': features,
                        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
                        'R2': r2_score(y_test, lr_pred)
                    }
                }
                
                # Find best model based on R2 score
                best_model = max(models[player_type].items(), 
                               key=lambda x: x[1]['R2'])
                
                metrics[player_type] = {
                    'best_model': best_model[0],
                    'metrics': {
                        model_name: {
                            'RMSE': model_info['RMSE'],
                            'R2': model_info['R2']
                        }
                        for model_name, model_info in models[player_type].items()
                    }
                }
    
    return models, metrics, scalers

# Streamlit App
def main():
    st.set_page_config(page_title="IPL Player Price Predictor", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .prediction-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and Description
    st.title("🏏 IPL Player Price Predictor")
    st.markdown("---")

    # Load and train models
    @st.cache_data
    def load_and_train():
        df = load_and_preprocess_data()
        if df is not None:
            models, metrics, scalers = train_player_specific_models(df)
            return df, models, metrics, scalers
        return None, None, None, None

    df, models, metrics, scalers = load_and_train()

    if df is not None and models:
        # Display dataset info in an expander
        with st.expander("Dataset Information"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Players", len(df))
            with col2:
                st.metric("Batsmen", len(df[df['player_type'] == 'Batsman']))
            with col3:
                st.metric("Bowlers", len(df[df['player_type'] == 'Bowler']))
            with col4:
                st.metric("All-rounders", len(df[df['player_type'] == 'Allrounder']))
        
        # Main content
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Player Information")
            player_type = st.selectbox("Select Player Type", ['Batsman', 'Bowler', 'Allrounder'])

            # Input fields based on player type
            if player_type == 'Batsman':
                bat_sr = st.number_input("Batting Strike Rate", min_value=0.0, max_value=200.0, value=120.0)
                runs = st.number_input("Runs Scored", min_value=0, value=500)
                average = st.number_input("Batting Average", min_value=0.0, value=30.0)
                sixers = st.number_input("Number of Sixes", min_value=0, value=10)
                experience = st.slider("Years of Experience", 0, 15, 5)
                
                batting_impact = bat_sr * average
                runs_per_match = runs / max(sixers, 1)
                
                input_data = pd.DataFrame([[bat_sr, runs, average, sixers, batting_impact, 
                                          runs_per_match, experience]], 
                                        columns=models[player_type]['Random Forest']['features'])

            elif player_type == 'Bowler':
                economy = st.number_input("Economy Rate", min_value=0.0, max_value=15.0, value=7.5)
                bowling_sr = st.number_input("Bowling Strike Rate", min_value=0.0, value=20.0)
                wickets = st.number_input("Wickets Taken", min_value=0, value=15)
                experience = st.slider("Years of Experience", 0, 15, 5)
                
                bowling_impact = wickets / max(economy, 1)
                wicket_rate = wickets / 100
                
                input_data = pd.DataFrame([[economy, bowling_sr, wickets, economy, bowling_impact, 
                                          wicket_rate, experience]], 
                                        columns=models[player_type]['Random Forest']['features'])

            else:  # Allrounder
                bat_sr = st.number_input("Batting Strike Rate", min_value=0.0, max_value=200.0, value=120.0)
                runs = st.number_input("Runs Scored", min_value=0, value=300)
                economy = st.number_input("Economy Rate", min_value=0.0, max_value=15.0, value=8.0)
                wickets = st.number_input("Wickets Taken", min_value=0, value=10)
                experience = st.slider("Years of Experience", 0, 15, 5)
                
                batting_impact = bat_sr * (runs/20)
                bowling_impact = wickets / max(economy, 1)
                
                input_data = pd.DataFrame([[bat_sr, runs, economy, wickets, batting_impact,
                                          bowling_impact, experience]], 
                                        columns=models[player_type]['Random Forest']['features'])

        with col2:
            st.subheader("Model Predictions and Analysis")
            
            if st.button("Predict Price", key="predict_button"):
                if player_type in models and player_type in scalers:
                    scaled_input = scalers[player_type].transform(input_data)
                    
                    # Make predictions with all models
                    predictions = {}
                    for model_name, model_info in models[player_type].items():
                        pred = model_info['model'].predict(scaled_input)[0]
                        predictions[model_name] = np.expm1(pred)
                    
                    # Display predictions
                    st.markdown("### Price Predictions")
                    for model_name, price in predictions.items():
                        st.info(f"{model_name}: ₹{price:,.2f}")
                    
                    # Best model highlight
                    best_model_name = metrics[player_type]['best_model']
                    st.success(f"Best performing model: {best_model_name}")
                    
                    # Model metrics comparison
                    st.markdown("### Model Performance Comparison")
                    metrics_df = pd.DataFrame({
                        'Model': list(metrics[player_type]['metrics'].keys()),
                        'RMSE': [m['RMSE'] for m in metrics[player_type]['metrics'].values()],
                        'R² Score': [m['R2'] for m in metrics[player_type]['metrics'].values()]
                    })
                    
                    # Bar chart for model comparison
                    fig1 = px.bar(metrics_df, x='Model', y=['RMSE', 'R² Score'],
                                title='Model Performance Metrics',
                                barmode='group')
                    st.plotly_chart(fig1)
                    
                    # Feature importance
                    st.markdown("### Feature Importance Analysis")
                    rf_importance = pd.DataFrame({
                        'Feature': models[player_type]['Random Forest']['features'],
                        'Importance': models[player_type]['Random Forest']['model'].feature_importances_
                    })
                    rf_importance = rf_importance.sort_values('Importance', ascending=True)
                    
                    fig2 = px.bar(rf_importance, x='Importance', y='Feature',
                                orientation='h',
                                title='Feature Importance (Random Forest)')
                    st.plotly_chart(fig2)
                    
                    # Scatter plot with best fit line
                    st.markdown("### Model Performance Visualization")
                    
                    # Get actual and predicted values
                    X = df[df['player_type'] == player_type][models[player_type]['Random Forest']['features']]
                    y_actual = df[df['player_type'] == player_type]['Base Price']
                    
                    # Get predictions from the best model
                    best_model = models[player_type][best_model_name]['model']
                    X_scaled = scalers[player_type].transform(X)
                    y_pred = np.expm1(best_model.predict(X_scaled))
                    
                    # Create DataFrame for plotting
                    scatter_df = pd.DataFrame({
                        'Actual Price': y_actual,
                        'Predicted Price': y_pred
                    })
                    
                    # Calculate the line of best fit
                    line_model = LinearRegression()
                    X_line = scatter_df['Actual Price'].values.reshape(-1, 1)
                    y_line = scatter_df['Predicted Price'].values.reshape(-1, 1)
                    line_model.fit(X_line, y_line)
                    
                    # Create scatter plot with best fit line
                    fig3 = px.scatter(scatter_df, x='Actual Price', y='Predicted Price',
                                    title=f'Actual vs Predicted Prices ({best_model_name})')
                    
                    # Add best fit line
                    x_range = np.linspace(scatter_df['Actual Price'].min(), 
                                        scatter_df['Actual Price'].max(), 100)
                    y_range = line_model.predict(x_range.reshape(-1, 1))
                    
                    fig3.add_traces(
                        px.line(x=x_range, y=y_range.flatten())
                        .data[0]
                        .update(name='Best Fit Line', line_color='red')
                    )
                    
                    # Add perfect prediction line
                    fig3.add_trace(
                        px.line(x=x_range, y=x_range)
                        .data[0]
                        .update(name='Perfect Predictions', 
                               line=dict(dash='dash'), 
                               line_color='green')
                    )
                    
                    # Update layout
                    fig3.update_layout(
                        xaxis_title="Actual Price (₹)",
                        yaxis_title="Predicted Price (₹)",
                        showlegend=True,
                        legend=dict(
                            yanchor="bottom",
                            y=0.01,
                            xanchor="right",
                            x=0.99
                        )
                    )
                    
                    # Show the plot
                    st.plotly_chart(fig3)
                    
                    # Add interpretation
                    st.markdown("""
                    #### Scatter Plot Interpretation:
                    - **Red Line**: Line of best fit showing the trend of predictions
                    - **Green Dashed Line**: Perfect prediction line (where actual = predicted)
                    - **Points**: Each point represents a player
                    - Closer the points to the green dashed line, better the predictions
                    """)
                    
    else:
        st.error("Error loading data or training models. Please check if 'IPL_Data_2015.csv' exists in the same directory.")

if __name__ == "__main__":
    main()
