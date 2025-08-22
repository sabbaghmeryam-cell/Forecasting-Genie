import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')
import math
from datetime import datetime, timedelta

# Enhanced page config with modern styling
st.set_page_config(
    page_title="AI Forecasting Assistant Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI inspired by Material Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        text-align: center;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px 0 rgba(31, 38, 135, 0.4);
    }
    
    .algorithm-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 15px;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    .forecast-title {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 class="forecast-title">🤖 AI Forecasting Assistant Pro</h1>
    <p class="subtitle">Advanced machine learning-powered time series forecasting with modern algorithms</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with modern styling
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model selection with descriptions
    st.markdown("#### 🧠 Choose Algorithm")
    method = st.selectbox(
        "Forecasting Method",
        [
            "Random Forest",
            "Linear Regression", 
            "Support Vector Regression",
            "Neural Network (MLP)",
            "Simple Average",
            "Linear Trend",
            "Seasonal Naive"
        ],
        help="Select the machine learning algorithm for forecasting"
    )
    
    st.markdown("#### 📊 Forecast Settings")
    forecast_periods = st.slider("Forecast Periods", min_value=1, max_value=365, value=30)
    
    # Feature engineering options
    st.markdown("#### 🔧 Feature Engineering")
    use_lag_features = st.checkbox("Use Lag Features", value=True)
    lag_periods = st.slider("Number of Lag Periods", min_value=1, max_value=20, value=7)
    
    use_rolling_features = st.checkbox("Use Rolling Statistics", value=True)
    rolling_window = st.slider("Rolling Window Size", min_value=2, max_value=30, value=7)
    
    seasonal_period = st.slider("Seasonal Period", min_value=2, max_value=52, value=7)
    
    # Train-test split
    st.markdown("#### 🎯 Model Validation")
    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20) / 100

def create_features(df, lag_periods=7, rolling_window=7, seasonal_period=7, use_lag=True, use_rolling=True):
    """Create advanced features for ML models"""
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Basic time features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Cyclical features
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    if use_lag:
        # Lag features
        for lag in range(1, lag_periods + 1):
            df[f'Value_lag_{lag}'] = df['Value'].shift(lag)
    
    if use_rolling:
        # Rolling statistics
        df[f'Value_rolling_mean_{rolling_window}'] = df['Value'].rolling(window=rolling_window).mean()
        df[f'Value_rolling_std_{rolling_window}'] = df['Value'].rolling(window=rolling_window).std()
        df[f'Value_rolling_min_{rolling_window}'] = df['Value'].rolling(window=rolling_window).min()
        df[f'Value_rolling_max_{rolling_window}'] = df['Value'].rolling(window=rolling_window).max()
    
    # Trend feature
    df['Trend'] = range(len(df))
    
    return df

def prepare_ml_data(df, target_col='Value'):
    """Prepare data for ML models"""
    feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
    X = df[feature_cols]
    y = df[target_col]
    
    # Remove rows with NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    dates = df['Date'][mask]
    
    return X, y, dates, feature_cols

def train_ml_model(X, y, method):
    """Train ML model based on selected method"""
    if method == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    elif method == "Linear Regression":
        model = LinearRegression()
    elif method == "Support Vector Regression":
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    elif method == "Neural Network (MLP)":
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
    else:
        return None
    
    # Scale features for SVR and Neural Network
    if method in ["Support Vector Regression", "Neural Network (MLP)"]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        return model, scaler
    else:
        model.fit(X, y)
        return model, None

def simple_forecast_methods(data, periods, method, seasonal_period):
    """Simple forecasting methods"""
    values = data['Value'].values
    
    if method == "Simple Average":
        forecast_value = np.mean(values[-30:])
        forecast = [forecast_value] * periods
    elif method == "Linear Trend":
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        future_x = np.arange(len(values), len(values) + periods)
        forecast = np.polyval(coeffs, future_x)
    else:  # Seasonal Naive
        if len(values) >= seasonal_period:
            last_season = values[-seasonal_period:]
            forecast = np.tile(last_season, int(np.ceil(periods / seasonal_period)))[:periods]
        else:
            forecast = [np.mean(values)] * periods
    
    return forecast

def create_future_features(last_row, periods, feature_cols):
    """Create features for future predictions"""
    future_features = []
    
    for i in range(periods):
        future_date = last_row['Date'] + timedelta(days=i+1)
        
        # Create future row with time-based features
        future_row = {}
        future_row['DayOfWeek'] = future_date.weekday()
        future_row['Month'] = future_date.month
        future_row['Quarter'] = (future_date.month - 1) // 3 + 1
        future_row['DayOfYear'] = future_date.timetuple().tm_yday
        
        # Cyclical features
        future_row['DayOfWeek_sin'] = np.sin(2 * np.pi * future_row['DayOfWeek'] / 7)
        future_row['DayOfWeek_cos'] = np.cos(2 * np.pi * future_row['DayOfWeek'] / 7)
        future_row['Month_sin'] = np.sin(2 * np.pi * future_row['Month'] / 12)
        future_row['Month_cos'] = np.cos(2 * np.pi * future_row['Month'] / 12)
        
        # Trend feature
        future_row['Trend'] = last_row['Trend'] + i + 1
        
        # For other features, use last known values or zeros
        for col in feature_cols:
            if col not in future_row:
                future_row[col] = 0
        
        future_features.append([future_row.get(col, 0) for col in feature_cols])
    
    return np.array(future_features)

def plot_enhanced_forecast(historical, forecast, method, train_score=None, test_score=None):
    """Enhanced plot with modern styling"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Set style
    plt.style.use('default')
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    
    # Main forecast plot
    ax1.plot(historical['Date'], historical['Value'], 
             label='Historical Data', color=colors[0], linewidth=2.5, alpha=0.8)
    
    # Create future dates
    last_date = historical['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=len(forecast), freq='D')
    
    ax1.plot(future_dates, forecast, 
             label=f'{method} Forecast', color=colors[1], 
             linewidth=3, linestyle='--', alpha=0.9)
    
    ax1.fill_between(future_dates, forecast, alpha=0.3, color=colors[1])
    
    ax1.set_title(f'{method} - Time Series Forecast', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add performance metrics as text if available
    if train_score is not None and test_score is not None:
        textstr = f'Train R²: {train_score:.3f}\nTest R²: {test_score:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    # Distribution plot
    ax2.hist(historical['Value'], bins=30, alpha=0.7, color=colors[0], 
             label='Historical Distribution', density=True)
    ax2.hist(forecast, bins=20, alpha=0.7, color=colors[1], 
             label='Forecast Distribution', density=True)
    ax2.set_title('Value Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Value', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_sample_data():
    """Create sample time series data with more complexity"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    
    # Multiple components
    trend = np.linspace(100, 200, len(dates))
    seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly
    yearly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Yearly
    noise = np.random.normal(0, 5, len(dates))
    
    # Add some business logic (lower values on weekends)
    weekend_effect = -10 * (pd.Series(dates).dt.dayofweek >= 5).astype(int)
    
    values = trend + seasonal + yearly + noise + weekend_effect
    
    return pd.DataFrame({'Date': dates, 'Value': values})

# Main application logic
uploaded_file = st.file_uploader("📁 Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
        st.success("✅ File uploaded successfully!")
        
        # Data preview with enhanced styling
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Rows", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Columns", len(df.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Missing Values", df.isnull().sum().sum())
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Column selection
        st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
        st.subheader("🎯 Select Columns")
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("📅 Date column", df.columns)
        with col2:
            value_col = st.selectbox("📈 Value column", df.columns)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate forecast
        if st.button("🚀 Generate Advanced Forecast"):
            try:
                with st.spinner('Training model and generating forecast...'):
                    # Prepare data
                    forecast_df = df[[date_col, value_col]].copy()
                    forecast_df = forecast_df.dropna()
                    forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
                    forecast_df = forecast_df.sort_values(date_col).reset_index(drop=True)
                    forecast_df.columns = ['Date', 'Value']
                    
                    if method in ["Random Forest", "Linear Regression", "Support Vector Regression", "Neural Network (MLP)"]:
                        # Create features
                        df_features = create_features(
                            forecast_df, 
                            lag_periods=lag_periods,
                            rolling_window=rolling_window, 
                            seasonal_period=seasonal_period,
                            use_lag=use_lag_features,
                            use_rolling=use_rolling_features
                        )
                        
                        # Prepare ML data
                        X, y, dates, feature_cols = prepare_ml_data(df_features)
                        
                        if len(X) > 10:  # Minimum samples for ML
                            # Train-test split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42, shuffle=False
                            )
                            
                            # Train model
                            result = train_ml_model(X_train, y_train, method)
                            if isinstance(result, tuple):
                                model, scaler = result
                            else:
                                model = result
                                scaler = None
                            
                            # Evaluate model
                            if scaler:
                                train_pred = model.predict(scaler.transform(X_train))
                                test_pred = model.predict(scaler.transform(X_test))
                            else:
                                train_pred = model.predict(X_train)
                                test_pred = model.predict(X_test)
                            
                            train_r2 = r2_score(y_train, train_pred)
                            test_r2 = r2_score(y_test, test_pred)
                            test_mae = mean_absolute_error(y_test, test_pred)
                            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                            
                            # Create future features for prediction
                            last_row = df_features.iloc[-1]
                            future_X = create_future_features(last_row, forecast_periods, feature_cols)
                            
                            # Generate forecast
                            if scaler:
                                forecast = model.predict(scaler.transform(future_X))
                            else:
                                forecast = model.predict(future_X)
                        else:
                            st.error("Not enough data for ML model. Using simple method instead.")
                            forecast = simple_forecast_methods(forecast_df, forecast_periods, "Linear Trend", seasonal_period)
                            train_r2 = test_r2 = None
                            test_mae = test_rmse = None
                    else:
                        # Use simple methods
                        forecast = simple_forecast_methods(forecast_df, forecast_periods, method, seasonal_period)
                        train_r2 = test_r2 = None
                        test_mae = test_rmse = None
                
                # Display results
                st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
                st.subheader("📈 Forecast Results")
                
                # Model performance metrics
                if train_r2 is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Train R²", f"{train_r2:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Test R²", f"{test_r2:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Test MAE", f"{test_mae:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Test RMSE", f"{test_rmse:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Plot forecast
                fig = plot_enhanced_forecast(forecast_df, forecast, method, train_r2, test_r2)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Forecast summary
                st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
                st.subheader("📊 Forecast Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Forecast Periods", len(forecast))
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Avg Forecast", f"{np.mean(forecast):.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Forecast Std", f"{np.std(forecast):.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Create forecast dataframe for download
                last_date = forecast_df['Date'].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                           periods=len(forecast), freq='D')
                forecast_output = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast,
                    'Method': method
                })
                
                st.subheader("📋 Forecast Data")
                st.dataframe(forecast_output, use_container_width=True)
                
                # Download button
                csv = forecast_output.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"forecast_{method.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
                st.info("Please check your data format and try again.")
                
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

else:
    # Sample data section
    st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
    st.info("📁 Please upload a CSV file to get started, or try the sample data below.")
    
    if st.button("🎲 Use Sample Data"):
        sample_df = create_sample_data()
        
        st.subheader("📊 Sample Data Preview")
        st.dataframe(sample_df.head(10), use_container_width=True)
        
        # Sample data visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sample_df['Date'], sample_df['Value'], linewidth=2, color='#667eea')
        ax.set_title('Sample Time Series Data', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Generate sample forecast
        if st.button("🚀 Generate Sample Forecast"):
            try:
                with st.spinner('Generating sample forecast...'):
                    if method in ["Random Forest", "Linear Regression", "Support Vector Regression", "Neural Network (MLP)"]:
                        # ML approach for sample data
                        df_features = create_features(
                            sample_df, 
                            lag_periods=lag_periods,
                            rolling_window=rolling_window,
                            seasonal_period=seasonal_period,
                            use_lag=use_lag_features,
                            use_rolling=use_rolling_features
                        )
                        
                        X, y, dates, feature_cols = prepare_ml_data(df_features)
                        
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, shuffle=False
                        )
                        
                        # Train model
                        result = train_ml_model(X_train, y_train, method)
                        if isinstance(result, tuple):
                            model, scaler = result
                        else:
                            model = result
                            scaler = None
                        
                        # Evaluate
                        if scaler:
                            train_pred = model.predict(scaler.transform(X_train))
                            test_pred = model.predict(scaler.transform(X_test))
                        else:
                            train_pred = model.predict(X_train)
                            test_pred = model.predict(X_test)
                        
                        train_r2 = r2_score(y_train, train_pred)
                        test_r2 = r2_score(y_test, test_pred)
                        
                        # Generate forecast
                        last_row = df_features.iloc[-1]
                        future_X = create_future_features(last_row, forecast_periods, feature_cols)
                        
                        if scaler:
                            forecast = model.predict(scaler.transform(future_X))
                        else:
                            forecast = model.predict(future_X)
                    else:
                        forecast = simple_forecast_methods(sample_df, forecast_periods, method, seasonal_period)
                        train_r2 = test_r2 = None
                
                st.subheader("📈 Sample Forecast Results")
                fig = plot_enhanced_forecast(sample_df, forecast, method, train_r2, test_r2)
                st.pyplot(fig)
                
                # Sample metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Forecast Periods", len(forecast))
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Avg Forecast", f"{np.mean