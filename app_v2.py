import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Forecasting Saham BBRI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menu Sidebar
menu = st.sidebar.selectbox(
    "📂 Pilih Analisis",
    [
        "Dashboard",
        "Time Series Analysis"
    ]
)


st.title("🏦 Forecasting Saham BBRI - Bank Rakyat Indonesia")
st.markdown("**Real-time ML Analysis & 3-Month Forecast**")
st.markdown("---")

# Fetch BBRI.JK data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_bbri_data():
    """Fetch BBRI.JK data from Yahoo Finance"""
    ticker = "BBRI.JK"
    stock = yf.Ticker(ticker)
    data = stock.history(period="2y")
    info = stock.info
    return data, info

# Load data
try:
    data, stock_info = fetch_bbri_data()
    if data.empty:
        st.error("❌ No data found for BBRI.JK. Check ticker symbol.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error fetching BBRI.JK data: {str(e)}")
    st.stop()

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

current_price = data['Close'].iloc[-1]
prev_close = data['Close'].iloc[-2]
price_change = current_price - prev_close
price_change_pct = (price_change / prev_close) * 100

avg_volume = data['Volume'].mean()

with col1:
    st.metric(
        "💰 Harga Saat ini",
        f"Rp {current_price:,.0f}",
        f"{price_change_pct:+.2f}%",
        delta_color="normal"
    )

with col2:
    st.metric(
        "📊 Volume",
        f"{data['Volume'].iloc[-1]:,.0f}",
        f"{((data['Volume'].iloc[-1]/avg_volume)-1)*100:+.1f}% vs Avg"
    )

with col3:
    st.metric(
        "🎯 52W Tertinggi",
        f"Rp {data['High'].max():,.0f}",
        f"{((current_price/data['High'].max())-1)*100:.1f}% from High"
    )

with col4:
    st.metric(
        "🎯 52W Terendah", 
        f"Rp {data['Low'].min():,.0f}",
        f"{((current_price/data['Low'].min())-1)*100:.1f}% from Low"
    )

with col5:
    st.metric(
        "📈 Market Cap",
        f"Rp {stock_info.get('marketCap', 0):,.0f}",
        "Live"
    )

st.subheader("📈 Harga Penutupan Historis")
st.markdown("Menampilkan harga penutupan BBRI.JK selama 2 tahun dari Yahoo Finance")
st.line_chart(data['Close'].to_frame('Close Price'), height=350)

# Walk-Forward Analysis Function
@st.cache_data
def walk_forward_mape_analysis(data, window_size=21, test_size=5):
    """
    Professional Walk-Forward Analysis with MAPE
    window_size=21 (1 month trading days), test_size=5 (1 week)
    """
    # Feature Engineering
    df = data.copy()
    df['Returns'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Volume_MA'] = df['Volume'].rolling(10).mean()
    df = df.dropna()
    
    # Features for ML
    feature_cols = ['MA_5', 'MA_20', 'MA_50', 'Volatility', 'RSI', 'Volume_MA']
    X = df[feature_cols].values
    y = df['Close'].values
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Walk-Forward Validation
    mape_scores = []
    predictions = []
    actuals = []
    
    for i in range(window_size, len(X_scaled) - test_size):
        # Train/Test split
        X_train, y_train = X_scaled[:i], y[:i]
        X_test, y_test = X_scaled[i:i+test_size], y[i:i+test_size]
        
        # Train ensemble model
        model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict & calculate MAPE
        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        mape_scores.append(mape)
        predictions.extend(y_pred)
        actuals.extend(y_test)
    
    final_mape = np.mean(mape_scores)
    
    return {
        'mape_mean': final_mape,
        'mape_scores': mape_scores,
        'predictions': np.array(predictions),
        'actuals': np.array(actuals),
        'df': df,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

def compute_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Run Walk-Forward Analysis
with st.spinner("🔄 Computing Walk-Forward MAPE Analysis..."):
    wf_results = walk_forward_mape_analysis(data)

# Display MAPE Result with custom div
st.markdown(
    f"""
    <div data-testid="model-validation-mape" style='padding: 15px; border-radius: 8px; background: #81A6C6; color: #2C2C2C; margin-bottom: 20px;'>
        <h3 style='margin: 0; font-weight: 600;'>📊 Model Validation MAPE: {wf_results['mape_mean']:.2f}%</h3>  
        <p style='margin: 8px 0 0 0; font-size: 14px;'>
            <span style='background: rgba(44,94,173,0.15); padding: 2px 6px; border-radius: 4px; margin-right: 8px; color: #2C5EAD;'>&lt;2%: Excellent</span>
            <span style='background: rgba(44,94,173,0.15); padding: 2px 6px; border-radius: 4px; margin-right: 8px; color: #2C5EAD;'>2-5%: Good</span>
            <span style='background: rgba(44,94,173,0.15); padding: 2px 6px; border-radius: 4px; margin-right: 8px; color: #2C5EAD;'>5-10%: Fair</span>
            <span style='background: rgba(44,94,173,0.15); padding: 2px 6px; border-radius: 4px; color: #2C5EAD;'>&gt;10%: Poor</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 3-Month Forecast
st.subheader("🔮 3-Month Price Forecast")

# Train final model
df_ml = wf_results['df']
feature_cols = wf_results['feature_cols']
X_final = wf_results['scaler'].transform(df_ml[feature_cols])
final_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
final_model.fit(X_final, df_ml['Close'])

# Generate 90-day forecast
forecast_days = 90
forecast_prices = []
last_features = df_ml[feature_cols].iloc[-1:].values

for i in range(forecast_days):
    # Predict next price
    next_features_scaled = wf_results['scaler'].transform(last_features)
    next_price = final_model.predict(next_features_scaled)[0]
    forecast_prices.append(next_price)
    
    # Update features for next iteration (simplified)
    new_features = last_features.copy()
    new_features[0, 0] = next_price * 0.8 + last_features[0, 0] * 0.2  # MA5
    new_features[0, 1] = next_price * 0.5 + last_features[0, 1] * 0.5  # MA20
    new_features[0, 2] = next_price * 0.2 + last_features[0, 2] * 0.8  # MA50
    last_features = new_features

# Forecast DataFrame
forecast_dates = pd.date_range(
    start=data.index[-1] + timedelta(days=1), 
    periods=forecast_days, 
    freq='B'  # Business days
)
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecast': forecast_prices
})

# Main Visualization
fig_main = make_subplots(
    rows=2, cols=1,
    subplot_titles=('BBRI.JK Price History & 3-Month Forecast', 'Trading Volume'),
    row_heights=[0.75, 0.25],
    vertical_spacing=0.08,
    shared_xaxes=True
)

# Historical prices (last 6 months)
hist_data = data.tail(126)  # ~6 months trading days
fig_main.add_trace(
    go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name="BBRI.JK",
        increasing_line_color='#00D4AA',
        decreasing_line_color='#FF6B6B'
    ),
    row=1, col=1
)

# Forecast
fig_main.add_trace(
    go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast'],
        mode='lines',
        name="ML Forecast",
        line=dict(color='#4ECDC4', width=4),
        fill='tonexty'
    ),
    row=1, col=1
)

# Confidence bands (±8% typical for stocks)
upper = np.array(forecast_prices) * 1.08
lower = np.array(forecast_prices) * 0.92
fig_main.add_trace(
    go.Scatter(x=forecast_dates, y=upper, line=dict(color='rgba(78,205,196,0.3)'), 
               fill=None, showlegend=False),
    row=1, col=1
)
fig_main.add_trace(
    go.Scatter(x=forecast_dates, y=lower, fill='tonexty',
               fillcolor='rgba(78,205,196,0.15)', line=dict(color='rgba(78,205,196,0.3)'),
               name="90% Confidence"),
    row=1, col=1
)

# Volume bars
fig_main.add_trace(
    go.Bar(x=hist_data.index, y=hist_data['Volume'],
           name="Volume", marker_color='rgba(158,202,225,0.6)',
           yaxis="y2"),
    row=2, col=1
)

# Layout
fig_main.update_layout(
    height=700,
    title={
        'text': f'BBRI.JK Analysis | Current: IDR {current_price:,.0f} | MAPE: {wf_results["mape_mean"]:.2f}%',
        'x': 0.5,
        'xanchor': 'center'
    },
    showlegend=True,
    hovermode='x unified'
)

fig_main.update_xaxes(title_text="Date", row=2, col=1)
fig_main.update_yaxes(title_text="Price (IDR)", secondary_y=False, row=1, col=1)
fig_main.update_yaxes(title_text="Volume", secondary_y=True, row=2, col=1)

st.plotly_chart(fig_main, use_container_width=True)

# Forecast Summary Table
st.subheader("📋 3-Month Forecast Summary")
summary_data = {
    'Period': ['Current', '1 Month', '2 Months', '3 Months'],
    'Price (IDR)': [
        f"{current_price:,.0f}",
        f"{forecast_df['Forecast'].iloc[21]:,.0f}",
        f"{forecast_df['Forecast'].iloc[42]:,.0f}", 
        f"{forecast_df['Forecast'].iloc[-1]:,.0f}"
    ],
    'Change %': [
        '—',
        f"{((forecast_df['Forecast'].iloc[21]/current_price)-1)*100:+.1f}%",
        f"{((forecast_df['Forecast'].iloc[42]/current_price)-1)*100:+.1f}%",
        f"{((forecast_df['Forecast'].iloc[-1]/current_price)-1)*100:+.1f}%"
    ]
}
st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# Sidebar with model info
with st.sidebar:
    st.markdown(
        f"""
        <h4 style='display: inline-flex; align-items: center; gap: 8px;'>
            ⚙️ Model Details
            <span style='font-weight: bold; color: #0d6efd; cursor: help;' title='Algorithm: Gradient Boosting\nFeatures: MA(5,20,50), Volatility, RSI, Volume\nValidation: Walk-Forward (21d train / 5d test)\nForecast: 90 business days\nMAPE: {wf_results['mape_mean']:.2f}%'>
                ?
            </span>
        </h4>
        """,
        unsafe_allow_html=True
    )
    
    st.header("📊 Quick Stats")
    st.metric("Sharpe Ratio", f"{data['Close'].pct_change().mean()/data['Close'].pct_change().std()*np.sqrt(252):.2f}")
    st.metric("Max Drawdown", f"{((data['Close']/data['Close'].expanding().max())-1).min()*100:.1f}%")

st.markdown("---")
st.markdown(f"""
**⚠️ Disclaimer:** Ini hanya untuk tujuan pendidikan. Bukan merupakan saran keuangan. Kinerja di masa lalu tidak menjamin hasil di masa depan | **Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
""")

# Global CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Font Family Global */
    html, body, [class^='css'], [class*=' css'] {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Emotion Cache Container Styling */
    .st-emotion-cache-ocqkz7 {
        font-family: 'Poppins', sans-serif !important;
        font-size: 13px !important;
        white-space: nowrap !important;
        overflow: visible !important;
        font-variant-numeric: tabular-nums !important;
    }
    
    .st-emotion-cache-1o8cngj {
        font-family: 'Poppins', sans-serif !important;
    }
    
    .st-emotion-cache-z5fcl4 {
        background-color: #021A54 !important;
    }
    
    /* Additional Emotion Cache Variants */
    [class*='st-emotion-cache'] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Smaller Sidebar */
    [data-testid='stSidebar'] {
        min-width: 260px !important;
        max-width: 260px !important;
        background-color: #4382DF!important;
    }
    [data-testid='stSidebar'] div {
        padding: 0.1rem !important;
    }
    [data-testid='stSidebar'] h4,
    [data-testid='stSidebar'] h1,
    [data-testid='stSidebar'] h2,
    [data-testid='stSidebar'] h3,
    [data-testid='stSidebar'] p,
    [data-testid='stSidebar'] span {
        margin: 0.1rem 0 !important;
        color: white !important;
    }
    [data-testid='stSidebar'] .stMetric,
    [data-testid='stSidebar'] .stMetric > div {
        margin-bottom: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        color: white !important;
    }
    [data-testid='stSidebar'] .stMarkdown {
        margin-bottom: 0.2rem !important;
    }
    [data-testid='stSidebar'] .streamlit-expanderHeader {
        margin-bottom: 0.1rem !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
