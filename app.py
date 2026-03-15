import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set up the mobile-friendly page layout
st.set_page_config(page_title="Sentiment Tracker", page_icon="📈", layout="centered")

st.title("Smart vs. Dumb Money")
st.write("Track institutional accumulation vs. retail trend-chasing.")

# Master list of all your assets
assets = {
    'S&P 500': 'SPY', 'Nasdaq 100': 'QQQ', 'Dow Jones': 'DIA', 'Russell 2000': 'IWM',
    'Technology': 'XLK', 'Financials': 'XLF', 'Communications': 'XLC', 'Energy': 'XLE',
    'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP', 'Real Estate': 'XLRE',
    'Industrials': 'XLI', 'Healthcare': 'XLV', 'Utilities': 'XLU', 'Biotech': 'XBI', 'Materials': 'XLB'
}

# 1. Create the Dropdown Menu
selected_name = st.selectbox("Select an Index or Sector:", list(assets.keys()))
ticker = assets[selected_name]

# 2. Fetch and Calculate Data
@st.cache_data(ttl=3600) # Caches data for 1 hour so the app loads instantly on your phone
def get_data(t):
    data = yf.Ticker(t).history(period="2y")
    if data.empty:
        return None, None, None, None
        
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -1 * delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    clv = clv.fillna(0)
    smart_raw = ((clv.rolling(14).mean() + 1) / 2) * 100

    dumb_smoothed = rsi.rolling(window=21).mean()
    smart_smoothed = smart_raw.rolling(window=21).mean()
    
    return data, smart_smoothed, dumb_smoothed

with st.spinner(f"Loading data for {selected_name}..."):
    data, smart_smoothed, dumb_smoothed = get_data(ticker)

# 3. Build the Chart
if data is not None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=data.index, y=smart_smoothed, name='Smart Money',
        line=dict(color='#2ca02c', width=3)
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data.index, y=dumb_smoothed, name='Dumb Money',
        line=dict(color='#d62728', width=3)
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'], name='Price',
        line=dict(color='#1f77b4', width=2), opacity=0.3,
        fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.1)'
    ), secondary_y=True)

    fig.update_layout(
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0) # Tighter margins for mobile screens
    )

    fig.update_yaxes(title_text="Confidence", range=[0, 100], secondary_y=False)
    fig.add_hline(y=50, line_dash="dash", line_color="black", secondary_y=False)
    fig.update_yaxes(showgrid=False, secondary_y=True)

    # Display it in the app
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Failed to load data. Yahoo Finance might be down.")
