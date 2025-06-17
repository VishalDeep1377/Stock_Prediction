import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import sys
import os
import numpy as np
from textblob import TextBlob
import requests
from io import BytesIO
import base64

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessor import StockDataPreprocessor
from src.visualization.visualizer import StockVisualizer

# Set page config
st.set_page_config(
    page_title="Stock Market Analysis",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Market Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive stock market analysis including:
- Historical price data and technical indicators
- Portfolio analysis and optimization
- News sentiment analysis
- Price predictions
- Advanced technical analysis
- Data export capabilities
""")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Stock symbol input with multi-select for portfolio
symbols = st.sidebar.text_input("Stock Symbols (comma-separated)", "AAPL").upper().split(',')
symbols = [s.strip() for s in symbols]

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        datetime.now() - timedelta(days=365)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        datetime.now()
    )

# Analysis type selection
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Technical Analysis", "Portfolio Analysis", "News Analysis", "Prediction Analysis"]
)

# Download and process data
@st.cache_data
def load_data(symbols, start_date, end_date):
    """Load and process stock data."""
    try:
        data_dict = {}
        for symbol in symbols:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                st.error(f"No data found for {symbol}")
                continue
                
            # Process data
            preprocessor = StockDataPreprocessor(df)
            df_processed = preprocessor.handle_missing_values()
            df_processed = preprocessor.add_technical_indicators()
            df_processed = preprocessor.add_time_features()
            df_processed = preprocessor.add_price_features()
            
            data_dict[symbol] = df_processed
            
        return data_dict
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def get_news_sentiment(symbol):
    """Get news sentiment for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return None
            
        # Process news sentiment
        sentiments = []
        for article in news[:10]:  # Analyze last 10 news articles
            if 'title' in article:
                blob = TextBlob(article['title'])
                sentiments.append(blob.sentiment.polarity)
                
        return {
            'mean_sentiment': np.mean(sentiments) if sentiments else 0,
            'news': news[:10]
        }
    except Exception as e:
        st.error(f"Error getting news: {str(e)}")
        return None

def calculate_portfolio_metrics(data_dict):
    """Calculate portfolio metrics."""
    portfolio_returns = pd.DataFrame()
    for symbol, df in data_dict.items():
        portfolio_returns[symbol] = df['Returns']
    
    # Calculate portfolio metrics
    mean_returns = portfolio_returns.mean()
    cov_matrix = portfolio_returns.cov()
    
    # Generate random portfolios
    num_portfolios = 1000
    results = np.zeros((num_portfolios, len(symbols) + 2))
    
    for i in range(num_portfolios):
        weights = np.random.random(len(symbols))
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[i, 0] = portfolio_std
        results[i, 1] = portfolio_return
        results[i, 2:] = weights
    
    return results, mean_returns, cov_matrix

def get_download_link(df, filename):
    """Generate a download link for a dataframe."""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Main content
if st.sidebar.button("Analyze"):
    with st.spinner("Loading and processing data..."):
        data_dict = load_data(symbols, start_date, end_date)
        
        if data_dict:
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "Technical Analysis", 
                "Portfolio Analysis", 
                "News Analysis",
                "Prediction Analysis"
            ])
            
            with tab1:
                st.header("Technical Analysis")
                
                # Select symbol for technical analysis
                selected_symbol = st.selectbox("Select Symbol", symbols)
                df = data_dict[selected_symbol]
                
                # Technical indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price and Moving Averages
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                      vertical_spacing=0.1,
                                      subplot_titles=('Price & Moving Averages', 'RSI'))
                    
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50'), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.update_layout(height=800, title_text=f"{selected_symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # MACD and Bollinger Bands
                    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.1,
                                       subplot_titles=('MACD', 'Bollinger Bands'))
                    
                    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=1, col=1)
                    fig2.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line'), row=1, col=1)
                    
                    fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'), row=2, col=1)
                    fig2.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='Upper BB'), row=2, col=1)
                    fig2.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='Middle BB'), row=2, col=1)
                    fig2.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='Lower BB'), row=2, col=1)
                    
                    fig2.update_layout(height=800)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Key statistics
                st.subheader("Key Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                with col2:
                    st.metric("Daily Return", f"{df['Returns'].iloc[-1]*100:.2f}%")
                with col3:
                    st.metric("Volatility", f"{df['Volatility'].iloc[-1]*100:.2f}%")
                with col4:
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                
                # Export data
                st.markdown(get_download_link(df, f"{selected_symbol}_data.csv"), unsafe_allow_html=True)
            
            with tab2:
                st.header("Portfolio Analysis")
                
                if len(symbols) > 1:
                    # Calculate portfolio metrics
                    results, mean_returns, cov_matrix = calculate_portfolio_metrics(data_dict)
                    
                    # Plot efficient frontier
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results[:, 0],
                        y=results[:, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=results[:, 1],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name='Portfolios'
                    ))
                    
                    fig.update_layout(
                        title='Efficient Frontier',
                        xaxis_title='Portfolio Risk (Std Dev)',
                        yaxis_title='Portfolio Return',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display correlation matrix
                    st.subheader("Correlation Matrix")
                    fig_corr = px.imshow(cov_matrix,
                                       title='Asset Correlation Matrix',
                                       color_continuous_scale='RdBu')
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Portfolio statistics
                    st.subheader("Portfolio Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Mean Returns:")
                        for symbol, ret in mean_returns.items():
                            st.write(f"{symbol}: {ret*100:.2f}%")
                    
                    with col2:
                        st.write("Volatility:")
                        for symbol in symbols:
                            vol = data_dict[symbol]['Volatility'].iloc[-1]
                            st.write(f"{symbol}: {vol*100:.2f}%")
                else:
                    st.warning("Please add more symbols for portfolio analysis")
            
            with tab3:
                st.header("News Analysis")
                
                selected_symbol = st.selectbox("Select Symbol for News", symbols)
                news_data = get_news_sentiment(selected_symbol)
                
                if news_data:
                    # Display sentiment
                    st.subheader("News Sentiment")
                    sentiment = news_data['mean_sentiment']
                    st.metric("Overall Sentiment", f"{sentiment:.2f}")
                    
                    # Display news articles
                    st.subheader("Recent News")
                    for article in news_data['news']:
                        title = article.get('title', 'No Title Available')
                        with st.expander(title):
                            st.write(f"Source: {article.get('publisher', 'N/A')}")
                            st.write(f"Published: {datetime.fromtimestamp(article.get('providerPublishTime', 0))}")
                            st.write(f"Link: {article.get('link', 'N/A')}")
                else:
                    st.warning("No news data available")
            
            with tab4:
                st.header("Prediction Analysis")
                
                selected_symbol = st.selectbox("Select Symbol for Prediction", symbols)
                df = data_dict[selected_symbol]
                
                # Simple moving average prediction
                st.subheader("Moving Average Prediction")
                ma_period = st.slider("Moving Average Period", 5, 50, 20)
                
                df['MA_Prediction'] = df['Close'].rolling(window=ma_period).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual Price'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_Prediction'], name='MA Prediction'))
                
                fig.update_layout(
                    title=f'{selected_symbol} Price Prediction (MA{ma_period})',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction metrics
                mse = np.mean((df['Close'] - df['MA_Prediction'])**2)
                rmse = np.sqrt(mse)
                
                st.metric("Root Mean Square Error", f"${rmse:.2f}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data from Yahoo Finance") 