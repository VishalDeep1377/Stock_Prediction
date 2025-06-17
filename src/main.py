import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
from pathlib import Path
import time
import requests
from .data_preprocessor import StockDataPreprocessor
from .visualization.visualizer import StockVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary output directories if they don't exist."""
    # Get the project root directory (parent of src)
    project_root = Path(__file__).parent.parent
    directories = ['data', 'models', 'visuals', 'reports']
    for directory in directories:
        os.makedirs(project_root / directory, exist_ok=True)
    logger.info("Ensured all project directories exist.")

def download_stock_data(symbol, start_date, end_date, max_retries=3):
    """Download stock data with retry mechanism using Ticker object."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} to download data for {symbol}")
            
            # Create Ticker object
            ticker = yf.Ticker(symbol)
            
            # Verify the ticker exists
            info = ticker.info
            if not info:
                logger.error(f"Invalid ticker symbol: {symbol}")
                return None
                
            # Download historical data
            df = ticker.history(start=start_date, end=end_date)
            
            if not df.empty:
                logger.info(f"Successfully downloaded {len(df)} rows of data for {symbol}")
                return df
                
            logger.warning(f"Empty dataframe received for {symbol}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during download attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait 5 seconds before retrying
            continue
        except Exception as e:
            logger.error(f"Error during download attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait 5 seconds before retrying
            continue
    return None

def main():
    create_directories()
    
    # --- Configuration Parameters ---
    symbol = 'AAPL'  # Example stock symbol (Apple Inc.)
    
    # Use current date as end date and 5 years before as start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)
    
    # Format dates for yfinance
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    logger.info(f"Starting stock market analysis for {symbol} from {start_date_str} to {end_date_str}")
    
    # --- Week 1-2: Data Collection ---
    logger.info("Phase 1: Data Collection")
    
    df = download_stock_data(symbol, start_date_str, end_date_str)
    if df is None or df.empty:
        logger.error(f"Failed to download data for {symbol} after multiple attempts. Exiting.")
        return
        
    # Save raw data
    data_path = Path(__file__).parent.parent / 'data' / f"{symbol}_historical_data.csv"
    df.to_csv(data_path)
    logger.info(f"Successfully downloaded and saved historical data for {symbol} to {data_path}")
    
    # --- Week 3-4: Data Cleaning & Visualization ---
    logger.info("Phase 2: Data Cleaning & Visualization")
    
    # Initialize preprocessor and process data
    preprocessor = StockDataPreprocessor(df)
    df_processed = preprocessor.handle_missing_values()
    df_processed = preprocessor.add_technical_indicators()
    df_processed = preprocessor.add_time_features()
    df_processed = preprocessor.add_price_features()
    
    # Save processed data
    processed_data_path = Path(__file__).parent.parent / 'data' / f"{symbol}_processed_data.csv"
    df_processed.to_csv(processed_data_path)
    logger.info(f"Processed data saved to {processed_data_path}")
    
    # Initialize visualizer and create plots
    visualizer = StockVisualizer(df_processed)
    
    # Create and save various plots
    visuals_dir = Path(__file__).parent.parent / 'visuals'
    visualizer.plot_price_history(save_path=visuals_dir / f"{symbol}_price_history.png")
    visualizer.plot_moving_averages(save_path=visuals_dir / f"{symbol}_moving_averages.png")
    visualizer.plot_technical_indicators(save_path=visuals_dir / f"{symbol}_technical_indicators.png")
    visualizer.plot_returns_distribution(save_path=visuals_dir / f"{symbol}_returns_distribution.png")
    visualizer.plot_volatility(save_path=visuals_dir / f"{symbol}_volatility.png")
    visualizer.plot_correlation_heatmap(save_path=visuals_dir / f"{symbol}_correlation_heatmap.png")
    
    logger.info("Data processing and visualization complete. Check the 'visuals' directory for plots.")
    
    # Prepare data for modeling
    X, y = preprocessor.prepare_for_modeling()
    logger.info(f"Data prepared for modeling with {len(X.columns)} features")
    
    # Next steps will involve implementing:
    # 1. ARIMA model
    # 2. SARIMA model
    # 3. Model evaluation and comparison
    
    logger.info("Project setup for main execution complete. Please run 'python -m src.main' after all modules are created and saved.")

if __name__ == "__main__":
    main() 