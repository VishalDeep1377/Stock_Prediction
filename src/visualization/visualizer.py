import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path
from typing import Optional, List, Union

logger = logging.getLogger(__name__)

class StockVisualizer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualizer with a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with stock data
        """
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        
        # Set style
        plt.style.use('seaborn-v0_8')  # Using a valid seaborn style
        sns.set_theme()  # Initialize seaborn with default theme
        
        logger.info("StockVisualizer initialized.")

    def plot_price_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the historical price data.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Close'], label='Close Price')
        plt.title('Historical Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved price history plot to {save_path}")
        plt.close()
        
    def plot_moving_averages(self, save_path: Optional[str] = None) -> None:
        """
        Plot the price with moving averages.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Close'], label='Close Price', alpha=0.5)
        plt.plot(self.df.index, self.df['MA20'], label='20-day MA', linewidth=2)
        plt.plot(self.df.index, self.df['MA50'], label='50-day MA', linewidth=2)
        plt.plot(self.df.index, self.df['MA200'], label='200-day MA', linewidth=2)
        
        plt.title('Stock Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved moving averages plot to {save_path}")
        plt.close()
        
    def plot_technical_indicators(self, save_path: Optional[str] = None) -> None:
        """
        Plot technical indicators (RSI, MACD).
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        
        # Plot price and Bollinger Bands
        ax1.plot(self.df.index, self.df['Close'], label='Close Price')
        ax1.plot(self.df.index, self.df['BB_Upper'], 'r--', label='Upper BB')
        ax1.plot(self.df.index, self.df['BB_Middle'], 'g--', label='Middle BB')
        ax1.plot(self.df.index, self.df['BB_Lower'], 'r--', label='Lower BB')
        ax1.set_title('Price with Bollinger Bands')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot RSI
        ax2.plot(self.df.index, self.df['RSI'], label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved technical indicators plot to {save_path}")
        plt.close()
        
    def plot_returns_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of returns.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['Returns'].dropna(), kde=True)
        plt.title('Distribution of Daily Returns')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved returns distribution plot to {save_path}")
        plt.close()
        
    def plot_volatility(self, save_path: Optional[str] = None) -> None:
        """
        Plot the rolling volatility.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Volatility'])
        plt.title('20-day Rolling Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved volatility plot to {save_path}")
        plt.close()
        
    def plot_correlation_heatmap(self, save_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap of numeric features.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved correlation heatmap to {save_path}")
        plt.close()
        
    def plot_predictions(self, original_data: pd.Series, predictions: pd.Series,
                        title: str = 'Stock Price Predictions',
                        save_path: Optional[str] = None) -> None:
        """
        Plot original data with predictions.
        
        Args:
            original_data (pd.Series): Original price data
            predictions (pd.Series): Predicted price data
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(original_data.index, original_data, label='Actual', alpha=0.7)
        plt.plot(predictions.index, predictions, label='Predicted', linestyle='--')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved predictions plot to {save_path}")
        plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running StockVisualizer example.")
    
    # Create a dummy DataFrame for demonstration
    data = {
        'Open': [100, 102, 101, 103, 105, 104, 106, 107, 108, 109],
        'High': [103, 104, 103, 105, 107, 106, 108, 109, 110, 111],
        'Low': [99, 101, 100, 102, 104, 103, 105, 106, 107, 108],
        'Close': [102, 101, 103, 105, 104, 106, 107, 108, 109, 110],
        'Volume': [100000, 120000, 110000, 130000, 140000, 125000, 135000, 145000, 150000, 160000]
    }
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    dummy_df = pd.DataFrame(data, index=dates)

    visualizer = StockVisualizer(dummy_df)
    visualizer.plot_price_history(save_path='visuals/example_price_history.png')
    
    # To test other plots, ensure the dummy_df has the necessary columns (e.g., after preprocessing)
    # For MA, RSI, MACD, assume preprocessor has run:
    from src.data_preprocessor import StockDataPreprocessor
    preprocessor = StockDataPreprocessor(dummy_df.copy())
    dummy_df_processed = preprocessor.handle_missing_values()
    dummy_df_processed = preprocessor.add_time_features()
    dummy_df_processed = preprocessor.add_technical_indicators()

    visualizer_processed = StockVisualizer(dummy_df_processed)
    visualizer_processed.plot_moving_averages(save_path='visuals/example_moving_averages.png')
    visualizer_processed.plot_technical_indicators(save_path='visuals/example_indicators.png')
    visualizer_processed.plot_acf_pacf(save_path='visuals/example_acf_pacf.png')

    # Example for plot_predictions
    # Create dummy predictions for demonstration
    forecast_dates = pd.date_range(start=dummy_df_processed.index[-1] + pd.Timedelta(days=1), periods=5, freq='D')
    dummy_predictions = pd.DataFrame({'Forecast': np.array([111, 112, 113, 114, 115])}, index=forecast_dates)

    visualizer_processed.plot_predictions(
        original_data=dummy_df_processed['Close'], 
        predictions=dummy_predictions['Forecast'],
        title='Stock Price Predictions',
        save_path='visuals/example_predictions.png'
    )

    logger.info("StockVisualizer example finished.")