import pandas as pd
import numpy as np
from typing import Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the preprocessor with a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with stock data
        """
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        self.df.sort_index(inplace=True)
        logger.info("StockDataPreprocessor initialized.")

    def handle_missing_values(self, method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            method (str): Method to handle missing values ('ffill', 'bfill', or 'interpolate')
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        if method == 'ffill':
            self.df = self.df.ffill().bfill()  # Forward fill then backward fill
        elif method == 'bfill':
            self.df = self.df.bfill().ffill()  # Backward fill then forward fill
        elif method == 'interpolate':
            self.df = self.df.interpolate(method='time')
            
        logger.info(f"Handled missing values using {method} method")
        return self.df

    def select_target_and_features(self, target_column='Close', features=None):
        """
        Selects the target column and specified features. If no features are specified,
        it uses a default set of relevant columns.
        Args:
            target_column (str): The column to be used as the prediction target.
            features (list, optional): A list of column names to use as features. Defaults to None.
        Returns:
            pd.DataFrame: DataFrame with selected target and features.
        """
        if features is None:
            # Default features, excluding target if it's in the list
            default_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            features = [f for f in default_features if f != target_column]

        if target_column not in self.df.columns:
            logger.error(f"Target column '{target_column}' not found in DataFrame.")
            raise ValueError(f"Target column '{target_column}' not found.")

        # Ensure all selected features exist
        for f in features:
            if f not in self.df.columns:
                logger.warning(f"Feature '{f}' not found in DataFrame. Skipping this feature.")
                features.remove(f)
        
        # Ensure target column is included for models that predict its transformed value
        all_columns = list(set(features + [target_column]))
        self.df = self.df[all_columns].copy()
        logger.info(f"Selected target '{target_column}' and features: {features}")
        return self.df

    def add_time_features(self) -> pd.DataFrame:
        """
        Add time-based features to the dataset.
        
        Returns:
            pd.DataFrame: DataFrame with added time features
        """
        # Extract time features from index
        self.df['Year'] = self.df.index.year
        self.df['Month'] = self.df.index.month
        self.df['Day'] = self.df.index.day
        self.df['DayOfWeek'] = self.df.index.dayofweek
        self.df['Quarter'] = self.df.index.quarter
        
        # Add cyclical features for month and day of week
        self.df['Month_sin'] = np.sin(2 * np.pi * self.df['Month']/12)
        self.df['Month_cos'] = np.cos(2 * np.pi * self.df['Month']/12)
        self.df['DayOfWeek_sin'] = np.sin(2 * np.pi * self.df['DayOfWeek']/7)
        self.df['DayOfWeek_cos'] = np.cos(2 * np.pi * self.df['DayOfWeek']/7)
        
        logger.info("Added time-based features")
        return self.df

    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Add technical indicators to the dataset.
        
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        # Moving Averages
        self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA50'] = self.df['Close'].rolling(window=50).mean()
        self.df['MA200'] = self.df['Close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        self.df['BB_Middle'] = self.df['Close'].rolling(window=20).mean()
        self.df['BB_Upper'] = self.df['BB_Middle'] + 2 * self.df['Close'].rolling(window=20).std()
        self.df['BB_Lower'] = self.df['BB_Middle'] - 2 * self.df['Close'].rolling(window=20).std()
        
        logger.info("Added technical indicators: MA20, MA50, MA200, RSI, MACD, Bollinger Bands")
        return self.df

    def add_price_features(self) -> pd.DataFrame:
        """
        Add price-based features to the dataset.
        
        Returns:
            pd.DataFrame: DataFrame with added price features
        """
        # Daily returns
        self.df['Returns'] = self.df['Close'].pct_change()
        
        # Log returns
        self.df['Log_Returns'] = np.log(self.df['Close']/self.df['Close'].shift(1))
        
        # Volatility (20-day rolling standard deviation of returns)
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
        
        # Price momentum (5-day change)
        self.df['Momentum'] = self.df['Close'].pct_change(periods=5)
        
        logger.info("Added price-based features: Returns, Log Returns, Volatility, Momentum")
        return self.df

    def prepare_for_modeling(self, target_column: str = 'Close', 
                           feature_columns: Optional[List[str]] = None) -> tuple:
        """
        Prepare data for modeling by splitting into features and target.
        
        Args:
            target_column (str): Column to use as target variable
            feature_columns (List[str]): List of columns to use as features
            
        Returns:
            tuple: (X, y) where X is features and y is target
        """
        if feature_columns is None:
            # Use all numeric columns except the target
            feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        X = self.df[feature_columns]
        y = self.df[target_column]
        
        # Remove any rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared data for modeling with {len(feature_columns)} features")
        return X, y

    def get_processed_data(self):
        """
        Returns the processed DataFrame.
        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        return self.df

if __name__ == "__main__":
    # Example Usage (for testing the module independently)
    logging.basicConfig(level=logging.INFO)
    logger.info("Running StockDataPreprocessor example.")
    
    # Create a dummy DataFrame for demonstration
    data = {
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100 + 5,
        'Low': np.random.rand(100) * 100 - 5,
        'Close': np.random.rand(100) * 100,
        'Volume': np.random.randint(100000, 1000000, 100)
    }
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    dummy_df = pd.DataFrame(data, index=dates)
    
    # Introduce some NaN values for testing missing value handling
    dummy_df.iloc[10:15, 0] = np.nan 
    dummy_df.iloc[20:22, 4] = np.nan

    preprocessor = StockDataPreprocessor(dummy_df)
    
    df_cleaned = preprocessor.handle_missing_values()
    df_features = preprocessor.add_time_features()
    df_indicators = preprocessor.add_technical_indicators()
    
    final_df = preprocessor.get_processed_data()
    
    logger.info(f"Original data shape: {dummy_df.shape}")
    logger.info(f"Final processed data shape: {final_df.shape}")
    logger.info(f"Columns in final processed data: {final_df.columns.tolist()}")
    logger.info("StockDataPreprocessor example finished.") 