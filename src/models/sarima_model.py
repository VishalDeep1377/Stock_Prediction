import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class SARIMAModel(BaseModel):
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """
        Initialize the SARIMA model.
        
        Args:
            order (tuple): The (p,d,q) order of the non-seasonal part.
            seasonal_order (tuple): The (P,D,Q,S) order of the seasonal part.
        """
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        
    def train(self, X_train, y_train):
        """
        Train the SARIMA model.
        
        Args:
            X_train: Training features (not directly used by SARIMA, but kept for signature compatibility).
            y_train: Training targets (pandas Series expected for SARIMA).
        """
        logger.info(f"Training SARIMA model with order {self.order} and seasonal order {self.seasonal_order}...")
        try:
            if isinstance(y_train, pd.DataFrame):
                if y_train.shape[1] > 1:
                    logger.warning("SARIMA received a DataFrame with multiple columns for y_train. Using the first column.")
                y_train_series = y_train.iloc[:, 0] # Take the first column if it's a DataFrame
            else:
                y_train_series = y_train

            self.model = SARIMAX(y_train_series, order=self.order, seasonal_order=self.seasonal_order)
            self.model_fit = self.model.fit(disp=False) # disp=False to suppress convergence messages
            logger.info("SARIMA model training complete.")
            logger.info(self.model_fit.summary())
        except Exception as e:
            logger.error(f"Error during SARIMA model training: {e}")
            self.model_fit = None

    def predict(self, X_test=None, n_periods=1):
        """
        Make predictions using the trained SARIMA model.
        
        Args:
            X_test: Test features (not directly used by SARIMA for out-of-sample prediction).
            n_periods (int): Number of future periods to forecast.
            
        Returns:
            pd.Series: Predicted values with a DatetimeIndex.
        """
        if self.model_fit is None:
            logger.error("SARIMA model has not been trained. Please run train() first.")
            return pd.Series()

        logger.info(f"Making {n_periods} SARIMA predictions.")
        try:
            last_date = self.model_fit.data.dates[-1] if self.model_fit.data.dates is not None else pd.to_datetime(pd.Timestamp.now().date())
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq='D')
            
            forecast_result = self.model_fit.forecast(steps=n_periods)
            
            if isinstance(forecast_result, pd.Series):
                predictions = forecast_result
                if not predictions.index.equals(future_dates):
                    predictions.index = future_dates
            else:
                predictions = pd.Series(forecast_result, index=future_dates)

            logger.info("SARIMA predictions generated.")
            return predictions
        except Exception as e:
            logger.error(f"Error during SARIMA prediction: {e}")
            return pd.Series()

    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance using common metrics. Overrides base method for series.
        
        Args:
            y_true (pd.Series): Actual values
            y_pred (pd.Series): Predicted values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        common_index = y_true.index.intersection(y_pred.index)
        y_true_aligned = y_true.loc[common_index]
        y_pred_aligned = y_pred.loc[common_index]

        if y_true_aligned.empty or y_pred_aligned.empty:
            logger.warning("Cannot evaluate: No common dates between true and predicted values.")
            return {}

        return super().evaluate(y_true_aligned, y_pred_aligned)

    def save_model(self, path):
        """
        Save the trained SARIMA model.
        Args:
            path (str): Path to save the model.
        """
        if self.model_fit is not None:
            try:
                self.model_fit.save(path)
                logger.info(f"SARIMA model saved to {path}")
            except Exception as e:
                logger.error(f"Error saving SARIMA model to {path}: {e}")
        else:
            logger.warning("No trained SARIMA model to save.")

    def load_model(self, path):
        """
        Load a trained SARIMA model.
        Args:
            path (str): Path to load the model from.
        """
        try:
            self.model_fit = SARIMAX.load(path)
            logger.info(f"SARIMA model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading SARIMA model from {path}: {e}")
            self.model_fit = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running SARIMAModel example.")

    # Create a dummy time series for demonstration with seasonality
    np.random.seed(42)
    # Generate a series with a yearly trend and monthly seasonality
    index = pd.date_range(start='2020-01-01', periods=365*3, freq='D') # 3 years of daily data
    data = np.linspace(0, 50, len(index)) + np.sin(np.linspace(0, 3*2*np.pi, len(index))) * 10 + np.random.randn(len(index)) * 2
    dummy_series = pd.Series(data, index=index)

    # Split data into train and test
    train_size = int(len(dummy_series) * 0.8)
    train_data = dummy_series[:train_size]
    test_data = dummy_series[train_size:]

    # Initialize and train SARIMA model (example orders)
    # A seasonal order (1,1,1,12) means 1 seasonal AR, 1 seasonal differencing, 1 seasonal MA, with seasonality period of 12 (months)
    # For daily data, S=7 (weekly) or S=30 (monthly approximation) or S=365 (yearly) could be used.
    # Given daily data, a monthly seasonality (S=30) is common for finance.
    sarima_model = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,30))
    sarima_model.train(None, train_data) # X_train is None for SARIMA

    # Make predictions for the test set
    n_forecast_periods = len(test_data)
    predictions = sarima_model.predict(n_periods=n_forecast_periods)

    # Evaluate the model (only if predictions are not empty)
    if not predictions.empty and not test_data.empty:
        sarima_model.evaluate(test_data, predictions)

    logger.info("SARIMAModel example finished.") 