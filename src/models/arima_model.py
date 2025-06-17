import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class ARIMAModel(BaseModel):
    def __init__(self, order=(5,1,0)):
        """
        Initialize the ARIMA model.
        
        Args:
            order (tuple): The (p,d,q) order of the model. Default is (5,1,0).
        """
        super().__init__("ARIMA")
        self.order = order
        self.model = None
        self.model_fit = None
        
    def train(self, X_train, y_train):
        """
        Train the ARIMA model.
        
        Args:
            X_train: Training features (not directly used by ARIMA, but kept for signature compatibility).
            y_train: Training targets (pandas Series expected for ARIMA).
        """
        logger.info(f"Training ARIMA model with order {self.order}...")
        try:
            # ARIMA model expects a Series, not a DataFrame for y_train
            if isinstance(y_train, pd.DataFrame):
                if y_train.shape[1] > 1:
                    logger.warning("ARIMA received a DataFrame with multiple columns for y_train. Using the first column.")
                y_train_series = y_train.iloc[:, 0] # Take the first column if it's a DataFrame
            else:
                y_train_series = y_train

            self.model = ARIMA(y_train_series, order=self.order)
            self.model_fit = self.model.fit()
            logger.info("ARIMA model training complete.")
            logger.info(self.model_fit.summary())
        except Exception as e:
            logger.error(f"Error during ARIMA model training: {e}")
            self.model_fit = None # Ensure model_fit is None if training fails

    def predict(self, X_test=None, n_periods=1):
        """
        Make predictions using the trained ARIMA model.
        
        Args:
            X_test: Test features (not directly used by ARIMA for out-of-sample prediction).
            n_periods (int): Number of future periods to forecast.
            
        Returns:
            pd.Series: Predicted values with a DatetimeIndex.
        """
        if self.model_fit is None:
            logger.error("ARIMA model has not been trained. Please run train() first.")
            return pd.Series()

        logger.info(f"Making {n_periods} ARIMA predictions.")
        try:
            # For out-of-sample forecasts, ARIMA uses `forecast()` method.
            # The index for predictions needs to be generated manually.
            # Get the last date from the training data (assuming y_train was a Series with datetime index)
            last_date = self.model_fit.data.dates[-1] if self.model_fit.data.dates is not None else pd.to_datetime(pd.Timestamp.now().date())
            
            # Generate future dates
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq='D')
            
            forecast_result = self.model_fit.forecast(steps=n_periods)
            
            # If forecast_result is a pandas Series, use its index. Otherwise, assign new dates.
            if isinstance(forecast_result, pd.Series):
                predictions = forecast_result
                # Ensure the index matches the expected future_dates if it's not already aligned
                if not predictions.index.equals(future_dates):
                    predictions.index = future_dates
            else:
                predictions = pd.Series(forecast_result, index=future_dates)

            logger.info("ARIMA predictions generated.")
            return predictions
        except Exception as e:
            logger.error(f"Error during ARIMA prediction: {e}")
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
        # Ensure y_true and y_pred are aligned for evaluation
        common_index = y_true.index.intersection(y_pred.index)
        y_true_aligned = y_true.loc[common_index]
        y_pred_aligned = y_pred.loc[common_index]

        if y_true_aligned.empty or y_pred_aligned.empty:
            logger.warning("Cannot evaluate: No common dates between true and predicted values.")
            return {}

        # Call the base model's evaluate method
        return super().evaluate(y_true_aligned, y_pred_aligned)

    def save_model(self, path):
        """
        Save the trained ARIMA model.
        Args:
            path (str): Path to save the model.
        """
        if self.model_fit is not None:
            try:
                self.model_fit.save(path)
                logger.info(f"ARIMA model saved to {path}")
            except Exception as e:
                logger.error(f"Error saving ARIMA model to {path}: {e}")
        else:
            logger.warning("No trained ARIMA model to save.")

    def load_model(self, path):
        """
        Load a trained ARIMA model.
        Args:
            path (str): Path to load the model from.
        """
        try:
            self.model_fit = ARIMA.load(path)
            logger.info(f"ARIMA model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading ARIMA model from {path}: {e}")
            self.model_fit = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running ARIMAModel example.")

    # Create a dummy time series for demonstration
    np.random.seed(42)
    data = np.random.randn(100).cumsum() + 100 # Simple random walk like stock price
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    dummy_series = pd.Series(data, index=dates)

    # Split data into train and test
    train_size = int(len(dummy_series) * 0.8)
    train_data = dummy_series[:train_size]
    test_data = dummy_series[train_size:]

    # Initialize and train ARIMA model
    arima_model = ARIMAModel(order=(5,1,0)) # Example ARIMA order
    arima_model.train(None, train_data) # X_train is None for ARIMA

    # Make predictions for the test set
    # For in-sample prediction using fit().predict()
    # For out-of-sample prediction using fit().forecast()
    
    # For simplicity, let's predict for the length of the test set
    # ARIMA's predict function on model_fit can do in-sample and out-of-sample.
    # For out-of-sample, it's generally best to use forecast() as used in the class method.
    
    # Let's get predictions for the next 20 days (length of dummy test_data)
    n_forecast_periods = len(test_data)
    predictions = arima_model.predict(n_periods=n_forecast_periods)

    # Evaluate the model (only if predictions are not empty)
    if not predictions.empty and not test_data.empty:
        arima_model.evaluate(test_data, predictions)

    logger.info("ARIMAModel example finished.") 