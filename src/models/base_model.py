from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, name):
        """
        Initialize the base model.
        
        Args:
            name (str): Name of the model
        """
        self.name = name
        self.model = None
        
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        pass
        
    @abstractmethod
    def predict(self, X_test):
        """
        Make predictions using the trained model.
        
        Args:
            X_test: Test features
            
        Returns:
            np.array: Predicted values
        """
        pass
        
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance using common metrics.
        
        Args:
            y_true (np.array): Actual values
            y_pred (np.array): Predicted values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        logger.info(f"{self.name} Model Evaluation:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R2 Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def save_model(self, path):
        """
        Save the trained model.
        Args:
            path (str): Path to save the model.
        """
        logger.warning(f"Save method not implemented for {self.name} base model.")

    def load_model(self, path):
        """
        Load a trained model.
        Args:
            path (str): Path to load the model from.
        """
        logger.warning(f"Load method not implemented for {self.name} base model.") 