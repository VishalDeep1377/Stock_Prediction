import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        """
        Initializes the ModelEvaluator.
        """
        self.metrics = {}
        logger.info("ModelEvaluator initialized.")

    def evaluate_model(self, model_name, y_true, y_pred):
        """
        Evaluates a single model's performance.
        Args:
            model_name (str): Name of the model.
            y_true (pd.Series or np.array): Actual values.
            y_pred (pd.Series or np.array): Predicted values.
        Returns:
            dict: Dictionary of evaluation metrics for the model.
        """
        # Ensure y_true and y_pred are pandas Series and align them by index
        if isinstance(y_true, np.ndarray):
            y_true = pd.Series(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred)
        
        # Align indices for accurate comparison
        common_index = y_true.index.intersection(y_pred.index)
        y_true_aligned = y_true.loc[common_index]
        y_pred_aligned = y_pred.loc[common_index]

        if y_true_aligned.empty or y_pred_aligned.empty:
            logger.warning(f"Cannot evaluate {model_name}: No common dates between true and predicted values.")
            return {}

        mse = mean_squared_error(y_true_aligned, y_pred_aligned)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
        r2 = r2_score(y_true_aligned, y_pred_aligned)
        
        model_metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2
        }
        self.metrics[model_name] = model_metrics
        
        logger.info(f"--- {model_name} Model Evaluation ---")
        for metric, value in model_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return model_metrics

    def compare_models(self, metric='RMSE', save_path=None):
        """
        Compares all evaluated models based on a specified metric and visualizes the comparison.
        Args:
            metric (str): The metric to use for comparison (e.g., 'RMSE', 'MAE', 'R2_Score').
            save_path (str, optional): Path to save the comparison plot.
        """
        if not self.metrics:
            logger.warning("No model metrics available for comparison. Run evaluate_model first.")
            return

        comparison_data = []
        for model_name, model_metrics in self.metrics.items():
            if metric in model_metrics:
                comparison_data.append({'Model': model_name, metric: model_metrics[metric]})
            else:
                logger.warning(f"Metric '{metric}' not found for model '{model_name}'. Skipping.")

        if not comparison_data:
            logger.warning(f"No models found with the metric '{metric}' for comparison.")
            return

        df_comparison = pd.DataFrame(comparison_data)
        df_comparison_sorted = df_comparison.sort_values(by=metric, ascending=(metric != 'R2_Score'))

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y=metric, data=df_comparison_sorted, palette='viridis')
        plt.title(f'Model Comparison by {metric}', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Model comparison plot saved to {save_path}")
        plt.show()
        logger.info(f"Model comparison based on {metric} completed.")

    def get_all_metrics(self):
        """
        Returns all collected metrics for all models.
        Returns:
            dict: A dictionary containing metrics for all evaluated models.
        """
        return self.metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running ModelEvaluator example.")

    # Create dummy true and predicted values for demonstration
    np.random.seed(42)
    y_true_data = np.random.rand(100) * 100
    y_true_index = pd.date_range(start='2023-01-01', periods=100, freq='D')
    y_true = pd.Series(y_true_data, index=y_true_index)

    # Simulate predictions for two models
    y_pred_arima_data = y_true_data + np.random.randn(100) * 5
    y_pred_arima = pd.Series(y_pred_arima_data, index=y_true_index)

    y_pred_sarima_data = y_true_data + np.random.randn(100) * 3
    y_pred_sarima = pd.Series(y_pred_sarima_data, index=y_true_index)

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Evaluate models
    evaluator.evaluate_model('ARIMA', y_true, y_pred_arima)
    evaluator.evaluate_model('SARIMA', y_true, y_pred_sarima)

    # Compare models
    evaluator.compare_models(metric='RMSE', save_path='reports/model_rmse_comparison.png')
    evaluator.compare_models(metric='MAE', save_path='reports/model_mae_comparison.png')

    all_metrics = evaluator.get_all_metrics()
    logger.info(f"All collected metrics: {all_metrics}")

    logger.info("ModelEvaluator example finished.") 