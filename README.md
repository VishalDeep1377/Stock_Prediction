# Stock Market Forecasting System for Zudio Internship

This project develops a comprehensive stock market time series forecasting system, fulfilling the requirements for my Zudio internship. It covers data collection, preprocessing, model implementation (ARIMA, SARIMA, Prophet, LSTM), evaluation, visualization, and optional web deployment.

## 📊 Project Objectives

- Collect and preprocess historical stock market data.
- Implement various time series models: ARIMA, SARIMA, Prophet, and LSTM.
- Evaluate and compare the accuracy of different forecasting models.
- Visualize key insights, trends, seasonality, and model predictions.
- (Optional) Deploy a user-friendly web interface for interactive forecasting.

## 🛠️ Tech Stack & Tools

- **Language**: Python
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Data Acquisition**: `yfinance`
- **Statistical Models**: `statsmodels`, `pmdarima` (for ARIMA/SARIMA)
- **Prophet Model**: `prophet` (Facebook Prophet)
- **Deep Learning Model**: `tensorflow`, `keras` (for LSTM)
- **Model Evaluation**: `scikit-learn`
- **Web Deployment (Optional)**: `streamlit` or `flask`
- **Development Environment**: Visual Studio Code

## 📁 Project Folder Structure

```
stock_forecasting_project/
├── data/                  # Stores raw and cleaned historical stock data (CSV)
├── notebooks/            # Jupyter notebooks for exploratory data analysis (EDA) and experimentation
├── models/                # Saved trained models (e.g., .pkl, .h5 files)
├── visuals/               # Generated plots and charts (PNG, HTML)
├── reports/               # Model comparison tables, evaluation reports (Markdown, Excel)
├── app/                   # Source code for the optional Streamlit/Flask web application
├── main.py                # Main script to run the entire forecasting pipeline
├── requirements.txt       # Lists all Python dependencies and their versions
└── README.md              # Project documentation, setup, and usage instructions
```

## 🚀 Setup and Installation

**IMPORTANT:** For `tensorflow` and `pmdarima` to install correctly on Windows, you **MUST** have the **"Desktop development with C++" workload** installed via the Visual Studio Build Tools. Download from [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select this workload during installation.

1.  **Clone the repository** (if applicable, or create the `stock_forecasting_project` directory).

2.  **Install Python 3.10 (Recommended for TensorFlow/pmdarima compatibility):**
    If you don't have Python 3.10 installed, download it from [https://www.python.org/downloads/release/python-31012/](https://www.python.org/downloads/release/python-31012/). During installation, ensure you check "Add Python 3.10 to PATH".

3.  **Create and Activate a Virtual Environment:**
    Open your terminal (Command Prompt or PowerShell) **as an Administrator**.
    Navigate to your project root directory: `cd D:\Internship project`

    Then, create the virtual environment using Python 3.10 (adjust path as per your installation):
    ```bash
    "C:\Users\YOUR_WINDOWS_USERNAME\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv_py310
    ```
    (Replace `YOUR_WINDOWS_USERNAME` with your actual Windows username. Common path alternatives: `C:\Python310\python.exe` or `C:\Program Files\Python310\python.exe`)

    Activate the virtual environment:
    ```bash
    .\venv_py310\Scripts\activate
    ```
    (You should see `(venv_py310)` at the start of your terminal prompt).

4.  **Install Dependencies:**
    With the virtual environment active, install all required packages:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 How to Run the Project

After setting up the environment and installing dependencies, run the main script from your project root directory:

```bash
python main.py
```

This script orchestrates the entire pipeline: data collection, preprocessing, model training, evaluation, and visualization. Output files (data, models, visuals, reports) will be saved in their respective directories.

## 🎯 Project Stages and Deliverables

### 🗓️ Week 1-2: Data Collection & Time Series Concepts
- `src/data_collector.py`: Python script to fetch historical stock data (e.g., AAPL) using `yfinance`.
- `data/AAPL_historical_data.csv`: Saved raw historical data.

### 🧹 Week 3-4: Data Cleaning & Visualization
- `src/data_preprocessor.py`: Python script for data cleaning (handling nulls, focusing on 'Close' price).
- `src/visualization.py`: Module for visualizing stock trends, moving averages, and seasonality.
- `visuals/price_trend.png`: Plot of historical stock prices.
- `visuals/moving_average.png`: Plot with moving averages.
- `notebooks/eda.ipynb`: Jupyter notebook for exploratory data analysis, including ACF/PACF plots.

### 📊 Week 5-6: ARIMA, SARIMA, Prophet Implementation
- `src/models/arima_model.py`: Implementation of ARIMA model with auto-selection or custom order.
- `src/models/sarima_model.py`: Implementation of SARIMA model for seasonal data.
- `src/models/prophet_model.py`: Implementation of Facebook Prophet model.
- `visuals/arima_forecast.png`, `visuals/sarima_forecast.png`, `visuals/prophet_forecast.png`: Plots showing 30-day forecasts vs. actual data.
- `models/arima_model.pkl`, `models/sarima_model.pkl`, `models/prophet_model.pkl`: Saved trained models.

### 🤖 Week 7-8: Deep Learning using LSTM
- `src/data_preprocessor.py`: Extended to include data scaling (`MinMaxScaler`) and creation of time windows for LSTM.
- `src/models/lstm_model.py`: Implementation of LSTM model using `tensorflow.keras`.
- `visuals/lstm_forecast.png`: Plot showing LSTM 30-day forecasts vs. actual data.
- `models/lstm_model.h5`: Saved trained LSTM model.

### 🔍 Week 9: Model Comparison
- `src/model_evaluator.py`: Module to calculate RMSE, MAE, MAPE for all models.
- `reports/model_comparison.md` (or `.xlsx`): Table comparing model accuracy and runtime.
- `visuals/model_comparison_chart.png`: Bar chart or similar visualizing model performance metrics.

### 🧑‍💻 Week 10-12: Final Output & Optional Deployment
- `README.md`: This comprehensive documentation.
- `app/streamlit_dashboard.py` (Optional): Streamlit application allowing user selection of stock, model, and date range for interactive forecasting.
- GitHub repository with the entire codebase.

## 📝 Code Standards

All Python scripts will adhere to:
- **Modularity**: Functions and classes for distinct tasks.
- **Readability**: Clear variable names, consistent formatting.
- **Comments**: Inline comments explaining complex logic, docstrings for functions/classes.
- **Error Handling**: Basic error handling for robust operation.

--- 