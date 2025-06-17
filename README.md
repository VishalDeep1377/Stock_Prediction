# Stock Market Analysis Dashboard

## Project Overview

This project delivers a comprehensive Stock Market Analysis Dashboard, developed rapidly within a one-month timeframe. It leverages Python and popular data science libraries to provide in-depth analysis of stock data, including historical trends, technical indicators, and interactive visualizations. The dashboard is designed to be user-friendly, offering insights into market performance and aiding in data-driven decision-decision-making.

## Key Features

*   **Automated Data Collection**: Downloads historical stock data using `yfinance`.
*   **Robust Data Preprocessing**: Handles missing values, adds essential technical indicators (e.g., Moving Averages, RSI, MACD, Bollinger Bands), and extracts time-based features.
*   **Interactive Visualizations**: Generates a variety of plots (price history, moving averages, technical indicators, returns distribution, volatility, correlation heatmaps) using Plotly and Matplotlib.
*   **Streamlit Web Dashboard**: A user-friendly web interface built with Streamlit for interactive exploration of stock data and analysis.
*   **Portfolio Analysis**: (As seen in `app.py`) Calculates portfolio metrics and visualizes efficient frontiers.
*   **News Sentiment Analysis**: (As seen in `app.py`) Integrates news sentiment for selected stocks.
*   **Export Capabilities**: Allows users to download processed data.

## Project Structure

```
.
├── data/
│   ├── AAPL_cleaned_data.csv
│   ├── AAPL_historical_data.csv
│   └── AAPL_processed_data.csv
├── src/
│   ├── app.py                     # Streamlit web application
│   ├── main.py                    # Main script for data pipeline (download, preprocess, visualize)
│   ├── data_preprocessor.py       # Handles data cleaning and feature engineering
│   ├── visualization/
│   │   └── visualizer.py          # Functions for generating plots
│   └── ... (other modules for models, evaluation, etc.)
├── requirements.txt               # Project dependencies
├── visuals/                       # Directory for saved plots
└── README.md                      # This file
```

## Setup and Installation

To get this project up and running locally, follow these steps:

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone https://github.com/VishalDeep1377/Stock_Prediction.git
    cd Stock_Prediction
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.

    *   **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Install all required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

There are two main ways to run this project:

### 1. Run the Data Pipeline (main.py)

This script will download data, preprocess it, and generate static visualization files in the `visuals/` directory.

```bash
python -m src.main
```
This will output various CSV files in the `data/` directory and PNG plots in the `visuals/` directory.

### 2. Run the Streamlit Web Dashboard (app.py)

This launches the interactive dashboard in your web browser. Ensure your virtual environment is activated.

```bash
streamlit run src/app.py
```
After running this command, your web browser should automatically open to the Streamlit application (usually `http://localhost:8501`).

## Distributing as a Standalone Executable (Windows)

For users who do not have Python installed, a Windows executable (`.exe`) can be generated. This `app.exe` (found in the `dist/` folder after running PyInstaller) bundles the entire application and its dependencies.

**Note for Users:** When running the `.exe` file, Windows SmartScreen or antivirus software might display a "not safe" warning because the application is not digitally signed. Users should click "More info" and then "Run anyway" to proceed, as this is a trusted application from this project.

## Development Timeline

This project was developed under an intensive one-month timeline, focusing on delivering core functionalities and a robust analysis pipeline efficiently. This concentrated effort allowed for rapid prototyping and deployment of the key features highlighted above.

## Contributing

For any questions, issues, or potential contributions, please refer to the project's GitHub repository.

## License

This project is open-source and available under the MIT License. See the `LICENSE` file (if applicable) for more details.
