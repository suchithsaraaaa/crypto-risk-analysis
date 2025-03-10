# Cryptocurrency Prediction & Risk Management App

A comprehensive cryptocurrency analysis application using machine learning for market prediction and risk management. The application provides detailed information about selected cryptocurrencies, including market analysis, price predictions, and risk assessment metrics, with robust error handling and fallback mechanisms.

## Features

- Cryptocurrency selection with detailed coin information
- Price history visualization and prediction
- Risk assessment metrics and analysis
- Real-time risk monitoring with dynamic indicators
- Animated risk visualization with color-changing effects
- Risk simulation capabilities for alert testing
- Market dominance visualization
- Fear and greed index tracking
- Developer and community stats
- ML-powered price predictions

## Installation

1. Clone this repository to your local machine
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install matplotlib numpy pandas plotly requests scikit-learn streamlit
```

## Running the Application

Run the Streamlit application:
```bash
streamlit run main.py
```

The application will be available at http://localhost:8501

## Files

- `main.py`: Main application entry point and UI components
- `data_retrieval.py`: Cryptocurrency data retrieval functions
- `ml_models.py`: Machine learning models for price prediction
- `utils.py`: Utility functions for data formatting
- `visualizations.py`: Data visualization components
- `.streamlit/config.toml`: Streamlit configuration
- `assets/custom.css`: Custom styling

## API Usage

The application uses the following APIs:
- CoinGecko API for cryptocurrency data
- Alternative.me API for Fear & Greed Index
- CryptoCompare API for news (optional)

Note: The application includes fallback mechanisms when APIs are rate-limited or unavailable.

## Notes for Local Use

- CoinGecko API has rate limits that may affect data retrieval
- For better performance, consider obtaining API keys for the services
- Set any API keys as environment variables before running the application

## Risk Monitoring Features

The application includes advanced risk monitoring capabilities:

- Real-time risk monitoring dashboard with animated indicators
- Color-coded risk levels that dynamically change based on market conditions
- Pulsing effects that intensify with higher risk levels
- Risk simulation tools for testing alert scenarios
- Detailed risk descriptions and personalized recommendations based on user risk tolerance
- Multiple visualization methods (animated meters, real-time monitors, traditional gauges)