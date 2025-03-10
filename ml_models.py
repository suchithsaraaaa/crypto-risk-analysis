import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def prepare_time_series_data(df, target_col='price', window=7):
    """
    Prepare time series data for prediction by creating lag features
    
    Args:
        df: DataFrame containing historical price data
        target_col: Column to predict
        window: Number of lag features to create
        
    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    if df.empty:
        return None, None, None, None, None
    
    df = df.copy()
    
    # Create lag features
    for i in range(1, window + 1):
        df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
        df[f'volume_lag_{i}'] = df['volume'].shift(i)
    
    # Create technical indicators
    # Simple Moving Average (SMA)
    df['sma_5'] = df[target_col].rolling(window=5).mean()
    df['sma_10'] = df[target_col].rolling(window=10).mean()
    
    # Relative Strength Index (RSI) simplified
    delta = df[target_col].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 0.001)
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values created by shifting and technical indicators
    df = df.dropna()
    
    # Feature and target selection
    features = [col for col in df.columns if 'lag' in col or 'sma' in col or 'rsi' in col]
    X = df[features]
    y = df[target_col]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets (80/20)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler, features

def train_price_prediction_model(df, target_col='price', window=7):
    """
    Train a model to predict cryptocurrency prices
    
    Args:
        df: DataFrame containing historical price data
        target_col: Column to predict
        window: Number of lag features to create
        
    Returns:
        Trained model and evaluation metrics
    """
    X_train, y_train, X_test, y_test, scaler, features = prepare_time_series_data(df, target_col, window)
    
    if X_train is None:
        return None, None, None, None, None
    
    # Train random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, test_rmse, test_mae, test_r2, feature_importance

def predict_future_prices(model, df, days_to_predict=7, target_col='price', window=7):
    """
    Predict future cryptocurrency prices
    
    Args:
        model: Trained prediction model
        df: DataFrame containing historical price data
        days_to_predict: Number of days to predict into the future
        target_col: Column to predict
        window: Number of lag features used in model
        
    Returns:
        DataFrame containing predicted prices
    """
    if model is None or df.empty:
        return pd.DataFrame()
    
    # Prepare dataframe for prediction
    df = df.copy()
    
    # Create a copy of the last 'window' days of data
    future_df = df.tail(window).copy()
    
    # Initialize lists to store predictions and dates
    predictions = []
    dates = []
    last_date = df['date'].iloc[-1]
    
    # Create lag features for initial prediction
    for i in range(1, window + 1):
        future_df[f'{target_col}_lag_{i}'] = future_df[target_col].shift(i)
        future_df[f'volume_lag_{i}'] = future_df['volume'].shift(i)
    
    # Create technical indicators
    future_df['sma_5'] = future_df[target_col].rolling(window=5).mean()
    future_df['sma_10'] = future_df[target_col].rolling(window=10).mean()
    
    # Simplified RSI calculation
    delta = future_df[target_col].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    avg_loss = avg_loss.replace(0, 0.001)
    rs = avg_gain / avg_loss
    future_df['rsi'] = 100 - (100 / (1 + rs))
    
    # For each day we want to predict
    for i in range(days_to_predict):
        # Get features for prediction
        features = [col for col in future_df.columns if 'lag' in col or 'sma' in col or 'rsi' in col]
        
        # For the first prediction, use the most recent data point
        if i == 0:
            X_pred = future_df[features].iloc[-1].values.reshape(1, -1)
        else:
            # Update lag features based on previous predictions
            for j in range(window, 0, -1):
                if j > 1:
                    future_df.loc[future_df.index[-1], f'{target_col}_lag_{j}'] = future_df.loc[future_df.index[-1], f'{target_col}_lag_{j-1}']
                else:
                    future_df.loc[future_df.index[-1], f'{target_col}_lag_1'] = predicted_price
            
            # Update SMA and RSI (simplified)
            future_df.loc[future_df.index[-1], 'sma_5'] = future_df[target_col].iloc[-5:].mean()
            future_df.loc[future_df.index[-1], 'sma_10'] = future_df[target_col].iloc[-10:].mean()
            
            # Get features for next prediction
            X_pred = future_df[features].iloc[-1].values.reshape(1, -1)
        
        # Make prediction
        predicted_price = model.predict(X_pred)[0]
        
        # Calculate next date
        next_date = last_date + pd.Timedelta(days=i+1)
        
        # Store prediction and date
        predictions.append(predicted_price)
        dates.append(next_date)
        
        # Add prediction to dataframe for next iteration
        new_row = future_df.iloc[-1].copy()
        new_row['date'] = next_date
        new_row[target_col] = predicted_price
        new_row['volume'] = future_df['volume'].mean()  # Use average volume
        
        future_df = pd.concat([future_df, pd.DataFrame([new_row])])
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'date': dates,
        'predicted_price': predictions
    })
    
    return results_df

def calculate_risk_metrics(df, volatility_window=14):
    """
    Calculate risk metrics for a cryptocurrency
    
    Args:
        df: DataFrame containing historical price data
        volatility_window: Window size for volatility calculation
        
    Returns:
        Dictionary of risk metrics
    """
    if df.empty:
        return {
            "volatility": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "var_95": 0,
            "risk_level": "Unknown"
        }
    
    # Calculate daily returns
    returns = df['price'].pct_change().dropna()
    
    # Volatility (standard deviation of returns)
    volatility = returns.std() * np.sqrt(365)  # Annualized
    
    # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    avg_return = returns.mean()
    sharpe_ratio = (avg_return * 365) / volatility if volatility > 0 else 0
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5)
    
    # Determine risk level
    if volatility > 0.8:
        risk_level = "Very High"
    elif volatility > 0.6:
        risk_level = "High"
    elif volatility > 0.4:
        risk_level = "Medium-High"
    elif volatility > 0.2:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    return {
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "risk_level": risk_level
    }

def analyze_sentiment(news_data):
    """
    Simple sentiment analysis for cryptocurrency news
    
    Args:
        news_data: List of news articles
        
    Returns:
        Sentiment score (-1 to 1) and sentiment label
    """
    if not news_data:
        return 0, "Neutral"
    
    # Simplified sentiment analysis using keyword approach
    positive_words = [
        'bullish', 'surge', 'gain', 'positive', 'up', 'rise', 'soar', 'rally',
        'good', 'grow', 'growth', 'profit', 'success', 'adopt', 'adoption',
        'opportunity', 'innovation', 'partnership', 'launch', 'breakthrough'
    ]
    
    negative_words = [
        'bearish', 'crash', 'drop', 'fall', 'down', 'decrease', 'plunge', 'loss',
        'sell', 'selling', 'dump', 'fear', 'risk', 'ban', 'regulation', 'fraud',
        'hack', 'scam', 'bubble', 'volatility', 'concern', 'warning'
    ]
    
    positive_count = 0
    negative_count = 0
    
    for article in news_data:
        title = article['title'].lower()
        body = article['body'].lower()
        content = title + " " + body
        
        for word in positive_words:
            if word in content:
                positive_count += 1
        
        for word in negative_words:
            if word in content:
                negative_count += 1
    
    total = positive_count + negative_count
    
    if total == 0:
        return 0, "Neutral"
    
    sentiment_score = (positive_count - negative_count) / total
    
    if sentiment_score > 0.25:
        sentiment_label = "Positive"
    elif sentiment_score < -0.25:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    return sentiment_score, sentiment_label
