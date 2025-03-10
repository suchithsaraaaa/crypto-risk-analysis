import pandas as pd
import numpy as np
import streamlit as st
import base64
from datetime import datetime, timedelta

def format_large_number(num):
    """Format large numbers with suffixes like K, M, B, T"""
    if num is None or num == 'N/A' or not isinstance(num, (int, float)):
        return "N/A"
    
    try:
        # Convert to float if it's a string representing a number
        if isinstance(num, str):
            num = float(num)
            
        magnitude = 0
        suffixes = ['', 'K', 'M', 'B', 'T']
        
        while abs(num) >= 1000 and magnitude < len(suffixes) - 1:
            magnitude += 1
            num /= 1000.0
        
        if magnitude > 0:
            # For numbers between 1 and 10, show one decimal place
            if 1 <= abs(num) < 10:
                return f"{num:.1f}{suffixes[magnitude]}"
            else:
                return f"{int(num)}{suffixes[magnitude]}"
        else:
            return f"{num}"
    except (ValueError, TypeError):
        return "N/A"

def format_currency(amount, precision=2):
    """Format currency with proper precision"""
    if amount is None:
        return "N/A"
    
    # Handle small amounts
    if 0 < abs(amount) < 0.01:
        return f"${amount:.6f}"
    elif 0.01 <= abs(amount) < 1:
        return f"${amount:.4f}"
    else:
        return f"${amount:,.{precision}f}"

def format_percentage(value):
    """Format percentage values"""
    if value is None:
        return "N/A"
    
    return f"{value:.2f}%"

def add_risk_color(risk_level):
    """Add color formatting to risk levels"""
    if risk_level == "Low":
        return f'<span style="color:#4bff4b">{risk_level}</span>'
    elif risk_level == "Medium":
        return f'<span style="color:#ffff4b">{risk_level}</span>'
    elif risk_level == "Medium-High":
        return f'<span style="color:#ffa64b">{risk_level}</span>'
    elif risk_level == "High":
        return f'<span style="color:#ff794b">{risk_level}</span>'
    elif risk_level == "Very High":
        return f'<span style="color:#ff4b4b">{risk_level}</span>'
    else:
        return f'<span style="color:#a0a0a0">{risk_level}</span>'

def add_trend_arrow(value):
    """Add trend arrows to numeric values"""
    if value is None:
        return "N/A"
    
    if value > 0:
        return f"{value:.2f}% ↑"
    elif value < 0:
        return f"{value:.2f}% ↓"
    else:
        return f"{value:.2f}% ↔"

def calculate_expected_return(historical_df, risk_free_rate=0.02):
    """
    Calculate expected return using historical data
    
    Args:
        historical_df: DataFrame with historical price data
        risk_free_rate: Risk-free rate (default 2%)
        
    Returns:
        Dictionary with expected return metrics
    """
    if historical_df.empty:
        return {
            'daily_returns_mean': 0,
            'annualized_return': 0,
            'risk_adjusted_return': 0
        }
    
    # Calculate daily returns
    returns = historical_df['price'].pct_change().dropna()
    
    # Mean daily return
    daily_mean = returns.mean()
    
    # Annualized return (assuming 365 trading days)
    annualized_return = (1 + daily_mean) ** 365 - 1
    
    # Annualized volatility
    annualized_vol = returns.std() * np.sqrt(365)
    
    # Risk-adjusted return (Sharpe ratio)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
    
    return {
        'daily_returns_mean': daily_mean,
        'annualized_return': annualized_return,
        'risk_adjusted_return': sharpe_ratio
    }

def create_tooltip(text, tooltip_text):
    """Create HTML tooltip for metrics"""
    return f'<span class="tooltip">{text}<span class="tooltiptext">{tooltip_text}</span></span>'

def get_time_periods():
    """Get time period options for historical data"""
    return {
        "7d": "7 Days",
        "30d": "30 Days",
        "90d": "90 Days",
        "180d": "180 Days",
        "365d": "1 Year",
        "max": "Max"
    }

def days_from_period(period):
    """Convert time period string to number of days"""
    if period == "7d":
        return 7
    elif period == "30d":
        return 30
    elif period == "90d":
        return 90
    elif period == "180d":
        return 180
    elif period == "365d":
        return 365
    else:  # "max"
        return 2000  # arbitrary large number

def generate_comparison_data(coin1_data, coin2_data):
    """
    Generate comparison data between two cryptocurrencies
    
    Args:
        coin1_data: Dictionary with data for first cryptocurrency
        coin2_data: Dictionary with data for second cryptocurrency
        
    Returns:
        DataFrame with comparison metrics
    """
    metrics = [
        "Current Price",
        "Market Cap",
        "24h Change",
        "7d Change",
        "30d Change",
        "Volume (24h)",
        "Volatility",
        "Risk Level"
    ]
    
    coin1_name = coin1_data.get("name", "Coin 1")
    coin2_name = coin2_data.get("name", "Coin 2")
    
    # Extract data
    coin1_values = [
        coin1_data.get("current_price"),
        coin1_data.get("market_cap"),
        coin1_data.get("price_change_percentage_24h"),
        coin1_data.get("price_change_percentage_7d_in_currency"),
        coin1_data.get("price_change_percentage_30d_in_currency"),
        coin1_data.get("total_volume"),
        coin1_data.get("volatility"),
        coin1_data.get("risk_level")
    ]
    
    coin2_values = [
        coin2_data.get("current_price"),
        coin2_data.get("market_cap"),
        coin2_data.get("price_change_percentage_24h"),
        coin2_data.get("price_change_percentage_7d_in_currency"),
        coin2_data.get("price_change_percentage_30d_in_currency"),
        coin2_data.get("total_volume"),
        coin2_data.get("volatility"),
        coin2_data.get("risk_level")
    ]
    
    # Create dataframe
    df = pd.DataFrame({
        "Metric": metrics,
        coin1_name: coin1_values,
        coin2_name: coin2_values
    })
    
    return df
