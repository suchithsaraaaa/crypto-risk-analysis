import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os

# API Configuration
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
CRYPTO_COMPARE_URL = "https://min-api.cryptocompare.com/data"
CRYPTO_COMPARE_API_KEY = os.getenv("CRYPTO_COMPARE_API_KEY", "")
ALTERNATIVE_ME_FEAR_GREED_URL = "https://api.alternative.me/fng/"

def get_top_coins(limit=100):
    """
    Get list of top cryptocurrencies by market cap
    """
    try:
        response = requests.get(
            f"{COINGECKO_BASE_URL}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": False
            },
            timeout=10
        )
        
        if response.status_code == 200:
            coins = response.json()
            return [{
                "id": coin["id"],
                "symbol": coin["symbol"].upper(),
                "name": coin["name"],
                "image": coin["image"],
                "current_price": coin["current_price"],
                "market_cap": coin["market_cap"],
                "market_cap_rank": coin["market_cap_rank"],
                "price_change_percentage_24h": coin["price_change_percentage_24h"]
            } for coin in coins]
        else:
            return []
    except Exception as e:
        print(f"Error fetching top coins: {e}")
        return []

def get_coin_details(coin_id):
    """
    Get detailed information about a specific cryptocurrency
    """
    try:
        response = requests.get(
            f"{COINGECKO_BASE_URL}/coins/{coin_id}",
            params={
                "localization": False,
                "tickers": False,
                "market_data": True,
                "community_data": True,
                "developer_data": True
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching coin details: {e}")
        return None

def get_historical_prices(coin_id, vs_currency="usd", days=30):
    """
    Get historical price data for a cryptocurrency
    """
    try:
        response = requests.get(
            f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart",
            params={
                "vs_currency": vs_currency,
                "days": days,
                "interval": "daily"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            prices_df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            volumes_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
            
            # Convert timestamp to datetime
            prices_df["date"] = pd.to_datetime(prices_df["timestamp"], unit="ms")
            prices_df = prices_df.drop("timestamp", axis=1)
            
            volumes_df["date"] = pd.to_datetime(volumes_df["timestamp"], unit="ms")
            volumes_df = volumes_df.drop("timestamp", axis=1)
            
            # Merge prices and volumes
            merged_df = pd.merge(prices_df, volumes_df, on="date")
            
            # Add additional columns for analysis
            merged_df["price_change"] = merged_df["price"].pct_change()
            merged_df["volatility"] = merged_df["price_change"].rolling(window=7).std()
            
            return merged_df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching historical prices: {e}")
        return pd.DataFrame()

def get_fear_greed_index():
    """
    Get the cryptocurrency fear and greed index
    """
    try:
        response = requests.get(ALTERNATIVE_ME_FEAR_GREED_URL, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "value": int(data["data"][0]["value"]),
                "value_classification": data["data"][0]["value_classification"],
                "timestamp": datetime.fromtimestamp(int(data["data"][0]["timestamp"])).strftime('%Y-%m-%d')
            }
        else:
            return {"value": 50, "value_classification": "Neutral", "timestamp": datetime.now().strftime('%Y-%m-%d')}
    except Exception as e:
        print(f"Error fetching fear and greed index: {e}")
        return {"value": 50, "value_classification": "Neutral", "timestamp": datetime.now().strftime('%Y-%m-%d')}

def get_crypto_news(coin="bitcoin", limit=10):
    """
    Get latest news for a cryptocurrency
    """
    if not CRYPTO_COMPARE_API_KEY:
        return []
        
    try:
        response = requests.get(
            f"{CRYPTO_COMPARE_URL}/v2/news/",
            params={
                "categories": coin,
                "excludeCategories": "Sponsored",
                "lang": "EN",
                "api_key": CRYPTO_COMPARE_API_KEY
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            news = data.get("Data", [])
            return [{
                "title": article["title"],
                "url": article["url"],
                "published_on": datetime.fromtimestamp(article["published_on"]).strftime('%Y-%m-%d %H:%M'),
                "source": article["source"],
                "body": article["body"][:150] + "..." if len(article["body"]) > 150 else article["body"]
            } for article in news[:limit]]
        else:
            return []
    except Exception as e:
        print(f"Error fetching crypto news: {e}")
        return []

def get_market_dominance():
    """
    Get market dominance of top cryptocurrencies
    """
    try:
        response = requests.get(
            f"{COINGECKO_BASE_URL}/global",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            market_cap_percentage = data["data"]["market_cap_percentage"]
            return {k.upper(): v for k, v in market_cap_percentage.items()}
        else:
            return {}
    except Exception as e:
        print(f"Error fetching market dominance: {e}")
        return {}

def get_exchange_rates():
    """
    Get crypto to fiat exchange rates
    """
    try:
        response = requests.get(
            f"{COINGECKO_BASE_URL}/exchange_rates",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            rates = data["rates"]
            
            # Get only fiat currencies
            fiat_rates = {k: v for k, v in rates.items() 
                         if v["type"] == "fiat" and k in ["usd", "eur", "gbp", "jpy", "cny"]}
            
            return fiat_rates
        else:
            return {}
    except Exception as e:
        print(f"Error fetching exchange rates: {e}")
        return {}
