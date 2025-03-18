import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go

# Import custom modules
from data_retrieval import (
    get_top_coins, get_coin_details, get_historical_prices,
    get_fear_greed_index, get_crypto_news, get_market_dominance,
    get_exchange_rates
)
from ml_models import (
    train_price_prediction_model, predict_future_prices,
    calculate_risk_metrics, analyze_sentiment
)
from visualizations import (
    plot_price_history, plot_price_prediction, plot_volatility,
    plot_risk_gauge, plot_animated_risk_meter, plot_feature_importance, 
    plot_market_dominance, plot_fear_greed_gauge, plot_correlation_heatmap, 
    plot_sentiment_gauge
)
from utils import (
    format_large_number, format_currency, format_percentage,
    add_risk_color, add_trend_arrow, calculate_expected_return,
    get_time_periods, days_from_period, get_risk_color_hex,
    get_risk_description, create_real_time_risk_monitor
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Crypto Analysis & Risk Management",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
with open("assets/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "selected_coin" not in st.session_state:
    st.session_state.selected_coin = "bitcoin"
if "time_period" not in st.session_state:
    st.session_state.time_period = "30d"
if "historical_data" not in st.session_state:
    st.session_state.historical_data = {}
if "coin_details" not in st.session_state:
    st.session_state.coin_details = {}
if "prediction_days" not in st.session_state:
    st.session_state.prediction_days = 7
if "risk_tolerance" not in st.session_state:
    st.session_state.risk_tolerance = "Medium"
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = datetime.now()

def load_header_images():
    """Load and display header images"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image("https://images.unsplash.com/photo-1639754390580-2e7437267698", caption="Crypto Market")
    with col2:
        st.image("https://images.unsplash.com/photo-1639389016105-2fb11199fb6b", caption="Trading Analysis")
    with col3:
        st.image("https://images.unsplash.com/photo-1640592276475-56a1c277a38f", caption="Risk Management")
    with col4:
        st.image("https://images.unsplash.com/photo-1639987402632-d7273e921454", caption="Market Predictions")

def load_dashboard_images():
    """Load and display dashboard indicator images"""
    col1, col2 = st.columns(2)

    with col1:
        st.image("https://images.unsplash.com/photo-1444653389962-8149286c578a", caption="Financial Analysis")
    with col2:
        st.image("https://images.unsplash.com/photo-1444653614773-995cb1ef9efa", caption="Market Indicators")

def sidebar():
    """Render the sidebar"""
    st.sidebar.title("Cryptocurrency Analysis")

    # Display sidebar image
    st.sidebar.image("https://images.unsplash.com/photo-1579623261984-41f9a81d4044", caption="Crypto Dashboard")

    # Fetch top coins for selection dropdown
    # Use cached data if available and less than 5 minutes old
    if "top_coins" not in st.session_state or (datetime.now() - st.session_state.last_update_time).seconds > 300:
        # Display a spinner in the sidebar
        spinner_text = st.sidebar.text("Loading cryptocurrencies...")
        top_coins = get_top_coins(limit=100)
        if top_coins:
            st.session_state.top_coins = top_coins
            st.session_state.last_update_time = datetime.now()
        # Remove the spinner text
        spinner_text.empty()

    # Coin selection dropdown
    if "top_coins" in st.session_state and st.session_state.top_coins:
        coin_options = {f"{coin['name']} ({coin['symbol']})": coin['id'] for coin in st.session_state.top_coins}

        # Get current index
        current_index = 0
        if st.session_state.selected_coin in list(coin_options.values()):
            current_index = list(coin_options.values()).index(st.session_state.selected_coin)

        selected_coin_name = st.sidebar.selectbox(
            "Select Cryptocurrency",
            options=list(coin_options.keys()),
            index=current_index,
            key="coin_selector"  # Add unique key to force update
        )

        # Update selected coin
        new_selected_coin = coin_options[selected_coin_name]
        if new_selected_coin != st.session_state.selected_coin:
            st.session_state.selected_coin = new_selected_coin
            # Clear cached data for the previous coin
            if new_selected_coin in st.session_state.historical_data:
                del st.session_state.historical_data[new_selected_coin]
            if new_selected_coin in st.session_state.coin_details:
                del st.session_state.coin_details[new_selected_coin]
    else:
        st.sidebar.error("Failed to load cryptocurrencies. Please try again later.")

    # Time period selection
    time_periods = get_time_periods()
    selected_period = st.sidebar.selectbox(
        "Select Time Period",
        options=list(time_periods.keys()),
        format_func=lambda x: time_periods[x],
        index=list(time_periods.keys()).index(st.session_state.time_period)
    )
    st.session_state.time_period = selected_period

    # Prediction days slider
    st.session_state.prediction_days = st.sidebar.slider(
        "Prediction Days",
        min_value=1,
        max_value=30,
        value=st.session_state.prediction_days
    )

    # Risk tolerance selection
    risk_tolerance_options = ["Low", "Medium", "High"]
    st.session_state.risk_tolerance = st.sidebar.radio(
        "Risk Tolerance",
        options=risk_tolerance_options,
        index=risk_tolerance_options.index(st.session_state.risk_tolerance)
    )

    # Refresh button
    if st.sidebar.button("Refresh Data"):
        st.session_state.last_update_time = datetime.now() - timedelta(minutes=10)  # Force refresh
        st.rerun()

    # Display last update time
    st.sidebar.caption(f"Last updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This dashboard provides cryptocurrency analysis and risk management insights using machine learning models. "
        "Data is sourced from CoinGecko and CryptoCompare APIs."
    )

def fetch_data():
    """Fetch all required data for the selected cryptocurrency"""
    selected_coin = st.session_state.selected_coin
    time_period = st.session_state.time_period
    days = days_from_period(time_period)

    # Fetch historical price data
    if selected_coin not in st.session_state.historical_data or time_period not in st.session_state.historical_data.get(selected_coin, {}):
        with st.spinner(f"Loading historical data for {selected_coin}..."):
            historical_df = get_historical_prices(selected_coin, days=days)

            if not historical_df.empty:
                if selected_coin not in st.session_state.historical_data:
                    st.session_state.historical_data[selected_coin] = {}

                st.session_state.historical_data[selected_coin][time_period] = historical_df

    # Fetch coin details
    if selected_coin not in st.session_state.coin_details:
        with st.spinner(f"Loading details for {selected_coin}..."):
            coin_details = get_coin_details(selected_coin)

            if coin_details:
                st.session_state.coin_details[selected_coin] = coin_details

    # Fetch market data (only update if it's been more than 5 minutes)
    if "market_data" not in st.session_state or (datetime.now() - st.session_state.last_update_time).seconds > 300:
        with st.spinner("Loading market data..."):
            # Market dominance
            market_dominance = get_market_dominance()
            if market_dominance:
                st.session_state.market_dominance = market_dominance

            # Fear and greed index
            fear_greed = get_fear_greed_index()
            if fear_greed:
                st.session_state.fear_greed = fear_greed

            # Exchange rates
            exchange_rates = get_exchange_rates()
            if exchange_rates:
                st.session_state.exchange_rates = exchange_rates

            # News for selected coin
            news = get_crypto_news(coin=selected_coin)
            if news:
                st.session_state.news = news

            # Market sentiment
            if news:
                sentiment_score, sentiment_label = analyze_sentiment(news)
                st.session_state.sentiment = {
                    "score": sentiment_score,
                    "label": sentiment_label
                }

            st.session_state.market_data = True

def main_dashboard():
    """Render the main dashboard"""
    # Coin data
    selected_coin = st.session_state.selected_coin
    time_period = st.session_state.time_period

    if selected_coin in st.session_state.historical_data and time_period in st.session_state.historical_data[selected_coin]:
        historical_df = st.session_state.historical_data[selected_coin][time_period]
    else:
        st.error(f"No historical data available for {selected_coin}")
        return

    if selected_coin in st.session_state.coin_details:
        coin_details = st.session_state.coin_details[selected_coin]
    else:
        st.error(f"No details available for {selected_coin}")
        return

    # Main title with coin image
    st.title(f"Analysis Dashboard: {coin_details['name']} ({coin_details['symbol'].upper()})")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Price",
            format_currency(coin_details['market_data']['current_price']['usd']),
            add_trend_arrow(coin_details['market_data']['price_change_percentage_24h'])
        )

    with col2:
        st.metric(
            "Market Cap",
            format_large_number(coin_details['market_data']['market_cap']['usd']),
            add_trend_arrow(coin_details['market_data']['market_cap_change_percentage_24h'])
        )

    with col3:
        st.metric(
            "24h Volume",
            format_large_number(coin_details['market_data']['total_volume']['usd'])
        )

    with col4:
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(historical_df)

        # Display risk level with color
        st.markdown(
            f"""
            <div style='border:1px solid #555; border-radius:5px; padding:10px; text-align:center;'>
                <h4>Risk Level</h4>
                <h2>{add_risk_color(risk_metrics['risk_level'])}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis", "Risk Assessment", "Market Context", "News & Sentiment"])

    with tab1:
        st.subheader("Historical Price & Volume")
        plot_price_history(historical_df, coin_details['name'], days=days_from_period(time_period))

        st.subheader("Price Prediction")

        # Train prediction model
        model, rmse, mae, r2, feature_importance = train_price_prediction_model(historical_df)

        if model is not None:
            # Generate predictions
            prediction_df = predict_future_prices(
                model, historical_df, 
                days_to_predict=st.session_state.prediction_days
            )

            # Plot predictions
            plot_price_prediction(historical_df, prediction_df, coin_details['name'])

            # Display model metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Root Mean Squared Error", f"{rmse:.4f}")
            with metric_col2:
                st.metric("Mean Absolute Error", f"{mae:.4f}")
            with metric_col3:
                st.metric("R² Score", f"{r2:.4f}")

            # Show feature importance
            st.subheader("Price Prediction Factors")
            plot_feature_importance(feature_importance)
        else:
            st.error("Unable to train prediction model with available data")

    with tab2:
        st.subheader("Volatility Analysis")
        plot_volatility(historical_df, coin_details['name'])

        # Risk metrics
        st.subheader("Risk Assessment")

        # Create tabs for different risk visualization methods
        risk_tab1, risk_tab2, risk_tab3 = st.tabs(["Animated Risk Meter", "Real-Time Monitor", "Traditional Gauge"])

        with risk_tab1:
            # Display animated risk meter
            plot_animated_risk_meter(risk_metrics)

        with risk_tab2:
            # Create a placeholder for the real-time risk monitor
            risk_monitor_placeholder = st.empty()

            # Initialize the real-time risk monitor
            create_real_time_risk_monitor(risk_metrics, risk_monitor_placeholder)

            # Add simulation controls
            st.markdown("### Real-Time Monitoring Simulation")

            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                if st.button("Simulate Risk Alert"):
                    # Create a temporary copy of risk metrics with higher risk
                    temp_risk = risk_metrics.copy()
                    temp_risk["risk_level"] = "Very High"
                    temp_risk["volatility"] = risk_metrics["volatility"] * 1.5
                    temp_risk["max_drawdown"] = risk_metrics["max_drawdown"] * 1.2
                    create_real_time_risk_monitor(temp_risk, risk_monitor_placeholder)
                    time.sleep(3)
                    create_real_time_risk_monitor(risk_metrics, risk_monitor_placeholder)

            with sim_col2:
                if st.button("Refresh Monitor"):
                    # Update with current time
                    create_real_time_risk_monitor(risk_metrics, risk_monitor_placeholder)

        with risk_tab3:
            # Display traditional risk gauge
            plot_risk_gauge(risk_metrics)

        # Risk metrics details
        risk_col1, risk_col2, risk_col3 = st.columns(3)

        with risk_col1:
            st.markdown(
                f"""
                ### Volatility
                #### {risk_metrics['volatility']:.4f}
                *Annualized standard deviation of returns*
                """
            )

        with risk_col2:
            st.markdown(
                f"""
                ### Sharpe Ratio
                #### {risk_metrics['sharpe_ratio']:.4f}
                *Risk-adjusted return measure*
                """
            )

        with risk_col3:
            st.markdown(
                f"""
                ### Maximum Drawdown
                #### {risk_metrics['max_drawdown'] * 100:.2f}%
                *Largest peak-to-trough decline*
                """
            )

        # Expected returns
        st.subheader("Expected Returns")
        expected_returns = calculate_expected_return(historical_df)

        returns_col1, returns_col2 = st.columns(2)

        with returns_col1:
            st.markdown(
                f"""
                ### Annualized Return
                #### {expected_returns['annualized_return'] * 100:.2f}%
                *Expected yearly return based on historical data*
                """
            )

        with returns_col2:
            st.markdown(
                f"""
                ### Risk-Adjusted Return
                #### {expected_returns['risk_adjusted_return']:.4f}
                *Return adjusted for risk taken*
                """
            )

        # Risk tolerance recommendation
        st.subheader("Risk Tolerance Assessment")

        risk_match = {
            "Low": ["Low"],
            "Medium": ["Low", "Medium"],
            "High": ["Low", "Medium", "Medium-High", "High"]
        }

        user_risk = st.session_state.risk_tolerance
        coin_risk = risk_metrics['risk_level']

        if coin_risk in risk_match[user_risk]:
            st.success(f"This cryptocurrency's risk level ({coin_risk}) aligns with your risk tolerance ({user_risk}).")
        else:
            st.warning(f"This cryptocurrency's risk level ({coin_risk}) may not align with your risk tolerance ({user_risk}).")

            # Offer recommendations
            if user_risk == "Low" and coin_risk not in risk_match[user_risk]:
                st.info("Consider cryptocurrencies with lower volatility or stablecoins for a better match to your risk tolerance.")
            elif user_risk == "Medium" and coin_risk not in risk_match[user_risk]:
                st.info("Consider top market cap cryptocurrencies with established histories for a better match to your risk tolerance.")

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Market Dominance")
            if "market_dominance" in st.session_state:
                plot_market_dominance(st.session_state.market_dominance)
            else:
                st.info("Market dominance data not available")

        with col2:
            st.subheader("Fear & Greed Index")
            if "fear_greed" in st.session_state:
                plot_fear_greed_gauge(st.session_state.fear_greed)
            else:
                st.info("Fear and greed index not available")

        st.subheader("Market Correlation")

        # Collect historical data for top coins to calculate correlation
        if "top_coins" in st.session_state and len(st.session_state.top_coins) > 1:
            historical_data_dict = {}

            # Use existing historical data from session state
            for coin in st.session_state.historical_data:
                if st.session_state.time_period in st.session_state.historical_data[coin]:
                    historical_data_dict[coin] = st.session_state.historical_data[coin][st.session_state.time_period]

            # Fetch data for top 5 coins if we don't have enough
            if len(historical_data_dict) < 5:
                with st.spinner("Loading correlation data..."):
                    for coin in st.session_state.top_coins[:5]:
                        coin_id = coin['id']
                        if coin_id not in historical_data_dict:
                            df = get_historical_prices(coin_id, days=days_from_period(st.session_state.time_period))
                            if not df.empty:
                                # Store in session state for future use
                                if coin_id not in st.session_state.historical_data:
                                    st.session_state.historical_data[coin_id] = {}

                                st.session_state.historical_data[coin_id][st.session_state.time_period] = df
                                historical_data_dict[coin_id] = df

            if len(historical_data_dict) > 1:
                plot_correlation_heatmap(historical_data_dict, selected_coin)
            else:
                st.info("Insufficient data to generate correlation analysis")
        else:
            st.info("Correlation data not available")

        # Display exchange rates
        st.subheader("Exchange Rates")
        if "exchange_rates" in st.session_state and st.session_state.exchange_rates:
            rates_df = pd.DataFrame([
                {
                    "Currency": rate.upper(),
                    "Name": st.session_state.exchange_rates[rate]['name'],
                    "Value": st.session_state.exchange_rates[rate]['value'],
                    "Type": st.session_state.exchange_rates[rate]['type']
                }
                for rate in st.session_state.exchange_rates
            ])

            st.dataframe(rates_df, hide_index=True)
        else:
            st.info("Exchange rate data not available")

    with tab4:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Latest News")
            if "news" in st.session_state and st.session_state.news:
                for article in st.session_state.news[:5]:
                    st.markdown(
                        f"""
                        ### [{article['title']}]({article['url']})
                        **Source:** {article['source']} | **Published:** {article['published_on']}

                        {article['body']}

                        ---
                        """
                    )
            else:
                st.info("No news available")

        with col2:
            st.subheader("Market Sentiment")
            if "sentiment" in st.session_state:
                plot_sentiment_gauge(
                    st.session_state.sentiment["score"],
                    st.session_state.sentiment["label"]
                )

                st.markdown(
                    f"""
                    ### Sentiment Analysis
                    Based on recent news articles about {coin_details['name']}, the market sentiment appears to be **{st.session_state.sentiment["label"]}**.

                    *Sentiment is analyzed from news headlines and content using natural language processing techniques.*
                    """
                )
            else:
                st.info("Sentiment data not available")

            # Display cryptocurrency logo
            if "image" in coin_details and coin_details['image']['large']:
                st.image(coin_details['image']['large'], width=150)

    # Additional information
    with st.expander("Cryptocurrency Details"):
        if "description" in coin_details and "en" in coin_details["description"]:
            st.markdown(coin_details["description"]["en"])

        detail_col1, detail_col2, detail_col3 = st.columns(3)

        with detail_col1:
            st.markdown(
                f"""
                ### Market Information
                - **Genesis Date:** {coin_details.get('genesis_date', 'N/A')}
                - **Market Cap Rank:** #{coin_details.get('market_cap_rank', 'N/A')}
                - **CoinGecko Rank:** #{coin_details.get('coingecko_rank', 'N/A')}
                - **CoinGecko Score:** {coin_details.get('coingecko_score', 'N/A')}
                """
            )

        with detail_col2:
            # Check if developer data is available
            developer_data = coin_details.get('developer_data', {})

            if not developer_data:
                st.markdown(
                    """
                    ### Developer Stats
                    - **GitHub Stars:** N/A
                    - **GitHub Forks:** N/A
                    - **GitHub Subscribers:** N/A
                    - **Commits (4 Weeks):** N/A

                    *Developer data not available*
                    """
                )
            else:
                st.markdown(
                    f"""
                    ### Developer Stats
                    - **GitHub Stars:** {developer_data.get('stars', 'N/A')}
                    - **GitHub Forks:** {developer_data.get('forks', 'N/A')}
                    - **GitHub Subscribers:** {developer_data.get('subscribers', 'N/A')}
                    - **Commits (4 Weeks):** {developer_data.get('commit_count_4_weeks', 'N/A')}
                    """
                )

        with detail_col3:
            # Check if community data is available
            community_data = coin_details.get('community_data', {})

            if not community_data:
                st.markdown(
                    """
                    ### Community Stats
                    - **Twitter Followers:** N/A
                    - **Reddit Subscribers:** N/A
                    - **Telegram Users:** N/A
                    - **Facebook Likes:** N/A

                    *Community data not available*
                    """
                )
            else:
                st.markdown(
                    f"""
                    ### Community Stats
                    - **Twitter Followers:** {format_large_number(community_data.get('twitter_followers', 'N/A'))}
                    - **Reddit Subscribers:** {format_large_number(community_data.get('reddit_subscribers', 'N/A'))}
                    - **Telegram Users:** {format_large_number(community_data.get('telegram_channel_user_count', 'N/A'))}
                    - **Facebook Likes:** {format_large_number(community_data.get('facebook_likes', 'N/A'))}
                    """
                )

def main():
    """Main application entry point"""
    # Display app header
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Cryptocurrency Prediction & Risk Management</h1>
            <p>Advanced analysis, predictions, and risk assessment for cryptocurrency investors</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display sidebar
    sidebar()

    # Fetch required data
    fetch_data()

    # Display main dashboard
    main_dashboard()

if __name__ == "__main__":
    main()
