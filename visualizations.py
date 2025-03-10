import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def plot_price_history(df, coin_name, days=30):
    """
    Plot interactive price history chart
    """
    if df.empty:
        st.error(f"No historical data available for {coin_name}")
        return
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['price'],
            name="Price (USD)",
            line=dict(color="#3366FF", width=2)
        ),
        secondary_y=False
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name="Volume",
            marker_color="rgba(55, 83, 109, 0.3)",
            opacity=0.5
        ),
        secondary_y=True
    )
    
    # Add layout details
    fig.update_layout(
        title=f"{coin_name} Price History (Last {days} Days)",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_price_prediction(historical_df, prediction_df, coin_name):
    """
    Plot historical prices with future predictions
    """
    if historical_df.empty or prediction_df.empty:
        st.error(f"Insufficient data to generate predictions for {coin_name}")
        return
    
    # Create figure
    fig = go.Figure()
    
    # Add historical price line
    fig.add_trace(
        go.Scatter(
            x=historical_df['date'],
            y=historical_df['price'],
            name="Historical Price",
            line=dict(color="#3366FF", width=2)
        )
    )
    
    # Add prediction line
    fig.add_trace(
        go.Scatter(
            x=prediction_df['date'],
            y=prediction_df['predicted_price'],
            name="Predicted Price",
            line=dict(color="#FF6B3D", width=2, dash='dash')
        )
    )
    
    # Add confidence interval (simplified)
    volatility = historical_df['price'].pct_change().std()
    upper_bound = prediction_df['predicted_price'] * (1 + volatility * 1.96)
    lower_bound = prediction_df['predicted_price'] * (1 - volatility * 1.96)
    
    fig.add_trace(
        go.Scatter(
            x=prediction_df['date'],
            y=upper_bound,
            name="Upper Bound (95% CI)",
            line=dict(color="#FF6B3D", width=0),
            fill=None,
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=prediction_df['date'],
            y=lower_bound,
            name="Lower Bound (95% CI)",
            line=dict(color="#FF6B3D", width=0),
            fill='tonexty',
            fillcolor='rgba(255, 107, 61, 0.2)',
            showlegend=False
        )
    )
    
    # Add layout details
    fig.update_layout(
        title=f"{coin_name} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )
    
    # Add vertical line to separate historical data from predictions
    try:
        first_prediction_date = prediction_df['date'].iloc[0]
        
        # Convert timestamp to string to avoid arithmetic issues
        first_prediction_date_str = pd.to_datetime(first_prediction_date).strftime('%Y-%m-%d')
        
        fig.add_vline(
            x=first_prediction_date_str,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="Prediction Start",
            annotation_position="top right"
        )
    except Exception as e:
        st.warning(f"Could not add prediction separator: {e}")
        # Continue without the vertical line
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_volatility(df, coin_name, window=7):
    """
    Plot volatility chart
    """
    if df.empty:
        st.error(f"No volatility data available for {coin_name}")
        return
    
    # Calculate rolling volatility if not already in dataframe
    if 'volatility' not in df.columns:
        df['volatility'] = df['price'].pct_change().rolling(window=window).std() * np.sqrt(window)
    
    # Create figure
    fig = go.Figure()
    
    # Add volatility line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['volatility'],
            name="Volatility (7-day)",
            line=dict(color="#FF6B3D", width=2)
        )
    )
    
    # Add layout details
    fig.update_layout(
        title=f"{coin_name} Price Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility",
        template="plotly_dark",
        height=400,
        hovermode="x unified"
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_risk_gauge(risk_metrics):
    """
    Plot risk gauge chart
    """
    if not risk_metrics:
        st.error("No risk metrics available")
        return
    
    risk_level = risk_metrics["risk_level"]
    
    # Map risk level to a value between 0 and 1
    risk_values = {
        "Low": 0.2,
        "Medium": 0.4,
        "Medium-High": 0.6,
        "High": 0.8,
        "Very High": 1.0,
        "Unknown": 0.5
    }
    
    risk_value = risk_values.get(risk_level, 0.5)
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Level", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(50, 50, 50, 0.1)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(0, 250, 0, 0.4)'},
                {'range': [20, 40], 'color': 'rgba(150, 250, 0, 0.4)'},
                {'range': [40, 60], 'color': 'rgba(250, 250, 0, 0.4)'},
                {'range': [60, 80], 'color': 'rgba(250, 150, 0, 0.4)'},
                {'range': [80, 100], 'color': 'rgba(250, 0, 0, 0.4)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_value * 100
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(feature_importance, top_n=10):
    """
    Plot feature importance from prediction model
    """
    if feature_importance is None or feature_importance.empty:
        st.error("No feature importance data available")
        return
    
    # Get top N features
    top_features = feature_importance.head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top Features for Price Prediction',
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_market_dominance(dominance_data):
    """
    Plot market dominance pie chart
    """
    if not dominance_data:
        st.error("No market dominance data available")
        return
    
    # Create dataframe from dominance data
    df = pd.DataFrame({
        'Cryptocurrency': list(dominance_data.keys()),
        'Dominance (%)': list(dominance_data.values())
    })
    
    # Sort by dominance
    df = df.sort_values('Dominance (%)', ascending=False)
    
    # Keep top 5, group the rest as "Others"
    if len(df) > 5:
        top_5 = df.head(5)
        others = pd.DataFrame({
            'Cryptocurrency': ['Others'],
            'Dominance (%)': [df.iloc[5:]['Dominance (%)'].sum()]
        })
        df = pd.concat([top_5, others])
    
    # Create pie chart
    fig = px.pie(
        df,
        values='Dominance (%)',
        names='Cryptocurrency',
        title='Market Dominance by Cryptocurrency',
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#000000', width=1))
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_fear_greed_gauge(fear_greed_data):
    """
    Plot fear and greed index gauge chart
    """
    if not fear_greed_data:
        st.error("No fear and greed data available")
        return
    
    value = fear_greed_data.get("value", 50)
    classification = fear_greed_data.get("value_classification", "Neutral")
    timestamp = fear_greed_data.get("timestamp", datetime.now().strftime('%Y-%m-%d'))
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Fear & Greed Index: {classification}<br><sub>{timestamp}</sub>", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(50, 50, 50, 0.1)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(255, 0, 0, 0.4)'},      # Extreme Fear
                {'range': [25, 45], 'color': 'rgba(255, 165, 0, 0.4)'},   # Fear
                {'range': [45, 55], 'color': 'rgba(255, 255, 0, 0.4)'},   # Neutral
                {'range': [55, 75], 'color': 'rgba(173, 255, 47, 0.4)'},  # Greed
                {'range': [75, 100], 'color': 'rgba(0, 128, 0, 0.4)'}     # Extreme Greed
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=70, b=20)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(historical_data_dict, selected_coin_id):
    """
    Plot correlation heatmap between cryptocurrencies
    
    Args:
        historical_data_dict: Dictionary of historical data for different cryptocurrencies
        selected_coin_id: ID of the currently selected cryptocurrency
    """
    if not historical_data_dict or len(historical_data_dict) < 2:
        st.error("Insufficient data to generate correlation heatmap")
        return
    
    # Create price dataframe for correlation calculation
    price_df = pd.DataFrame()
    
    for coin_id, df in historical_data_dict.items():
        if not df.empty:
            price_df[coin_id] = df.set_index('date')['price']
    
    # Resample to align all time series
    price_df = price_df.resample('D').last().dropna()
    
    # Calculate correlation matrix
    corr_matrix = price_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Price Correlation Between Cryptocurrencies',
        labels=dict(color="Correlation")
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_gauge(sentiment_score, sentiment_label):
    """
    Plot sentiment gauge chart
    """
    # Normalize sentiment score to 0-100 scale
    normalized_score = (sentiment_score + 1) * 50
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=normalized_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"News Sentiment: {sentiment_label}", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(50, 50, 50, 0.1)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33.3], 'color': 'rgba(255, 0, 0, 0.4)'},     # Negative
                {'range': [33.3, 66.6], 'color': 'rgba(255, 255, 0, 0.4)'}, # Neutral
                {'range': [66.6, 100], 'color': 'rgba(0, 255, 0, 0.4)'}     # Positive
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': normalized_score
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=70, b=20)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
