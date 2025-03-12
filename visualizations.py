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

        # Handle integer timestamps (seconds or milliseconds)
        if isinstance(first_prediction_date, (int, float)):
            first_prediction_date = pd.to_datetime(first_prediction_date, unit='s', errors='coerce')
        else:
            first_prediction_date = pd.to_datetime(first_prediction_date, errors='coerce')

        if pd.isna(first_prediction_date):
            raise ValueError("Invalid date format in prediction data.")

        fig.add_vline(
            x=first_prediction_date,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="Prediction Start",
            annotation_position="top right"
        )
    except Exception as e:
        st.warning(f"Could not add prediction separator: {e}")

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
    
def plot_animated_risk_meter(risk_metrics):
    """
    Plot animated risk meter with real-time color-changing indicators
    """
    if not risk_metrics:
        st.error("No risk metrics available")
        return
    
    # Extract risk metrics
    volatility = risk_metrics.get("volatility", 0)
    sharpe_ratio = risk_metrics.get("sharpe_ratio", 0)
    max_drawdown = risk_metrics.get("max_drawdown", 0)
    risk_level = risk_metrics.get("risk_level", "Unknown")
    
    # Create container for animation
    risk_container = st.container()
    
    # Map risk level to a numeric scale (0-100)
    risk_value_map = {
        "Low": 20,
        "Medium": 40,
        "Medium-High": 60,
        "High": 80,
        "Very High": 100,
        "Unknown": 50
    }
    
    risk_value = risk_value_map.get(risk_level, 50)
    
    # Risk level color map
    risk_colors = {
        "Low": "#00FA00",  # Green
        "Medium": "#96FA00",  # Yellow-Green
        "Medium-High": "#FAFA00",  # Yellow
        "High": "#FA9600",  # Orange
        "Very High": "#FA0000",  # Red
        "Unknown": "#AAAAAA"  # Gray
    }
    
    risk_color = risk_colors.get(risk_level, "#AAAAAA")
    
    # Convert the hex color to RGB for opacity adjustments
    risk_color_rgb = tuple(int(risk_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    with risk_container:
        st.markdown(f"""
        <style>
        @keyframes pulse {{
            0% {{ opacity: 0.6; }}
            50% {{ opacity: 1.0; }}
            100% {{ opacity: 0.6; }}
        }}
        .risk-meter-container {{
            background-color: rgba(30, 30, 30, 0.7);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
            border: 1px solid rgba(100, 100, 100, 0.5);
        }}
        .risk-meter-title {{
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
            color: white;
        }}
        .risk-meter {{
            height: 30px;
            width: 100%;
            background-color: rgba(50, 50, 50, 0.3);
            border-radius: 15px;
            overflow: hidden;
            position: relative;
            margin-bottom: 20px;
        }}
        .risk-meter-fill {{
            height: 100%;
            width: {risk_value}%;
            background: linear-gradient(90deg, rgba({risk_color_rgb[0]}, {risk_color_rgb[1]}, {risk_color_rgb[2]}, 0.7) 0%, 
                                              rgba({risk_color_rgb[0]}, {risk_color_rgb[1]}, {risk_color_rgb[2]}, 1) 100%);
            border-radius: 15px;
            transition: width 1s ease-in-out;
            animation: pulse 2s infinite;
        }}
        .risk-level-indicator {{
            position: absolute;
            top: -25px;
            transform: translateX(-50%);
            color: {risk_color};
            font-weight: bold;
            text-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
            animation: pulse 2s infinite;
        }}
        .risk-level-text {{
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 10px;
            color: {risk_color};
            animation: pulse 2s infinite;
        }}
        .risk-metrics-container {{
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
        }}
        .risk-metric-box {{
            background-color: rgba(50, 50, 50, 0.5);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            width: 30%;
            border: 1px solid rgba({risk_color_rgb[0]}, {risk_color_rgb[1]}, {risk_color_rgb[2]}, 0.5);
            box-shadow: 0 0 10px rgba({risk_color_rgb[0]}, {risk_color_rgb[1]}, {risk_color_rgb[2]}, 0.2);
            transition: all 0.3s ease;
        }}
        .risk-metric-box:hover {{
            transform: translateY(-5px);
            box-shadow: 0 0 15px rgba({risk_color_rgb[0]}, {risk_color_rgb[1]}, {risk_color_rgb[2]}, 0.4);
        }}
        .risk-metric-title {{
            font-size: 16px;
            color: #CCCCCC;
            margin-bottom: 8px;
        }}
        .risk-metric-value {{
            font-size: 22px;
            font-weight: bold;
            color: white;
        }}
        </style>
        
        <div class="risk-meter-container">
            <div class="risk-meter-title">Real-time Risk Assessment</div>
            <div class="risk-meter">
                <div class="risk-meter-fill"></div>
                <div class="risk-level-indicator" style="left: {risk_value}%">{risk_level}</div>
            </div>
            <div class="risk-level-text">Risk Level: {risk_level}</div>
            
            <div class="risk-metrics-container">
                <div class="risk-metric-box">
                    <div class="risk-metric-title">Volatility</div>
                    <div class="risk-metric-value">{volatility:.4f}</div>
                </div>
                <div class="risk-metric-box">
                    <div class="risk-metric-title">Sharpe Ratio</div>
                    <div class="risk-metric-value">{sharpe_ratio:.4f}</div>
                </div>
                <div class="risk-metric-box">
                    <div class="risk-metric-title">Max Drawdown</div>
                    <div class="risk-metric-value">{max_drawdown * 100:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add real-time updating effect using Streamlit's built-in elements
        with st.expander("Risk Interpretation", expanded=False):
            # Import the utility function for risk descriptions
            from utils import get_risk_description
            
            # Get the risk description
            risk_description = get_risk_description(risk_level)
            
            st.markdown(f"""
            - **Current Risk Level:** {risk_level}
            - **Volatility:** {volatility:.4f} - {'Very High' if volatility > 0.8 else 'High' if volatility > 0.6 else 'Medium' if volatility > 0.4 else 'Low'}
            - **Sharpe Ratio:** {sharpe_ratio:.4f} - {'Excellent' if sharpe_ratio > 1.5 else 'Good' if sharpe_ratio > 1 else 'Average' if sharpe_ratio > 0.5 else 'Poor'}
            - **Maximum Drawdown:** {max_drawdown * 100:.2f}% - {'Severe' if max_drawdown < -0.5 else 'Significant' if max_drawdown < -0.3 else 'Moderate' if max_drawdown < -0.15 else 'Minimal'}
            
            **What This Means:**
            
            {risk_description}
            """,unsafe_allow_html=True)
    
    return True

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
