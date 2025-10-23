import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Use Yahoo Finance data api
from yf_stock_data import (
    get_stock_quote,
    get_stock_intraday_data,
    get_stock_daily_data,
    get_stock_weekly_data,
    search_stocks,
    get_company_name
)

# Import Indian stock market data
from indian_stock_data import (
    search_indian_stocks,
    get_indian_stock_quote,
    get_indian_stock_intraday_data,
    get_indian_stock_daily_data,
    get_indian_stock_weekly_data,
    get_popular_indian_stocks,
    get_indian_sectors
)
from technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_pe_ratio,
    calculate_volatility
)

# Import Indian technical indicators
from indian_technical_indicators import (
    calculate_indian_rsi,
    calculate_indian_macd,
    calculate_indian_bollinger_bands,
    calculate_indian_volume_profile,
    calculate_indian_momentum_indicators,
    calculate_indian_volatility_indicators,
    get_indian_stock_analysis,
    get_indian_stock_summary
)
from sentiment_analysis import get_stock_sentiment
from prediction_model import predict_trend
from utils import format_large_number, calculate_signal

# Page configuration
st.set_page_config(page_title="MoneyMentor-AI", page_icon="💹", layout="wide")

# Initialize session state
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'interval' not in st.session_state:
    st.session_state.interval = 'daily'
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = 'candlestick'
if 'show_indicators' not in st.session_state:
    st.session_state.show_indicators = {
        'sma': False,
        'ema': False,
        'bollinger': False
    }
if 'active_watchlist' not in st.session_state:
    st.session_state.active_watchlist = None
if 'view' not in st.session_state:
    st.session_state.view = 'dashboard'  # Can be 'dashboard' or 'watchlist'
if 'market' not in st.session_state:
    st.session_state.market = 'global'  # Can be 'global' or 'india'

# Import watchlist UI components
from watchlist_ui import render_watchlist_sidebar, render_watchlist_main


# App title
st.title("MoneyMentor-AI: Stock Analysis Platform")
st.markdown("Real-time market analysis with AI-powered insights for active traders and investors")

# Navigation
current_tab = st.radio("Navigation", ["Dashboard", "Watchlists"], horizontal=True, label_visibility="collapsed")

# Set the view based on selected tab
if current_tab == "Dashboard":
    st.session_state.view = 'dashboard'
else:  # Watchlists
    st.session_state.view = 'watchlist'

# Sidebar - Search & Filter and Watchlists
with st.sidebar:
    if st.session_state.view == 'dashboard':
        st.header("Stock Search & Filters")
        
        # Market selection
        st.subheader("Market Selection")
        market_choice = st.radio(
            "Choose Market:",
            options=["Global Markets", "Indian Markets"],
            index=0 if st.session_state.market == 'global' else 1,
            key="market_radio"
        )
        
        # Update session state based on selection
        if market_choice == "Global Markets":
            st.session_state.market = 'global'
        else:
            st.session_state.market = 'india'
        
        # Live search with suggestions
        search_query = st.text_input("Search by company name or ticker:", key="search_query")
        suggestions = []
        
        if search_query and len(search_query) >= 2:
            with st.spinner("Searching..."):
                if st.session_state.market == 'india':
                    suggestions = search_indian_stocks(search_query)[:10]
                else:
                    suggestions = search_stocks(search_query)[:10]
                st.session_state.search_results = suggestions
        
        if suggestions:
            st.caption("Suggestions")
            # Show as selectbox for easy selection
            choice = st.selectbox(
                "Select from suggestions:",
                options=[f"{s['symbol']} - {s['name']}" for s in suggestions],
                key="suggest_select",
            )
            if choice:
                sym = choice.split(" - ")[0]
                st.session_state.selected_stock = sym
        
        # Display last results (if any) for manual selection
        if 'search_results' in st.session_state and st.session_state.search_results:
            st.subheader("Results")
            selected_stock = st.selectbox(
                "Select a stock:",
                options=[(stock['symbol'], stock['name']) for stock in st.session_state.search_results],
                format_func=lambda x: f"{x[0]} - {x[1]}",
                key="results_select"
            )
            if selected_stock:
                st.session_state.selected_stock = selected_stock[0]
        
        # Show popular stocks based on market
        if st.session_state.market == 'india':
            st.subheader("Popular Indian Stocks")
            popular_stocks = get_popular_indian_stocks()
            
            # Create columns for popular stocks
            col1, col2 = st.columns(2)
            for i, stock in enumerate(popular_stocks[:10]):
                with col1 if i % 2 == 0 else col2:
                    if st.button(f"{stock['symbol']}", key=f"popular_{stock['symbol']}", help=stock['name']):
                        st.session_state.selected_stock = stock['symbol']
                        st.rerun()
        
        # Filters removed per request

    # Render watchlist sidebar for both views
    render_watchlist_sidebar()

# Main content area
if st.session_state.view == 'watchlist':
    # Render watchlist content
    render_watchlist_main()
elif st.session_state.selected_stock:  # Dashboard view with selected stock
    # Get stock data based on market selection
    with st.spinner(f"Loading data for {st.session_state.selected_stock}..."):
        if st.session_state.market == 'india':
            quote = get_indian_stock_quote(st.session_state.selected_stock)
        else:
            quote = get_stock_quote(st.session_state.selected_stock)
        
        # Handle case when API fails to return data
        if not quote:
            st.error(f"Unable to fetch data for {st.session_state.selected_stock}. Please try another stock.")
            st.session_state.selected_stock = None
        else:
            # Layout with two columns for current stock information
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.header(f"{quote['name']} ({quote['symbol']})")
                price_change = quote['price_change']
                price_change_percent = quote['price_change_percent']
                
                price_color = "green" if price_change >= 0 else "red"
                st.markdown(f"""
                <div style='display: flex; align-items: baseline;'>
                    <h2 style='margin: 0;'>${quote['price']:.2f}</h2>
                    <h3 style='margin: 0 0 0 10px; color: {price_color};'>
                        {price_change:.2f} ({price_change_percent:.2f}%)
                    </h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Market Info")
                st.markdown(f"**Market Cap:** ${format_large_number(quote['market_cap'])}")
                st.markdown(f"**Volume:** {format_large_number(quote['volume'])}")
                st.markdown(f"**P/E Ratio:** {quote.get('pe', 'N/A')}")
                st.markdown(f"**52-Week Range:** ${quote.get('52w_low', 0):.2f} - ${quote.get('52w_high', 0):.2f}")
            
            # Chart section
            st.subheader("Price Chart")
            chart_col, control_col = st.columns([4, 1])
            
            with control_col:
                # Chart controls
                st.session_state.interval = st.radio(
                    "Time Interval",
                    options=["intraday", "daily", "weekly"],
                    index=1
                )
                
                st.session_state.chart_type = st.radio(
                    "Chart Type",
                    options=["candlestick", "line"],
                    index=0
                )
                
                # Technical indicators
                st.subheader("Indicators")
                st.session_state.show_indicators['sma'] = st.checkbox("Simple Moving Avg (SMA)")
                st.session_state.show_indicators['ema'] = st.checkbox("Exponential Moving Avg (EMA)")
                st.session_state.show_indicators['bollinger'] = st.checkbox("Bollinger Bands")
            
            with chart_col:
                # Get the chart data based on selected interval and market
                if st.session_state.market == 'india':
                    if st.session_state.interval == 'intraday':
                        df = get_indian_stock_intraday_data(st.session_state.selected_stock)
                    elif st.session_state.interval == 'daily':
                        df = get_indian_stock_daily_data(st.session_state.selected_stock)
                    else:  # weekly
                        df = get_indian_stock_weekly_data(st.session_state.selected_stock)
                else:
                    if st.session_state.interval == 'intraday':
                        df = get_stock_intraday_data(st.session_state.selected_stock)
                    elif st.session_state.interval == 'daily':
                        df = get_stock_daily_data(st.session_state.selected_stock)
                    else:  # weekly
                        df = get_stock_weekly_data(st.session_state.selected_stock)
                
                if df is not None and not df.empty:
                    # Create the base chart figure
                    fig = go.Figure()
                    
                    # Add the main price data
                    if st.session_state.chart_type == 'candlestick':
                        fig.add_trace(
                            go.Candlestick(
                                x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name="Price"
                            )
                        )
                    else:  # line chart
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['close'],
                                mode='lines',
                                name="Price",
                                line=dict(color='royalblue', width=2)
                            )
                        )
                    
                    # Add SMA if selected
                    if st.session_state.show_indicators['sma']:
                        df['SMA20'] = df['close'].rolling(window=20).mean()
                        df['SMA50'] = df['close'].rolling(window=50).mean()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['SMA20'],
                                mode='lines',
                                name="SMA (20)",
                                line=dict(color='orange', width=1)
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['SMA50'],
                                mode='lines',
                                name="SMA (50)",
                                line=dict(color='green', width=1)
                            )
                        )
                    
                    # Add EMA if selected
                    if st.session_state.show_indicators['ema']:
                        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['EMA20'],
                                mode='lines',
                                name="EMA (20)",
                                line=dict(color='purple', width=1)
                            )
                        )
                    
                    # Add Bollinger Bands if selected
                    if st.session_state.show_indicators['bollinger']:
                        df['SMA20'] = df['close'].rolling(window=20).mean()
                        df['upper_band'], df['lower_band'] = calculate_bollinger_bands(df)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['upper_band'],
                                mode='lines',
                                name="Upper Band",
                                line=dict(color='rgba(250, 0, 0, 0.4)', width=1)
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['lower_band'],
                                mode='lines',
                                name="Lower Band",
                                line=dict(color='rgba(0, 250, 0, 0.4)', width=1),
                                fill='tonexty', 
                                fillcolor='rgba(0, 250, 0, 0.05)'
                            )
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{quote['symbol']} - {st.session_state.interval.capitalize()} Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500,
                        margin=dict(l=0, r=0, t=40, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis_rangeslider_visible=False
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"No {st.session_state.interval} data available for {st.session_state.selected_stock}")
            
            # Technical Analysis Section
            st.header("Technical Analysis")
            
            # Calculate technical indicators based on market selection
            if df is not None and not df.empty:
                if st.session_state.market == 'india':
                    # Use Indian market-specific indicators
                    rsi = calculate_indian_rsi(df)
                    macd, signal, hist = calculate_indian_macd(df)
                    volatility_indicators = calculate_indian_volatility_indicators(df)
                    volatility = volatility_indicators['volatility']
                    
                    # Get sector information for Indian stocks
                    from indian_stock_data import INDIAN_STOCKS
                    sector = INDIAN_STOCKS.get(st.session_state.selected_stock, {}).get('sector', None)
                    
                    # Get comprehensive Indian analysis
                    indian_analysis = get_indian_stock_analysis(df, st.session_state.selected_stock, sector)
                else:
                    # Use global indicators
                    rsi = calculate_rsi(df)
                    macd, signal, hist = calculate_macd(df)
                    volatility = calculate_volatility(df)
                    indian_analysis = None
                
                # Display in three columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("RSI (14)")
                    if hasattr(rsi, 'iloc') and not rsi.empty and not pd.isna(rsi.iloc[-1]):
                        rsi_value = float(rsi.iloc[-1])
                    elif hasattr(rsi, 'item'):
                        rsi_value = float(rsi.item())
                    else:
                        rsi_value = float(rsi) if not pd.isna(rsi) else 50
                    
                    # RSI gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=rsi_value,
                        domain=dict(x=[0, 1], y=[0, 1]),
                        gauge=dict(
                            axis=dict(range=[0, 100]),
                            bar=dict(color="gray"),
                            steps=[
                                dict(range=[0, 30], color="green"),
                                dict(range=[30, 70], color="yellow"),
                                dict(range=[70, 100], color="red")
                            ],
                            threshold=dict(
                                line=dict(color="black", width=4),
                                thickness=0.75,
                                value=rsi_value
                            )
                        )
                    ))
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if rsi_value < 30:
                        st.markdown("**Interpretation:** Potentially oversold")
                    elif rsi_value > 70:
                        st.markdown("**Interpretation:** Potentially overbought")
                    else:
                        st.markdown("**Interpretation:** Neutral")
                
                with col2:
                    st.subheader("MACD")
                    
                    # MACD Line chart
                    fig = go.Figure()
                    
                    # Add MACD line
                    fig.add_trace(go.Scatter(
                        x=macd.index[-30:],
                        y=macd.values[-30:],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add Signal line
                    fig.add_trace(go.Scatter(
                        x=signal.index[-30:],
                        y=signal.values[-30:],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=1)
                    ))
                    
                    # Add histogram
                    colors = ['green' if x > 0 else 'red' for x in hist.values[-30:]]
                    fig.add_trace(go.Bar(
                        x=hist.index[-30:],
                        y=hist.values[-30:],
                        name='Histogram',
                        marker_color=colors
                    ))
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # MACD interpretation
                    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                        st.markdown("**Interpretation:** Bullish crossover (Buy signal)")
                    elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
                        st.markdown("**Interpretation:** Bearish crossover (Sell signal)")
                    elif macd.iloc[-1] > signal.iloc[-1]:
                        st.markdown("**Interpretation:** MACD above signal line (Bullish)")
                    else:
                        st.markdown("**Interpretation:** MACD below signal line (Bearish)")
                
                with col3:
                    st.subheader("Volatility")
                    
                    # Volatility gauge
                    if hasattr(volatility, 'iloc') and not volatility.empty and not pd.isna(volatility.iloc[-1]):
                        volatility_pct = float(volatility.iloc[-1]) * 100
                    elif hasattr(volatility, 'item'):
                        volatility_pct = float(volatility.item()) * 100
                    else:
                        volatility_pct = float(volatility) * 100 if not pd.isna(volatility) else 0
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=volatility_pct,
                        number=dict(suffix="%"),
                        domain=dict(x=[0, 1], y=[0, 1]),
                        gauge=dict(
                            axis=dict(range=[0, 5]),
                            bar=dict(color="darkblue"),
                            steps=[
                                dict(range=[0, 1], color="green"),
                                dict(range=[1, 3], color="yellow"),
                                dict(range=[3, 5], color="red")
                            ],
                            threshold=dict(
                                line=dict(color="black", width=4),
                                thickness=0.75,
                                value=volatility_pct
                            )
                        )
                    ))
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volatility interpretation
                    if volatility_pct < 1:
                        st.markdown("**Interpretation:** Low volatility")
                    elif volatility_pct < 3:
                        st.markdown("**Interpretation:** Moderate volatility")
                    else:
                        st.markdown("**Interpretation:** High volatility")
                
                # Display Indian market-specific analysis if available
                if st.session_state.market == 'india' and indian_analysis:
                    st.subheader("Indian Market Analysis")
                    
                    # Create columns for Indian analysis
                    ind_col1, ind_col2, ind_col3 = st.columns(3)
                    
                    with ind_col1:
                        st.markdown("### Sector Analysis")
                        if sector:
                            st.markdown(f"**Sector:** {sector}")
                        
                        # Display sector-specific indicators
                        if 'rate_sensitivity' in indian_analysis:
                            st.markdown("**Rate Sensitivity:** High")
                        if 'dollar_sensitivity' in indian_analysis:
                            st.markdown("**Dollar Sensitivity:** High")
                        if 'regulatory_risk' in indian_analysis:
                            st.markdown("**Regulatory Risk:** Moderate")
                    
                    with ind_col2:
                        st.markdown("### Market Sentiment")
                        fear_greed = indian_analysis.get('fear_greed', 50)
                        fear_greed_value = float(fear_greed.iloc[-1]) if hasattr(fear_greed, 'iloc') and not fear_greed.empty and not pd.isna(fear_greed.iloc[-1]) else 50
                        if fear_greed_value > 70:
                            st.markdown("**Market Sentiment:** 🟢 Greed")
                        elif fear_greed_value < 30:
                            st.markdown("**Market Sentiment:** 🔴 Fear")
                        else:
                            st.markdown("**Market Sentiment:** 🟡 Neutral")
                        
                        # Display volume analysis
                        volume_ratio = indian_analysis.get('volume_ratio', 1.0)
                        volume_ratio_value = float(volume_ratio.iloc[-1]) if hasattr(volume_ratio, 'iloc') and not volume_ratio.empty and not pd.isna(volume_ratio.iloc[-1]) else 1.0
                        if volume_ratio_value > 1.5:
                            st.markdown("**Volume:** 🔥 High Activity")
                        elif volume_ratio_value < 0.5:
                            st.markdown("**Volume:** 📉 Low Activity")
                        else:
                            st.markdown("**Volume:** 📊 Normal")
                    
                    with ind_col3:
                        st.markdown("### Momentum Analysis")
                        momentum_5 = indian_analysis.get('momentum_5', 0)
                        momentum_20 = indian_analysis.get('momentum_20', 0)
                        
                        momentum_5_value = float(momentum_5.iloc[-1]) if hasattr(momentum_5, 'iloc') and not momentum_5.empty and not pd.isna(momentum_5.iloc[-1]) else 0
                        momentum_20_value = float(momentum_20.iloc[-1]) if hasattr(momentum_20, 'iloc') and not momentum_20.empty and not pd.isna(momentum_20.iloc[-1]) else 0
                        
                        if momentum_5_value > 0 and momentum_20_value > 0:
                            st.markdown("**Short-term:** 🟢 Bullish")
                        elif momentum_5_value < 0 and momentum_20_value < 0:
                            st.markdown("**Short-term:** 🔴 Bearish")
                        else:
                            st.markdown("**Short-term:** 🟡 Mixed")
                        
                        if momentum_20_value > 0:
                            st.markdown("**Medium-term:** 🟢 Bullish")
                        else:
                            st.markdown("**Medium-term:** 🔴 Bearish")
            
            # Market Sentiment and AI Predictions Section
            st.header("AI-Powered Insights")
            
            # Create columns for sentiment and prediction
            sent_col, pred_col = st.columns(2)
            
            with sent_col:
                st.subheader("Market Sentiment")
                
                # Get sentiment analysis
                with st.spinner("Analyzing market sentiment..."):
                    sentiment_data = get_stock_sentiment(st.session_state.selected_stock)
                    
                    if sentiment_data:
                        # Create sentiment gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=sentiment_data['score'] * 100,
                            domain=dict(x=[0, 1], y=[0, 1]),
                            title=dict(text="Sentiment Score"),
                            gauge=dict(
                                axis=dict(range=[0, 100]),
                                bar=dict(color="darkblue"),
                                steps=[
                                    dict(range=[0, 33], color="red"),
                                    dict(range=[33, 66], color="yellow"),
                                    dict(range=[66, 100], color="green")
                                ],
                                threshold=dict(
                                    line=dict(color="black", width=4),
                                    thickness=0.75,
                                    value=sentiment_data['score'] * 100
                                )
                            )
                        ))
                        
                        fig.update_layout(
                            height=250,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Sentiment breakdown
                        st.markdown("### Recent News Sentiment")
                        st.markdown(f"**Sources Analyzed:** {sentiment_data['sources_count']}")
                        st.markdown(f"**Overall Outlook:** {sentiment_data['outlook']}")
                        
                        # Display a few news headlines
                        st.markdown("### Latest Headlines")
                        for headline in sentiment_data['headlines'][:3]:
                            sentiment_color = "green" if headline['sentiment'] > 0.5 else "red" if headline['sentiment'] < 0.3 else "orange"
                            st.markdown(f"• <span style='color:{sentiment_color}'>{headline['title']}</span>", unsafe_allow_html=True)
                    else:
                        st.warning("Sentiment data unavailable for this stock")
            
            with pred_col:
                st.subheader("AI Trend Prediction")
                
                # Get prediction data
                with st.spinner("Generating AI predictions..."):
                    prediction = predict_trend(st.session_state.selected_stock, df)
                    
                    if prediction:
                        # Display prediction score
                        trend_direction = "Bullish" if prediction['trend_direction'] == 'up' else "Bearish"
                        trend_color = "green" if prediction['trend_direction'] == 'up' else "red"
                        
                        st.markdown(f"### Predicted Trend: <span style='color:{trend_color}'>{trend_direction}</span>", unsafe_allow_html=True)
                        
                        # Create confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction['confidence'] * 100,
                            domain=dict(x=[0, 1], y=[0, 1]),
                            title=dict(text="Confidence Score"),
                            number=dict(suffix="%"),
                            gauge=dict(
                                axis=dict(range=[0, 100]),
                                bar=dict(color="darkblue"),
                                steps=[
                                    dict(range=[0, 33], color="red"),
                                    dict(range=[33, 66], color="yellow"),
                                    dict(range=[66, 100], color="green")
                                ],
                                threshold=dict(
                                    line=dict(color="black", width=4),
                                    thickness=0.75,
                                    value=prediction['confidence'] * 100
                                )
                            )
                        ))
                        
                        fig.update_layout(
                            height=250,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show prediction details
                        st.markdown("### Price Target Range (7 Days)")
                        st.markdown(f"**Low:** ${prediction['price_target_low']:.2f}")
                        st.markdown(f"**High:** ${prediction['price_target_high']:.2f}")
                        
                        # Feature importance
                        st.markdown("### Key Factors Influencing Prediction")
                        for factor in prediction['factors'][:3]:
                            impact_color = "green" if factor['impact'] > 0 else "red"
                            impact_symbol = "↑" if factor['impact'] > 0 else "↓"
                            st.markdown(f"• {factor['name']}: <span style='color:{impact_color}'>{impact_symbol}</span>", unsafe_allow_html=True)
                    else:
                        st.warning("Prediction data unavailable for this stock")
            
            # Buy/Sell Signal
            st.header("Trade Signal")
            
            # Calculate overall signal
            with st.spinner("Calculating trade signal..."):
                if df is not None and not df.empty:
                    # Get all the factors
                    signal_factors = {
                        'rsi': rsi.iloc[-1],
                        'macd_hist': hist.iloc[-1],
                        'sentiment': sentiment_data['score'] if sentiment_data else 0.5,
                        'prediction': prediction['confidence'] if prediction else 0.5,
                        'trend': 1 if prediction and prediction['trend_direction'] == 'up' else -1 if prediction else 0,
                        'price_momentum': (df['close'].iloc[-1] / df['close'].iloc[-7] - 1) * 100 if len(df) >= 7 else 0
                    }
                    
                    # Calculate signal
                    signal_result = calculate_signal(signal_factors)
                    
                    # Display signal
                    signal_col, explain_col = st.columns([1, 2])
                    
                    with signal_col:
                        if signal_result['action'] == 'buy':
                            st.markdown("## 🟢 BUY")
                            signal_color = "green"
                        elif signal_result['action'] == 'sell':
                            st.markdown("## 🔴 SELL")
                            signal_color = "red"
                        else:
                            st.markdown("## 🟡 HOLD")
                            signal_color = "orange"
                            
                        # Signal strength meter
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=signal_result['strength'] * 100,
                            domain=dict(x=[0, 1], y=[0, 1]),
                            title=dict(text="Signal Strength"),
                            number=dict(suffix="%"),
                            gauge=dict(
                                axis=dict(range=[0, 100]),
                                bar=dict(color=signal_color),
                                steps=[
                                    dict(range=[0, 33], color="lightgray"),
                                    dict(range=[33, 66], color="lightgray"),
                                    dict(range=[66, 100], color="lightgray")
                                ],
                                threshold=dict(
                                    line=dict(color="black", width=4),
                                    thickness=0.75,
                                    value=signal_result['strength'] * 100
                                )
                            )
                        ))
                        
                        fig.update_layout(
                            height=250,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with explain_col:
                        st.markdown("### Signal Explanation")
                        st.markdown(signal_result['explanation'])
                        
                        st.markdown("### Risk Level")
                        risk_level = signal_result.get('risk', 'Medium')
                        risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
                        st.markdown(f"**Risk:** <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                        
                        st.markdown("### Suggested Position Size")
                        position_size = signal_result.get('position_size', '5-10% of portfolio')
                        st.markdown(f"**Recommended Size:** {position_size}")
                        
                        # Disclaimer
                        st.info("This is an automated signal based on technical analysis and market sentiment. Always do your own research before making investment decisions.")
                else:
                    st.error("Unable to calculate trade signal due to missing data")
        
    # Update data automatically (uncomment for production)
    # time.sleep(60)  
    # st.rerun()

else:
    # Landing page when no stock is selected
    st.markdown("""
    ## Welcome to MoneyMentor-AI
    
    MoneyMentor-AI is a powerful stock analysis platform that combines real-time market data with AI-powered insights 
    to help you make informed trading decisions.
    
    ### Get Started
    
    1. Use the search box in the sidebar to find a stock by name or ticker symbol
    2. Apply filters to discover stocks that match your investment criteria
    3. Select a stock to view detailed analysis and AI predictions
    
    ### Key Features
    
    - **Real-time Charts:** View intraday, daily, and weekly price charts with technical indicators
    - **Technical Analysis:** Calculate key metrics like RSI, MACD, and volatility
    - **AI-Powered Insights:** Get sentiment analysis and trend predictions
    - **Trade Signals:** Receive buy, sell, or hold recommendations based on multiple factors
    """)
    
    # Sample stocks section based on market selection
    if st.session_state.market == 'india':
        st.subheader("Popular Indian Stocks")
        
        # Create a grid of popular Indian stocks
        col1, col2, col3, col4 = st.columns(4)
        
        popular_indian = get_popular_indian_stocks()[:8]
        
        with col1:
            if st.button(f"{popular_indian[0]['symbol']} - {popular_indian[0]['name'][:20]}..."):
                st.session_state.selected_stock = popular_indian[0]['symbol']
                st.rerun()
            if st.button(f"{popular_indian[1]['symbol']} - {popular_indian[1]['name'][:20]}..."):
                st.session_state.selected_stock = popular_indian[1]['symbol']
                st.rerun()
                
        with col2:
            if st.button(f"{popular_indian[2]['symbol']} - {popular_indian[2]['name'][:20]}..."):
                st.session_state.selected_stock = popular_indian[2]['symbol']
                st.rerun()
            if st.button(f"{popular_indian[3]['symbol']} - {popular_indian[3]['name'][:20]}..."):
                st.session_state.selected_stock = popular_indian[3]['symbol']
                st.rerun()
                
        with col3:
            if st.button(f"{popular_indian[4]['symbol']} - {popular_indian[4]['name'][:20]}..."):
                st.session_state.selected_stock = popular_indian[4]['symbol']
                st.rerun()
            if st.button(f"{popular_indian[5]['symbol']} - {popular_indian[5]['name'][:20]}..."):
                st.session_state.selected_stock = popular_indian[5]['symbol']
                st.rerun()
                
        with col4:
            if st.button(f"{popular_indian[6]['symbol']} - {popular_indian[6]['name'][:20]}..."):
                st.session_state.selected_stock = popular_indian[6]['symbol']
                st.rerun()
            if st.button(f"{popular_indian[7]['symbol']} - {popular_indian[7]['name'][:20]}..."):
                st.session_state.selected_stock = popular_indian[7]['symbol']
                st.rerun()
    else:
        st.subheader("Popular Global Stocks")
        
        # Create a grid of sample stocks
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("AAPL - Apple Inc."):
                st.session_state.selected_stock = "AAPL"
                st.rerun()
                
        with col2:
            if st.button("MSFT - Microsoft Corp"):
                st.session_state.selected_stock = "MSFT"
                st.rerun()
                
        with col3:
            if st.button("AMZN - Amazon.com"):
                st.session_state.selected_stock = "AMZN"
                st.rerun()
                
        with col4:
            if st.button("GOOGL - Alphabet Inc"):
                st.session_state.selected_stock = "GOOGL"
                st.rerun()
