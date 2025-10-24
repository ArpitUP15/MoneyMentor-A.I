import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def train_trend_prediction_model(symbol, df):
    """
    Train a simple trend prediction model for a stock
    This is a placeholder for a more sophisticated model in a real implementation
    """
    if df is None or len(df) < 30:
        return None
    
    # In a real implementation, this would train an actual ML model
    # For demo, we'll just return a function that makes "predictions"
    
    # Calculate some simple features
    df_copy = df.copy()
    
    # Calculate returns
    df_copy['returns'] = df_copy['close'].pct_change()
    
    # Calculate moving averages
    df_copy['sma20'] = df_copy['close'].rolling(window=20).mean()
    df_copy['sma50'] = df_copy['close'].rolling(window=50).mean()
    
    # Calculate price relative to moving averages
    df_copy['price_sma20'] = df_copy['close'] / df_copy['sma20']
    df_copy['price_sma50'] = df_copy['close'] / df_copy['sma50']
    
    # Calculate RSI
    delta = df_copy['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_copy['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate volatility
    df_copy['volatility'] = df_copy['returns'].rolling(window=20).std()
    
    # Get recent data
    recent_data = df_copy.dropna().iloc[-30:]
    
    # Create a model "bias" based on recent trends
    # This simulates a model prediction
    
    # Check moving average crossovers
    ma_trend = 0
    if recent_data['sma20'].iloc[-1] > recent_data['sma50'].iloc[-1]:
        ma_trend = 0.6  # Bullish
    elif recent_data['sma20'].iloc[-1] < recent_data['sma50'].iloc[-1]:
        ma_trend = 0.4  # Bearish
    else:
        ma_trend = 0.5  # Neutral
    
    # Check RSI
    rsi_trend = 0
    rsi = recent_data['rsi'].iloc[-1]
    if rsi > 70:
        rsi_trend = 0.3  # Overbought - bearish
    elif rsi < 30:
        rsi_trend = 0.7  # Oversold - bullish
    else:
        rsi_trend = 0.5  # Neutral
    
    # Check recent price momentum
    momentum = 0
    if recent_data['returns'].mean() > 0:
        momentum = 0.6  # Positive momentum
    else:
        momentum = 0.4  # Negative momentum
    
    # Average the trends
    trend_bias = (ma_trend + rsi_trend + momentum) / 3
    
    # Calculate model volatility based on stock volatility
    model_volatility = recent_data['volatility'].mean() * 10
    
    # Create a bias for this specific stock (including Indian stocks)
    stock_biases = {
        # Global stocks
        "AAPL": 0.55,  # Slightly bullish
        "MSFT": 0.6,   # Moderately bullish
        "GOOGL": 0.55, # Slightly bullish
        "AMZN": 0.5,   # Neutral
        "META": 0.5,   # Neutral
        "TSLA": 0.45,  # Slightly bearish
        "NVDA": 0.65,  # Very bullish
        "JPM": 0.5,    # Neutral
        "V": 0.55,     # Slightly bullish
        "JNJ": 0.5,    # Neutral
        # Indian stocks
        "RELIANCE": 0.6,      # Strong bullish (oil & gas leader)
        "TCS": 0.65,          # Very bullish (IT leader)
        "HDFCBANK": 0.6,      # Strong bullish (banking leader)
        "INFY": 0.6,          # Strong bullish (IT major)
        "HINDUNILVR": 0.55,   # Slightly bullish (FMCG leader)
        "ITC": 0.5,           # Neutral (FMCG)
        "SBIN": 0.5,          # Neutral (PSU bank)
        "BHARTIARTL": 0.6,    # Strong bullish (telecom leader)
        "KOTAKBANK": 0.6,     # Strong bullish (private bank)
        "LT": 0.55,           # Slightly bullish (engineering)
        "ASIANPAINT": 0.6,    # Strong bullish (paints leader)
        "MARUTI": 0.5,        # Neutral (auto)
        "AXISBANK": 0.6,      # Strong bullish (private bank)
        "NESTLEIND": 0.55,    # Slightly bullish (FMCG)
        "ULTRACEMCO": 0.55,   # Slightly bullish (cement)
        "TITAN": 0.6,         # Strong bullish (jewelry)
        "POWERGRID": 0.5,     # Neutral (power utility)
        "NTPC": 0.5,          # Neutral (power utility)
        "SUNPHARMA": 0.5,     # Neutral (pharma)
        "TECHM": 0.6,         # Strong bullish (IT)
        "WIPRO": 0.55,        # Slightly bullish (IT)
        "HCLTECH": 0.6,       # Strong bullish (IT)
        "ONGC": 0.5,          # Neutral (oil & gas)
        "COALINDIA": 0.5,     # Neutral (mining)
        "TATAMOTORS": 0.5,    # Neutral (auto)
        "BAJFINANCE": 0.6,    # Strong bullish (NBFC)
        "BAJAJFINSV": 0.6,    # Strong bullish (financial services)
        "DRREDDY": 0.55,      # Slightly bullish (pharma)
        "CIPLA": 0.55,        # Slightly bullish (pharma)
        "APOLLOHOSP": 0.6,    # Strong bullish (healthcare)
        "DIVISLAB": 0.6,      # Strong bullish (pharma)
        "EICHERMOT": 0.55,    # Slightly bullish (auto)
        "HEROMOTOCO": 0.5,    # Neutral (auto)
        "BAJAJ-AUTO": 0.5,    # Neutral (auto)
        "M&M": 0.5,           # Neutral (auto)
        "TATACONSUM": 0.55,   # Slightly bullish (FMCG)
        "BRITANNIA": 0.55,    # Slightly bullish (FMCG)
        "DABUR": 0.55,        # Slightly bullish (FMCG)
        "GODREJCP": 0.55,     # Slightly bullish (FMCG)
        "PIDILITIND": 0.6,    # Strong bullish (chemicals)
        "ADANIPORTS": 0.5,    # Neutral (infrastructure)
        "ADANIENT": 0.5,      # Neutral (diversified)
        "ADANIGREEN": 0.6,    # Strong bullish (renewable energy)
        "ADANITRANS": 0.5,    # Neutral (power)
        "GRASIM": 0.5,        # Neutral (textiles)
        "JSWSTEEL": 0.5,      # Neutral (steel)
        "TATASTEEL": 0.5,     # Neutral (steel)
        "HINDALCO": 0.5,      # Neutral (metals)
        "VEDL": 0.5,          # Neutral (mining)
        "JINDALSTEL": 0.5,    # Neutral (steel)
        "SAIL": 0.5,          # Neutral (steel)
        "BHEL": 0.5,          # Neutral (engineering)
        "SIEMENS": 0.55,      # Slightly bullish (engineering)
        "ABBOTINDIA": 0.6,    # Strong bullish (pharma)
        "BIOCON": 0.6,        # Strong bullish (pharma)
        "LUPIN": 0.55,        # Slightly bullish (pharma)
        "CADILAHC": 0.55,     # Slightly bullish (pharma)
        "TORNTPHARM": 0.55,   # Slightly bullish (pharma)
        "AUROPHARMA": 0.55,   # Slightly bullish (pharma)
        "GLENMARK": 0.55,     # Slightly bullish (pharma)
        "MINDTREE": 0.6,      # Strong bullish (IT)
        "LTI": 0.6,           # Strong bullish (IT)
        "MPHASIS": 0.6,       # Strong bullish (IT)
        "PERSISTENT": 0.6,    # Strong bullish (IT)
        "COFORGE": 0.6,       # Strong bullish (IT)
        "LALPATHLAB": 0.6,    # Strong bullish (healthcare)
        "FORTIS": 0.6,        # Strong bullish (healthcare)
        "MAXHEALTH": 0.6,     # Strong bullish (healthcare)
        "NARAYANK": 0.6,      # Strong bullish (healthcare)
        "METROPOLIS": 0.6,    # Strong bullish (healthcare)
        "REDINGTON": 0.55,    # Slightly bullish (technology)
    }
    
    stock_bias = stock_biases.get(symbol, 0.5)
    
    # Combine all biases
    combined_bias = (trend_bias * 0.5) + (stock_bias * 0.5)
    
    # Return the "model"
    return {
        "bias": combined_bias,
        "volatility": model_volatility,
        "recent_close": df_copy['close'].iloc[-1],
        "factors": [
            {"name": "Moving Average Trend", "value": ma_trend, "impact": ma_trend - 0.5},
            {"name": "RSI Signal", "value": rsi_trend, "impact": rsi_trend - 0.5},
            {"name": "Price Momentum", "value": momentum, "impact": momentum - 0.5},
            {"name": "Stock-Specific Factors", "value": stock_bias, "impact": stock_bias - 0.5}
        ]
    }

def predict_trend(symbol, df):
    """
    Generate trend prediction for a stock
    Returns a dictionary with prediction data
    """
    if df is None or len(df) < 30:
        return None
    
    # Train our simple "model"
    model = train_trend_prediction_model(symbol, df)
    
    if not model:
        return None
    
    # Generate prediction
    bias = model["bias"]
    
    # Determine trend direction
    trend_direction = "up" if bias > 0.5 else "down"
    
    # Calculate confidence
    # Map bias from [0,1] to confidence [0.5,1]
    confidence = 0.5 + abs(bias - 0.5)
    
    # Calculate price targets
    current_price = model["recent_close"]
    volatility = model["volatility"]
    
    # Price targets (7-day forecast)
    # Use volatility to determine range
    price_change_range = current_price * volatility
    
    if trend_direction == "up":
        price_target_low = current_price
        price_target_high = current_price * (1 + (confidence - 0.5) * 2 * volatility)
    else:
        price_target_low = current_price * (1 - (confidence - 0.5) * 2 * volatility)
        price_target_high = current_price
    
    # Get factors with highest impact
    factors = sorted(model["factors"], key=lambda x: abs(x["impact"]), reverse=True)
    
    return {
        "trend_direction": trend_direction,
        "confidence": confidence,
        "price_target_low": price_target_low,
        "price_target_high": price_target_high,
        "factors": factors
    }

def simulate_prediction_performance(symbol, period_days=30):
    """
    Simulate past prediction performance
    Returns accuracy metrics for backtesting display
    """
    # This would be a real backtesting system in a production model
    # For demo, we'll return simulated performance stats
    
    # Higher accuracy for some stocks (including Indian stocks)
    base_accuracy = {
        # Global stocks
        "AAPL": 0.62,
        "MSFT": 0.65,
        "GOOGL": 0.61,
        "AMZN": 0.58,
        "META": 0.55,
        "TSLA": 0.53,  # More volatile, harder to predict
        "NVDA": 0.68,
        "JPM": 0.60,
        "V": 0.63,
        "JNJ": 0.61,
        # Indian stocks
        "RELIANCE": 0.65,      # High accuracy (stable large cap)
        "TCS": 0.68,           # Very high accuracy (IT leader)
        "HDFCBANK": 0.66,      # High accuracy (banking leader)
        "INFY": 0.67,          # High accuracy (IT major)
        "HINDUNILVR": 0.64,    # High accuracy (FMCG leader)
        "ITC": 0.62,           # Good accuracy (FMCG)
        "SBIN": 0.60,          # Moderate accuracy (PSU bank)
        "BHARTIARTL": 0.65,    # High accuracy (telecom leader)
        "KOTAKBANK": 0.66,     # High accuracy (private bank)
        "LT": 0.63,            # Good accuracy (engineering)
        "ASIANPAINT": 0.65,    # High accuracy (paints leader)
        "MARUTI": 0.61,        # Good accuracy (auto)
        "AXISBANK": 0.65,      # High accuracy (private bank)
        "NESTLEIND": 0.64,     # High accuracy (FMCG)
        "ULTRACEMCO": 0.63,    # Good accuracy (cement)
        "TITAN": 0.66,         # High accuracy (jewelry)
        "POWERGRID": 0.62,     # Good accuracy (power utility)
        "NTPC": 0.61,          # Good accuracy (power utility)
        "SUNPHARMA": 0.63,     # Good accuracy (pharma)
        "TECHM": 0.66,         # High accuracy (IT)
        "WIPRO": 0.64,         # High accuracy (IT)
        "HCLTECH": 0.67,       # High accuracy (IT)
        "ONGC": 0.60,          # Moderate accuracy (oil & gas)
        "COALINDIA": 0.59,     # Moderate accuracy (mining)
        "TATAMOTORS": 0.58,    # Moderate accuracy (auto)
        "BAJFINANCE": 0.66,    # High accuracy (NBFC)
        "BAJAJFINSV": 0.65,    # High accuracy (financial services)
        "DRREDDY": 0.64,       # High accuracy (pharma)
        "CIPLA": 0.63,         # Good accuracy (pharma)
        "APOLLOHOSP": 0.66,    # High accuracy (healthcare)
        "DIVISLAB": 0.67,      # High accuracy (pharma)
        "EICHERMOT": 0.62,     # Good accuracy (auto)
        "HEROMOTOCO": 0.61,    # Good accuracy (auto)
        "BAJAJ-AUTO": 0.60,    # Good accuracy (auto)
        "M&M": 0.59,           # Moderate accuracy (auto)
        "TATACONSUM": 0.63,    # Good accuracy (FMCG)
        "BRITANNIA": 0.64,     # High accuracy (FMCG)
        "DABUR": 0.63,         # Good accuracy (FMCG)
        "GODREJCP": 0.62,      # Good accuracy (FMCG)
        "PIDILITIND": 0.65,    # High accuracy (chemicals)
        "ADANIPORTS": 0.61,     # Good accuracy (infrastructure)
        "ADANIENT": 0.60,      # Good accuracy (diversified)
        "ADANIGREEN": 0.64,    # High accuracy (renewable energy)
        "ADANITRANS": 0.62,     # Good accuracy (power)
        "GRASIM": 0.60,        # Good accuracy (textiles)
        "JSWSTEEL": 0.59,      # Moderate accuracy (steel)
        "TATASTEEL": 0.58,     # Moderate accuracy (steel)
        "HINDALCO": 0.58,      # Moderate accuracy (metals)
        "VEDL": 0.57,          # Moderate accuracy (mining)
        "JINDALSTEL": 0.57,    # Moderate accuracy (steel)
        "SAIL": 0.56,          # Moderate accuracy (steel)
        "BHEL": 0.59,          # Moderate accuracy (engineering)
        "SIEMENS": 0.63,       # Good accuracy (engineering)
        "ABBOTINDIA": 0.65,    # High accuracy (pharma)
        "BIOCON": 0.66,        # High accuracy (pharma)
        "LUPIN": 0.64,         # High accuracy (pharma)
        "CADILAHC": 0.63,      # Good accuracy (pharma)
        "TORNTPHARM": 0.64,    # High accuracy (pharma)
        "AUROPHARMA": 0.63,    # Good accuracy (pharma)
        "GLENMARK": 0.62,      # Good accuracy (pharma)
        "MINDTREE": 0.67,      # High accuracy (IT)
        "LTI": 0.66,           # High accuracy (IT)
        "MPHASIS": 0.65,       # High accuracy (IT)
        "PERSISTENT": 0.66,    # High accuracy (IT)
        "COFORGE": 0.65,       # High accuracy (IT)
        "LALPATHLAB": 0.66,    # High accuracy (healthcare)
        "FORTIS": 0.65,        # High accuracy (healthcare)
        "MAXHEALTH": 0.66,     # High accuracy (healthcare)
        "NARAYANK": 0.65,      # High accuracy (healthcare)
        "METROPOLIS": 0.66,    # High accuracy (healthcare)
        "REDINGTON": 0.63,     # Good accuracy (technology)
    }
    
    accuracy = base_accuracy.get(symbol, 0.58)
    
    # Add some randomness
    accuracy = min(0.85, max(0.45, accuracy + random.uniform(-0.05, 0.05)))
    
    # Generate daily predictions (up/down) and outcomes
    np.random.seed(hash(symbol) % 10000)
    
    days = []
    predictions = []
    outcomes = []
    confidence_scores = []
    
    for i in range(period_days):
        day = (datetime.now() - timedelta(days=period_days-i)).strftime("%Y-%m-%d")
        days.append(day)
        
        # Generate a prediction (1 for up, 0 for down)
        true_outcome = np.random.randint(0, 2)
        
        # Model is right with probability 'accuracy'
        if np.random.random() < accuracy:
            prediction = true_outcome
        else:
            prediction = 1 - true_outcome
        
        # Generate confidence score
        confidence = np.random.uniform(0.55, 0.85)
        
        predictions.append(prediction)
        outcomes.append(true_outcome)
        confidence_scores.append(confidence)
    
    # Calculate metrics
    correct_predictions = sum(p == o for p, o in zip(predictions, outcomes))
    total_predictions = len(predictions)
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate accuracy by confidence level
    high_conf_indices = [i for i, conf in enumerate(confidence_scores) if conf > 0.7]
    high_conf_correct = sum(predictions[i] == outcomes[i] for i in high_conf_indices)
    high_conf_accuracy = high_conf_correct / len(high_conf_indices) if high_conf_indices else 0
    
    # Calculate profit if trading on signals (simplified)
    profit = sum([(o * 2 - 1) * (p * 2 - 1) * 0.01 for p, o in zip(predictions, outcomes)])
    
    return {
        "days": days,
        "predictions": predictions,
        "outcomes": outcomes,
        "confidence_scores": confidence_scores,
        "overall_accuracy": overall_accuracy,
        "high_confidence_accuracy": high_conf_accuracy,
        "simulated_profit": profit
    }
