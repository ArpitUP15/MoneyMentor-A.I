def format_large_number(num):
    """Format large numbers with T, B, M suffixes"""
    if not isinstance(num, (int, float)):
        return str(num)
    
    if num >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f}T"
    elif num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"
    
def calculate_signal(factors):
    """
    Calculate a buy/sell signal based on multiple factors
    
    Args:
        factors: dictionary with various technical and sentiment factors
        
    Returns:
        Dictionary with signal information
    """
    # Extract factors
    rsi = factors.get('rsi', 50)
    macd_hist = factors.get('macd_hist', 0)
    sentiment = factors.get('sentiment', 0.5)
    prediction = factors.get('prediction', 0.5)
    trend = factors.get('trend', 0)
    price_momentum = factors.get('price_momentum', 0)
    
    # Calculate signal scores
    rsi_score = 0
    if rsi < 30:
        rsi_score = 1  # Oversold - bullish
    elif rsi > 70:
        rsi_score = -1  # Overbought - bearish
    else:
        rsi_score = (50 - rsi) / 20  # Linear scale between 30-70
    
    macd_score = 1 if macd_hist > 0 else -1
    
    # Scale sentiment from [0,1] to [-1,1]
    sentiment_score = (sentiment - 0.5) * 2
    
    # Scale prediction from [0,1] to [-1,1]
    prediction_score = (prediction - 0.5) * 2
    
    # Price momentum score
    momentum_score = 1 if price_momentum > 3 else (-1 if price_momentum < -3 else price_momentum / 3)
    
    # Weight the factors
    weights = {
        'rsi': 0.15,
        'macd': 0.20,
        'sentiment': 0.15,
        'prediction': 0.25,
        'trend': 0.15,
        'momentum': 0.10
    }
    
    # Calculate weighted score
    weighted_score = (
        rsi_score * weights['rsi'] +
        macd_score * weights['macd'] +
        sentiment_score * weights['sentiment'] +
        prediction_score * weights['prediction'] +
        trend * weights['trend'] +
        momentum_score * weights['momentum']
    )
    
    # Determine signal
    if weighted_score > 0.2:
        action = 'buy'
    elif weighted_score < -0.2:
        action = 'sell'
    else:
        action = 'hold'
    
    # Calculate signal strength (0-1)
    signal_strength = min(1.0, abs(weighted_score))
    
    # Generate explanation
    explanation = generate_signal_explanation(
        action, 
        signal_strength, 
        rsi, 
        macd_hist, 
        sentiment, 
        prediction, 
        trend
    )
    
    # Determine risk level
    risk = determine_risk_level(rsi, sentiment, prediction, signal_strength)
    
    # Suggest position size based on signal strength and risk
    position_size = suggest_position_size(signal_strength, risk)
    
    return {
        'action': action,
        'strength': signal_strength,
        'explanation': explanation,
        'risk': risk,
        'position_size': position_size
    }

def generate_signal_explanation(action, strength, rsi, macd_hist, sentiment, prediction, trend):
    """Generate human-readable explanation for a signal"""
    if action == 'buy':
        explanation = "**Buy Signal Analysis:**\n\n"
        
        # Add factor explanations
        factors = []
        
        if rsi < 30:
            factors.append("• RSI indicates the stock is oversold (RSI = {:.1f})".format(rsi))
        
        if macd_hist > 0:
            factors.append("• MACD histogram is positive, suggesting bullish momentum")
        
        if sentiment > 0.6:
            factors.append("• Market sentiment is positive ({:.0f}% bullish)".format(sentiment * 100))
        
        if prediction > 0.6:
            factors.append("• AI prediction model forecasts upward movement with {:.0f}% confidence".format(prediction * 100))
        
        if trend > 0:
            factors.append("• Current price trend is bullish")
        
        # Add fallback if no specific factors
        if not factors:
            factors.append("• Overall technical indicators and sentiment suggest positive outlook")
        
        # Add explanation based on strength
        if strength > 0.8:
            explanation += "Very strong buy signal based on multiple converging factors:\n\n"
        elif strength > 0.5:
            explanation += "Moderate buy signal supported by several indicators:\n\n"
        else:
            explanation += "Weak buy signal with some positive indicators:\n\n"
        
        # Add all factors
        explanation += "\n".join(factors)
        
    elif action == 'sell':
        explanation = "**Sell Signal Analysis:**\n\n"
        
        # Add factor explanations
        factors = []
        
        if rsi > 70:
            factors.append("• RSI indicates the stock is overbought (RSI = {:.1f})".format(rsi))
        
        if macd_hist < 0:
            factors.append("• MACD histogram is negative, suggesting bearish momentum")
        
        if sentiment < 0.4:
            factors.append("• Market sentiment is negative ({:.0f}% bearish)".format((1-sentiment) * 100))
        
        if prediction < 0.4:
            factors.append("• AI prediction model forecasts downward movement with {:.0f}% confidence".format((1-prediction) * 100))
        
        if trend < 0:
            factors.append("• Current price trend is bearish")
        
        # Add fallback if no specific factors
        if not factors:
            factors.append("• Overall technical indicators and sentiment suggest negative outlook")
        
        # Add explanation based on strength
        if strength > 0.8:
            explanation += "Very strong sell signal based on multiple converging factors:\n\n"
        elif strength > 0.5:
            explanation += "Moderate sell signal supported by several indicators:\n\n"
        else:
            explanation += "Weak sell signal with some concerning indicators:\n\n"
        
        # Add all factors
        explanation += "\n".join(factors)
        
    else:  # hold
        explanation = "**Hold Recommendation Analysis:**\n\n"
        
        explanation += "Mixed signals in the current market indicators suggest maintaining current positions:\n\n"
        
        # Add some specific mixed signals
        factors = []
        
        if 40 < rsi < 60:
            factors.append("• RSI is neutral (RSI = {:.1f})".format(rsi))
        elif rsi <= 40:
            factors.append("• RSI is trending low (RSI = {:.1f})".format(rsi))
        else:
            factors.append("• RSI is trending high (RSI = {:.1f})".format(rsi))
            
        if abs(macd_hist) < 0.2:
            factors.append("• MACD histogram is near zero, indicating potential trend shift")
        
        if 0.4 <= sentiment <= 0.6:
            factors.append("• Market sentiment is mixed/neutral")
        
        if 0.4 <= prediction <= 0.6:
            factors.append("• AI prediction shows uncertainty in future price direction")
        
        # Add fallback if no specific factors
        if not factors:
            factors.append("• Technical indicators and sentiment show conflicting signals")
            
        # Add all factors
        explanation += "\n".join(factors)
        
    return explanation

def determine_risk_level(rsi, sentiment, prediction, signal_strength):
    """Determine the risk level of a trade signal"""
    # Calculate risk factors
    
    # RSI extremes indicate higher risk
    rsi_risk = 0
    if rsi < 20 or rsi > 80:
        rsi_risk = 1  # High risk
    elif 30 <= rsi <= 70:
        rsi_risk = 0  # Low risk
    else:
        rsi_risk = 0.5  # Medium risk
    
    # Extreme sentiment indicates higher risk
    sentiment_risk = 0
    if sentiment < 0.2 or sentiment > 0.8:
        sentiment_risk = 1  # High risk
    elif 0.4 <= sentiment <= 0.6:
        sentiment_risk = 0  # Low risk
    else:
        sentiment_risk = 0.5  # Medium risk
    
    # Low prediction confidence indicates higher risk
    prediction_risk = 0
    if prediction < 0.6:
        prediction_risk = 1  # High risk
    elif prediction > 0.8:
        prediction_risk = 0  # Low risk
    else:
        prediction_risk = 0.5  # Medium risk
    
    # Low signal strength indicates higher risk
    strength_risk = 0
    if signal_strength < 0.5:
        strength_risk = 1  # High risk
    elif signal_strength > 0.8:
        strength_risk = 0  # Low risk
    else:
        strength_risk = 0.5  # Medium risk
    
    # Calculate overall risk score
    risk_score = (rsi_risk * 0.25 + sentiment_risk * 0.25 + 
                 prediction_risk * 0.25 + strength_risk * 0.25)
    
    # Determine risk level
    if risk_score > 0.7:
        return "High"
    elif risk_score > 0.3:
        return "Medium"
    else:
        return "Low"

def suggest_position_size(signal_strength, risk):
    """Suggest position size based on signal strength and risk level"""
    # Base position size on signal strength
    base_size = signal_strength * 10  # 0-10% range
    
    # Adjust for risk
    if risk == "High":
        adjusted_size = base_size * 0.5  # Reduce position size for high risk
    elif risk == "Medium":
        adjusted_size = base_size * 0.75  # Slightly reduce for medium risk
    else:
        adjusted_size = base_size  # Keep size for low risk
    
    # Format the output
    if adjusted_size < 2:
        return "< 2% of portfolio (high caution)"
    elif adjusted_size < 5:
        return "2-5% of portfolio"
    elif adjusted_size < 8:
        return "5-8% of portfolio"
    else:
        return "8-10% of portfolio"
