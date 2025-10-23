import os
import random
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

# News API
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "demo")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Cache mechanism
news_cache = {}
news_cache_expiry = {}
NEWS_CACHE_DURATION = 3600  # 1 hour

def get_news_data(symbol, company_name):
    """Fetch news data for a stock"""
    # Check cache first
    cache_key = f"news_{symbol}"
    current_time = time.time()
    
    if cache_key in news_cache and current_time < news_cache_expiry.get(cache_key, 0):
        return news_cache[cache_key]
    
    # Only make real API calls if key is not "demo"
    if NEWS_API_KEY != "demo":
        try:
            # Create search query
            query = f"{symbol} OR {company_name} stock"
            
            # Calculate date range (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Make API request
            params = {
                'q': query,
                'from': from_date,
                'to': to_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': NEWS_API_KEY
            }
            
            response = requests.get(NEWS_API_URL, params=params)
            data = response.json()
            
            # Check if request was successful
            if response.status_code == 200 and 'articles' in data:
                # Cache the response
                news_cache[cache_key] = data['articles']
                news_cache_expiry[cache_key] = current_time + NEWS_CACHE_DURATION
                
                return data['articles']
            else:
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return generate_synthetic_news(symbol, company_name)
                
        except Exception as e:
            print(f"News API error: {e}")
            return generate_synthetic_news(symbol, company_name)
    else:
        # Use synthetic data for demo
        return generate_synthetic_news(symbol, company_name)

def analyze_sentiment(text):
    """
    Analyze sentiment of text
    Returns score between 0 (negative) and 1 (positive)
    """
    # In a real implementation, this would use NLTK, TextBlob, or a more advanced model
    # For demo, we'll use a simple keyword approach
    
    positive_words = [
        'gain', 'gains', 'profit', 'profits', 'positive', 'up', 'upward',
        'rise', 'rises', 'rising', 'rose', 'beat', 'beats', 'beating',
        'growth', 'grew', 'grow', 'growing', 'increase', 'increases',
        'increasing', 'increased', 'higher', 'strong', 'strength',
        'bullish', 'opportunity', 'opportunities', 'success', 'successful',
        'outperform', 'outperformed', 'outperformance', 'exceed', 'exceeded',
        'exceeding', 'exceeds', 'expectations', 'record', 'high', 'upgrade',
        'upgraded', 'buy', 'recommend', 'recommended'
    ]
    
    negative_words = [
        'loss', 'losses', 'lost', 'lose', 'losing', 'down', 'downward',
        'fall', 'falls', 'falling', 'fell', 'miss', 'missed', 'misses',
        'missing', 'decline', 'declines', 'declining', 'declined',
        'decrease', 'decreases', 'decreasing', 'decreased', 'lower',
        'weak', 'weakness', 'bearish', 'risk', 'risks', 'risky',
        'danger', 'dangerous', 'trouble', 'troubled', 'fail', 'fails',
        'failed', 'failing', 'underperform', 'underperformed',
        'underperformance', 'cut', 'cuts', 'cutting', 'disappoint',
        'disappoints', 'disappointed', 'disappointing', 'low', 'downgrade',
        'downgraded', 'sell', 'concern', 'concerns', 'concerning'
    ]
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Count occurrences of positive and negative words
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # Calculate sentiment score
    total = positive_count + negative_count
    if total == 0:
        return 0.5  # Neutral
    
    return positive_count / total

def get_stock_sentiment(symbol):
    """
    Get sentiment analysis for a stock from news articles
    Returns a dictionary with sentiment score and related information
    """
    company_names = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "META": "Facebook OR Meta",
        "TSLA": "Tesla",
        "NVDA": "NVIDIA",
        "JPM": "JPMorgan",
        "V": "Visa",
        "JNJ": "Johnson & Johnson"
    }
    
    company_name = company_names.get(symbol, symbol)
    
    # Get news articles
    articles = get_news_data(symbol, company_name)
    
    if not articles:
        return None
    
    # Analyze sentiment for each article
    sentiment_scores = []
    headlines = []
    
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        
        # Combine text for analysis
        text = f"{title} {description} {content}"
        
        # Get sentiment score
        score = analyze_sentiment(text)
        sentiment_scores.append(score)
        
        # Save headline with its sentiment
        headlines.append({
            'title': title,
            'source': article.get('source', {}).get('name', 'Unknown'),
            'date': article.get('publishedAt', ''),
            'sentiment': score
        })
    
    # Calculate average sentiment score
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5
    
    # Determine overall outlook
    if avg_sentiment > 0.7:
        outlook = "Very Bullish"
    elif avg_sentiment > 0.6:
        outlook = "Bullish"
    elif avg_sentiment > 0.45:
        outlook = "Neutral"
    elif avg_sentiment > 0.35:
        outlook = "Bearish"
    else:
        outlook = "Very Bearish"
    
    return {
        'score': avg_sentiment,
        'outlook': outlook,
        'sources_count': len(articles),
        'headlines': headlines
    }

def generate_synthetic_news(symbol, company_name):
    """Generate synthetic news data for demo purposes"""
    # Base sentiment bias for different stocks
    sentiment_bias = {
        "AAPL": 0.65,   # Slightly positive
        "MSFT": 0.7,    # Positive
        "GOOGL": 0.6,   # Slightly positive
        "AMZN": 0.55,   # Neutral-positive
        "META": 0.5,    # Neutral
        "TSLA": 0.45,   # Neutral-negative (more volatile)
        "NVDA": 0.75,   # Very positive
        "JPM": 0.6,     # Slightly positive
        "V": 0.65,      # Slightly positive
        "JNJ": 0.55     # Neutral-positive
    }
    
    base_sentiment = sentiment_bias.get(symbol, 0.5)
    
    # News sources
    sources = [
        "Financial Times", "Bloomberg", "Reuters", "CNBC", 
        "Wall Street Journal", "MarketWatch", "Barron's", 
        "The Motley Fool", "Seeking Alpha", "Yahoo Finance"
    ]
    
    # Generate random dates in the last week
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=random.uniform(0, 7))).isoformat() for _ in range(10)]
    
    # Headline templates with placeholders
    positive_headlines = [
        "{Company} Reports Strong Quarterly Earnings, Beats Expectations",
        "Analysts Upgrade {Company} Stock, Cite Growth Opportunities",
        "{Company} Announces Expansion into New Markets",
        "{Company} Stock Rallies as New Products Generate Excitement",
        "{Company} Raises Full-Year Guidance Following Strong Performance",
        "Institutional Investors Increase Stakes in {Company}",
        "{Company} CEO Optimistic About Company's Future Growth",
        "{Company} Announces Strategic Partnership with Leading Tech Firm",
        "{Company} Receives Regulatory Approval for New Product Line",
        "{Company} Stock Hits New 52-Week High on Trading Volume Surge"
    ]
    
    negative_headlines = [
        "{Company} Misses Earnings Expectations, Stock Falls",
        "Analysts Downgrade {Company} Citing Competitive Pressures",
        "{Company} Faces Legal Challenges Over Business Practices",
        "{Company} Stock Drops Amid Market Sector Weakness",
        "{Company} Cuts Guidance Amid Challenging Business Environment",
        "Investors Concerned About {Company}'s Growth Prospects",
        "{Company} CEO Acknowledges Challenges in Quarterly Call",
        "{Company} Loses Market Share to Competitors in Key Segment",
        "Regulatory Scrutiny Intensifies for {Company}'s Business Model",
        "{Company} Announces Restructuring Plan, Possible Layoffs"
    ]
    
    neutral_headlines = [
        "{Company} Reports Quarterly Results In Line With Expectations",
        "Analysts Maintain Neutral Stance on {Company} Stock",
        "{Company} Holds Annual Shareholder Meeting, Discusses Strategy",
        "{Company} Stock Moves with Broader Market Trends",
        "{Company} Announces Management Changes in Key Division",
        "{Company} to Present at Upcoming Industry Conference",
        "Investors Await More Clarity on {Company}'s Long-term Plans",
        "{Company} Maintains Current Product Pricing Despite Inflation",
        "{Company} Completes Previously Announced Acquisition",
        "Analyst Report Highlights Strengths and Challenges for {Company}"
    ]
    
    # Generate synthetic articles
    articles = []
    
    for i in range(10):
        # Determine sentiment for this article with random variation around base
        sentiment = max(0.1, min(0.9, base_sentiment + random.uniform(-0.2, 0.2)))
        
        # Choose headline template based on sentiment
        if sentiment > 0.6:
            headline = random.choice(positive_headlines)
        elif sentiment < 0.4:
            headline = random.choice(negative_headlines)
        else:
            headline = random.choice(neutral_headlines)
        
        # Replace placeholder
        headline = headline.replace("{Company}", company_name)
        
        # Generate article
        articles.append({
            'title': headline,
            'source': {'name': random.choice(sources)},
            'description': f"Analysis of {company_name}'s recent performance and market position.",
            'content': f"This is a synthetic article about {company_name} ({symbol}) for demonstration purposes.",
            'publishedAt': dates[i]
        })
    
    return articles
