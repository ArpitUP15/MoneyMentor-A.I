import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Get database URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL')

# Create SQLAlchemy engine
# Handle potential connection errors gracefully
try:
    engine = create_engine(DATABASE_URL) if DATABASE_URL else None
except Exception as e:
    print(f"Database connection error: {e}")
    engine = None

# Create base class for models
Base = declarative_base()

# Define models
class Watchlist(Base):
    """Watchlist model for storing user watchlists"""
    __tablename__ = 'watchlists'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    stocks = relationship("WatchlistStock", back_populates="watchlist", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'stocks': [stock.to_dict() for stock in self.stocks]
        }

class WatchlistStock(Base):
    """Model for stocks in watchlists"""
    __tablename__ = 'watchlist_stocks'
    
    id = Column(Integer, primary_key=True)
    watchlist_id = Column(Integer, ForeignKey('watchlists.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    company_name = Column(String(200))
    added_at = Column(DateTime, default=datetime.now)
    notes = Column(Text)
    
    # Relationships
    watchlist = relationship("Watchlist", back_populates="stocks")
    
    def to_dict(self):
        return {
            'id': self.id,
            'watchlist_id': self.watchlist_id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'added_at': self.added_at.strftime('%Y-%m-%d %H:%M:%S'),
            'notes': self.notes
        }

class TradingSignal(Base):
    """Model for storing AI-generated trading signals"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)  # 'buy', 'sell', 'hold'
    confidence = Column(Float, nullable=False)
    generated_at = Column(DateTime, default=datetime.now)
    price_at_signal = Column(Float)
    rsi_value = Column(Float)
    macd_value = Column(Float)
    sentiment_score = Column(Float)
    is_backtested = Column(Boolean, default=False)
    backtest_result = Column(Float)  # Profit/loss percentage if backtested
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'generated_at': self.generated_at.strftime('%Y-%m-%d %H:%M:%S'),
            'price_at_signal': self.price_at_signal,
            'rsi_value': self.rsi_value,
            'macd_value': self.macd_value,
            'sentiment_score': self.sentiment_score,
            'is_backtested': self.is_backtested,
            'backtest_result': self.backtest_result
        }

class AnalysisNote(Base):
    """Model for storing user analysis notes"""
    __tablename__ = 'analysis_notes'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }

# Create all tables if engine is available
if engine:
    try:
        Base.metadata.create_all(engine)
        # Create session
        Session = sessionmaker(bind=engine)
    except Exception as e:
        print(f"Error creating tables: {e}")
        Session = None
else:
    print("Warning: Database engine not available, using memory-only storage")
    Session = None

# In-memory fallback storage (used when database connection is not available)
memory_storage = {
    'watchlists': [],
    'signals': [],
    'notes': []
}

# Database functions
def get_session():
    """Get a database session"""
    if Session:
        return Session()
    return None

# Import memory db for fallback
import memory_db

# Watchlist functions
def get_all_watchlists():
    """Get all watchlists"""
    session = get_session()
    
    # Use database if available
    if session:
        try:
            watchlists = session.query(Watchlist).all()
            return [watchlist.to_dict() for watchlist in watchlists]
        finally:
            session.close()
    
    # Fallback to memory db if database not available
    return memory_db.get_all_watchlists()

def get_watchlist(watchlist_id):
    """Get a watchlist by ID"""
    session = get_session()
    
    if session:
        try:
            watchlist = session.query(Watchlist).filter_by(id=watchlist_id).first()
            return watchlist.to_dict() if watchlist else None
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.get_watchlist(watchlist_id)

def create_watchlist(name):
    """Create a new watchlist"""
    session = get_session()
    
    if session:
        try:
            watchlist = Watchlist(name=name)
            session.add(watchlist)
            session.commit()
            return watchlist.to_dict()
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.create_watchlist(name)

def add_stock_to_watchlist(watchlist_id, symbol, company_name=None, notes=None):
    """Add a stock to a watchlist"""
    session = get_session()
    
    if session:
        try:
            # Check if stock already exists in watchlist
            existing = session.query(WatchlistStock).filter_by(
                watchlist_id=watchlist_id, symbol=symbol
            ).first()
            
            if existing:
                # Update notes if provided
                if notes:
                    existing.notes = notes
                    session.commit()
                return existing.to_dict()
            
            # Add new stock
            stock = WatchlistStock(
                watchlist_id=watchlist_id,
                symbol=symbol,
                company_name=company_name,
                notes=notes
            )
            session.add(stock)
            session.commit()
            return stock.to_dict()
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.add_stock_to_watchlist(watchlist_id, symbol, company_name, notes)

def remove_stock_from_watchlist(watchlist_id, symbol):
    """Remove a stock from a watchlist"""
    session = get_session()
    
    if session:
        try:
            stock = session.query(WatchlistStock).filter_by(
                watchlist_id=watchlist_id, symbol=symbol
            ).first()
            
            if stock:
                session.delete(stock)
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.remove_stock_from_watchlist(watchlist_id, symbol)

def delete_watchlist(watchlist_id):
    """Delete a watchlist"""
    session = get_session()
    
    if session:
        try:
            watchlist = session.query(Watchlist).filter_by(id=watchlist_id).first()
            if watchlist:
                session.delete(watchlist)
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.delete_watchlist(watchlist_id)

# Trading Signal functions
def save_trading_signal(signal_data):
    """Save a new trading signal"""
    session = get_session()
    
    if session:
        try:
            signal = TradingSignal(**signal_data)
            session.add(signal)
            session.commit()
            return signal.to_dict()
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.save_trading_signal(signal_data)

def get_signals_for_stock(symbol, limit=10):
    """Get recent trading signals for a stock"""
    session = get_session()
    
    if session:
        try:
            signals = session.query(TradingSignal).filter_by(
                symbol=symbol
            ).order_by(TradingSignal.generated_at.desc()).limit(limit).all()
            
            return [signal.to_dict() for signal in signals]
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.get_signals_for_stock(symbol, limit)

# Analysis Notes functions
def save_analysis_note(note_data):
    """Save a new analysis note"""
    session = get_session()
    
    if session:
        try:
            note = AnalysisNote(**note_data)
            session.add(note)
            session.commit()
            return note.to_dict()
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.save_analysis_note(note_data)

def get_notes_for_stock(symbol):
    """Get analysis notes for a stock"""
    session = get_session()
    
    if session:
        try:
            notes = session.query(AnalysisNote).filter_by(
                symbol=symbol
            ).order_by(AnalysisNote.updated_at.desc()).all()
            
            return [note.to_dict() for note in notes]
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.get_notes_for_stock(symbol)

def update_analysis_note(note_id, title, content):
    """Update an analysis note"""
    session = get_session()
    
    if session:
        try:
            note = session.query(AnalysisNote).filter_by(id=note_id).first()
            if note:
                note.title = title
                note.content = content
                session.commit()
                return note.to_dict()
            return None
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.update_analysis_note(note_id, title, content)

def delete_analysis_note(note_id):
    """Delete an analysis note"""
    session = get_session()
    
    if session:
        try:
            note = session.query(AnalysisNote).filter_by(id=note_id).first()
            if note:
                session.delete(note)
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    # Fallback to memory db
    return memory_db.delete_analysis_note(note_id)