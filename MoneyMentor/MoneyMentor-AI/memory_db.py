"""
Memory-only implementation of database functions.
Used as a fallback when database connection is not available.
"""

import uuid
from datetime import datetime

# In-memory storage
storage = {
    'watchlists': [],
    'stocks': [],
    'signals': [],
    'notes': []
}

def _format_datetime(dt):
    """Format datetime object to string"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def _generate_id():
    """Generate a unique ID"""
    return str(uuid.uuid4())[:8]

# Watchlist functions
def get_all_watchlists():
    """Get all watchlists"""
    return storage['watchlists']

def get_watchlist(watchlist_id):
    """Get a watchlist by ID"""
    for watchlist in storage['watchlists']:
        if watchlist['id'] == watchlist_id:
            # Include associated stocks
            watchlist_stocks = []
            for stock in storage['stocks']:
                if stock['watchlist_id'] == watchlist_id:
                    watchlist_stocks.append(stock)
            
            watchlist_copy = watchlist.copy()
            watchlist_copy['stocks'] = watchlist_stocks
            return watchlist_copy
    return None

def create_watchlist(name):
    """Create a new watchlist"""
    watchlist_id = _generate_id()
    now = _format_datetime(datetime.now())
    
    watchlist = {
        'id': watchlist_id,
        'name': name,
        'created_at': now
    }
    
    storage['watchlists'].append(watchlist)
    return watchlist

def add_stock_to_watchlist(watchlist_id, symbol, company_name=None, notes=None):
    """Add a stock to a watchlist"""
    # Check if watchlist exists
    watchlist_exists = False
    for watchlist in storage['watchlists']:
        if watchlist['id'] == watchlist_id:
            watchlist_exists = True
            break
    
    if not watchlist_exists:
        return None
    
    # Check if stock already exists in watchlist
    for stock in storage['stocks']:
        if stock['watchlist_id'] == watchlist_id and stock['symbol'] == symbol:
            # Update notes if provided
            if notes:
                stock['notes'] = notes
            return stock
    
    # Add new stock
    stock_id = _generate_id()
    now = _format_datetime(datetime.now())
    
    stock = {
        'id': stock_id,
        'watchlist_id': watchlist_id,
        'symbol': symbol,
        'company_name': company_name,
        'added_at': now,
        'notes': notes
    }
    
    storage['stocks'].append(stock)
    return stock

def remove_stock_from_watchlist(watchlist_id, symbol):
    """Remove a stock from a watchlist"""
    for i, stock in enumerate(storage['stocks']):
        if stock['watchlist_id'] == watchlist_id and stock['symbol'] == symbol:
            storage['stocks'].pop(i)
            return True
    return False

def delete_watchlist(watchlist_id):
    """Delete a watchlist"""
    # Remove watchlist
    for i, watchlist in enumerate(storage['watchlists']):
        if watchlist['id'] == watchlist_id:
            storage['watchlists'].pop(i)
            
            # Remove associated stocks
            storage['stocks'] = [
                stock for stock in storage['stocks'] 
                if stock['watchlist_id'] != watchlist_id
            ]
            
            return True
    return False

# Trading Signal functions
def save_trading_signal(signal_data):
    """Save a new trading signal"""
    signal_id = _generate_id()
    now = _format_datetime(datetime.now())
    
    signal = {
        'id': signal_id,
        'generated_at': now,
        **signal_data
    }
    
    storage['signals'].append(signal)
    return signal

def get_signals_for_stock(symbol, limit=10):
    """Get recent trading signals for a stock"""
    signals = [
        signal for signal in storage['signals']
        if signal['symbol'] == symbol
    ]
    
    # Sort by generated_at in descending order
    signals.sort(key=lambda x: x['generated_at'], reverse=True)
    
    return signals[:limit]

# Analysis Notes functions
def save_analysis_note(note_data):
    """Save a new analysis note"""
    note_id = _generate_id()
    now = _format_datetime(datetime.now())
    
    note = {
        'id': note_id,
        'created_at': now,
        'updated_at': now,
        **note_data
    }
    
    storage['notes'].append(note)
    return note

def get_notes_for_stock(symbol):
    """Get analysis notes for a stock"""
    notes = [
        note for note in storage['notes']
        if note['symbol'] == symbol
    ]
    
    # Sort by updated_at in descending order
    notes.sort(key=lambda x: x['updated_at'], reverse=True)
    
    return notes

def update_analysis_note(note_id, title, content):
    """Update an analysis note"""
    for note in storage['notes']:
        if note['id'] == note_id:
            note['title'] = title
            note['content'] = content
            note['updated_at'] = _format_datetime(datetime.now())
            return note
    return None

def delete_analysis_note(note_id):
    """Delete an analysis note"""
    for i, note in enumerate(storage['notes']):
        if note['id'] == note_id:
            storage['notes'].pop(i)
            return True
    return False