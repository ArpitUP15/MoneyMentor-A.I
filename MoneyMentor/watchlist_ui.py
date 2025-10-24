import streamlit as st
import pandas as pd
from database import (
    get_all_watchlists,
    get_watchlist,
    create_watchlist,
    add_stock_to_watchlist,
    remove_stock_from_watchlist,
    delete_watchlist
)
from yf_stock_data import get_stock_quote, get_company_name

def render_watchlist_sidebar():
    """Render watchlist management in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Watchlists")
    
    # Get all watchlists
    watchlists = get_all_watchlists()
    
    # Create new watchlist input
    with st.sidebar.form("new_watchlist_form"):
        st.write("Create New Watchlist")
        new_watchlist_name = st.text_input("Watchlist Name", key="new_watchlist_name")
        submitted = st.form_submit_button("Create")
        
        if submitted and new_watchlist_name:
            create_watchlist(new_watchlist_name)
            st.success(f"Created watchlist: {new_watchlist_name}")
            st.rerun()
    
    # Display existing watchlists
    if watchlists:
        st.sidebar.write("Your Watchlists:")
        for watchlist in watchlists:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(f"📋 {watchlist['name']}", key=f"watchlist_{watchlist['id']}"):
                    st.session_state.active_watchlist = watchlist['id']
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{watchlist['id']}"):
                    delete_watchlist(watchlist['id'])
                    if 'active_watchlist' in st.session_state and st.session_state.active_watchlist == watchlist['id']:
                        st.session_state.active_watchlist = None
                    st.rerun()
    else:
        st.sidebar.info("No watchlists found. Create one to get started!")

def render_watchlist_main():
    """Render the main watchlist content area"""
    if 'active_watchlist' not in st.session_state or not st.session_state.active_watchlist:
        st.info("Select a watchlist from the sidebar or create a new one to get started!")
        return
    
    # Get the active watchlist
    watchlist_id = st.session_state.active_watchlist
    watchlist = get_watchlist(watchlist_id)
    
    if not watchlist:
        st.error("Watchlist not found!")
        st.session_state.active_watchlist = None
        return
    
    # Display watchlist header
    st.header(f"Watchlist: {watchlist['name']}")
    
    # Form to add a stock to watchlist
    with st.form("add_stock_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            stock_symbol = st.text_input("Symbol", max_chars=10).upper()
        with col2:
            st.write("&nbsp;")
            add_button = st.form_submit_button("Add to Watchlist")
        
        if add_button and stock_symbol:
            # Get company name if possible
            company_name = get_company_name(stock_symbol)
            add_stock_to_watchlist(watchlist_id, stock_symbol, company_name)
            st.success(f"Added {stock_symbol} to watchlist!")
            st.rerun()
    
    # Display stocks in the watchlist
    if watchlist.get('stocks'):
        stocks_data = []
        for stock in watchlist['stocks']:
            # Get current data
            quote = get_stock_quote(stock['symbol'])
            
            # If quote is available, add to stocks data
            if quote:
                stocks_data.append({
                    'Symbol': stock['symbol'],
                    'Company': quote['name'],
                    'Price': f"${quote['price']:.2f}",
                    'Change': f"{quote['price_change']:.2f} ({quote['price_change_percent']:.2f}%)",
                    'Volume': quote['volume'],
                    'Market Cap': quote.get('market_cap', 'N/A'),
                    'P/E': quote.get('pe', 'N/A'),
                    'Added On': stock['added_at'],
                    'Notes': stock['notes'] or ''
                })
        
        if stocks_data:
            # Create DataFrame
            df = pd.DataFrame(stocks_data)
            
            # Format the change column with colors
            def color_change(val):
                change_val = float(val.split()[0])
                return 'color: green' if change_val > 0 else 'color: red'
            
            # Display stocks table
            st.dataframe(
                df.style.applymap(color_change, subset=['Change']),
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Company": st.column_config.TextColumn("Company", width="medium"),
                    "Price": st.column_config.TextColumn("Price", width="small"),
                    "Change": st.column_config.TextColumn("Change", width="medium"),
                    "Volume": st.column_config.NumberColumn("Volume", format="%d"),
                    "Market Cap": st.column_config.TextColumn("Market Cap", width="medium"),
                    "P/E": st.column_config.TextColumn("P/E", width="small"),
                    "Added On": st.column_config.DatetimeColumn("Added On", format="MMM D, YYYY"),
                    "Notes": st.column_config.TextColumn("Notes")
                }
            )
            
            # Buttons to view stock details or remove from watchlist
            for stock in watchlist['stocks']:
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(f"View {stock['symbol']} details", key=f"view_{stock['symbol']}"):
                        st.session_state.selected_stock = stock['symbol']
                        st.rerun()
                with col2:
                    if st.button("Remove", key=f"remove_{stock['symbol']}"):
                        remove_stock_from_watchlist(watchlist_id, stock['symbol'])
                        st.success(f"Removed {stock['symbol']} from watchlist!")
                        st.rerun()
    else:
        st.info("No stocks in this watchlist yet. Add some stocks to get started!")