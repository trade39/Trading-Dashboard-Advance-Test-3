"""
pages/9_📝_Trade_Notes.py

This page provides an interface for viewing, searching, and filtering
qualitative trade notes associated with individual trades.
It utilizes the NotesViewerComponent.
"""
import streamlit as st
import pandas as pd
import logging

# --- Assuming root-level modules are accessible ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS
    from components.notes_viewer import NotesViewerComponent
    from utils.common_utils import display_custom_message
except ImportError as e:
    st.error(f"Trade Notes Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error"
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Trade Notes Page: {e}", exc_info=True)
    st.stop()

logger = logging.getLogger(APP_TITLE)

def show_trade_notes_page():
    st.title("📝 Trade Notes Viewer")
    logger.info("Rendering Trade Notes Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data to view trade notes.", "info")
        return

    filtered_df = st.session_state.filtered_data
    if filtered_df.empty:
        display_custom_message("No data matches current filters. Cannot display trade notes.", "info")
        return

    date_col = EXPECTED_COLUMNS.get('date')
    notes_col_name = EXPECTED_COLUMNS.get('notes') # Get the configured name for notes
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    symbol_col = EXPECTED_COLUMNS.get('symbol')

    # Explicitly check if the configured 'notes' column exists in the DataFrame
    if not notes_col_name or notes_col_name not in filtered_df.columns:
        display_custom_message(f"The configured notes column ('{notes_col_name or 'notes'}') was not found in your uploaded CSV. Please check your `config.py` or CSV file.", "error")
        logger.error(f"TradeNotesPage: Notes column '{notes_col_name}' not found in filtered_df. Available columns: {filtered_df.columns.tolist()}")
        # You could list available columns to help the user: st.write(f"Available columns: {filtered_df.columns.tolist()}")
        return
    
    # Check other essential columns for the component that might have been renamed or are missing
    essential_cols_for_component = {'date': date_col, 'notes': notes_col_name, 'pnl': pnl_col}
    missing_display_cols = [name for name, col_key in essential_cols_for_component.items() if not (col_key and col_key in filtered_df.columns)]
    if missing_display_cols:
        display_custom_message(f"Essential columns for notes display ({', '.join(missing_display_cols)}) are missing from the data after processing.", "error")
        return

    try:
        notes_component = NotesViewerComponent(
            notes_dataframe=filtered_df.copy(), # Pass a copy to avoid modifying session state df
            date_col=date_col,
            notes_col=notes_col_name, # Use the actual notes column name from config
            pnl_col=pnl_col,
            symbol_col=symbol_col if symbol_col and symbol_col in filtered_df.columns else None,
            default_sort_by_display="Date",
            default_sort_ascending=False
        )
        notes_component.render()
    except Exception as e:
        logger.error(f"Error instantiating or rendering NotesViewerComponent: {e}", exc_info=True)
        display_custom_message(f"An error occurred while displaying trade notes: {e}", "error")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_trade_notes_page()
