"""
pages/8_📋_Data_View.py

This page provides a view of the currently filtered trade data,
allowing users to inspect the raw numbers and download the dataset.
It utilizes the DataTableDisplay component.
"""
import streamlit as st
import pandas as pd
import logging

# --- Assuming root-level modules are accessible ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS
    from components.data_table_display import DataTableDisplay
    from utils.common_utils import display_custom_message
except ImportError as e:
    st.error(f"Data View Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error"
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Data View Page: {e}", exc_info=True)
    st.stop()

logger = logging.getLogger(APP_TITLE)

def show_data_view_page():
    """
    Renders the content for the Filtered Data View page.
    """
    st.title("📋 Filtered Trade Data Log")
    logger.info("Rendering Data View Page.")

    # --- Check for necessary data in session state ---
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data in the main application to view the data log.", "info")
        return

    filtered_df = st.session_state.filtered_data

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. The data log is empty.", "info")
        return

    # --- Display Data using DataTableDisplay component ---
    st.markdown("View and download the trade data based on the currently applied filters.")

    try:
        # Define column configurations for better display in the data table
        # This can be expanded based on all expected columns and their desired formatting
        column_configs = {}
        date_col = EXPECTED_COLUMNS.get('date')
        pnl_col = EXPECTED_COLUMNS.get('pnl')
        risk_col = EXPECTED_COLUMNS.get('risk')
        # ... add other important columns from EXPECTED_COLUMNS

        if date_col and date_col in filtered_df.columns:
            column_configs[date_col] = st.column_config.DatetimeColumn(
                "Timestamp", format="YYYY-MM-DD HH:mm:ss", help="Date and time of the trade"
            )
        if pnl_col and pnl_col in filtered_df.columns:
            column_configs[pnl_col] = st.column_config.NumberColumn(
                "PnL", format="$%.2f", help="Profit or Loss"
            )
        if risk_col and risk_col in filtered_df.columns:
            column_configs[risk_col] = st.column_config.NumberColumn(
                "Risk", format="$%.2f", help="Amount risked"
            )
        
        # Example for a boolean 'win' column if it exists
        if 'win' in filtered_df.columns:
            column_configs['win'] = st.column_config.CheckboxColumn("Win Trade?", help="True if PnL > 0")


        data_table = DataTableDisplay(
            dataframe=filtered_df,
            title=None, # Title is already set by st.title for the page
            column_config=column_configs,
            height=600, # Make it reasonably tall
            download_button=True,
            download_file_name="filtered_trading_journal.csv"
        )
        data_table.render()
    except Exception as e:
        logger.error(f"Error rendering data table on Data View page: {e}", exc_info=True)
        display_custom_message(f"An error occurred while displaying the data table: {e}", "error")

# --- Main execution for the page ---
if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_data_view_page()
