"""
pages/1_📈_Overview.py

This page provides a high-level overview of trading performance,
focusing on Key Performance Indicators (KPIs) and the equity curve.
"""
import streamlit as st
import pandas as pd
import logging

# --- Assuming root-level modules are accessible ---
# This structure relies on app.py (or the main entry point) being in the root,
# and Python's import system being able to find these modules.
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, KPI_CONFIG, DEFAULT_KPI_DISPLAY_ORDER
    from components.kpi_display import KPIClusterDisplay
    from plotting import plot_equity_curve_and_drawdown # Direct plot call
    from utils.common_utils import display_custom_message
    # from services.analysis_service import AnalysisService # If complex plot generation is moved to service
except ImportError as e:
    st.error(f"Overview Page Error: Critical module import failed: {e}. Ensure the app structure is correct and all modules are available.")
    # Fallback or halt execution if essential modules are missing
    # For a real app, you might log this and show a persistent error.
    # For this generation, we'll assume imports work in the final integrated app.
    APP_TITLE = "TradingDashboard_Error" # Placeholder
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Overview Page: {e}", exc_info=True)
    st.stop() # Stop rendering this page if critical components are missing


# Get the main logger instance (configured in app.py)
logger = logging.getLogger(APP_TITLE)

def show_overview_page():
    """
    Renders the content for the Overview page.
    """
    st.title("📈 Performance Overview")
    logger.info("Rendering Overview Page.")

    # --- Check for necessary data in session state ---
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data in the main application to view the overview.", "info")
        logger.info("OverviewPage: No filtered_data in session_state.")
        return

    if 'kpi_results' not in st.session_state or st.session_state.kpi_results is None:
        display_custom_message("KPI results are not available. Please ensure data is processed.", "warning")
        logger.info("OverviewPage: No kpi_results in session_state.")
        return
    
    if 'error' in st.session_state.kpi_results:
        display_custom_message(f"Error in KPI calculation: {st.session_state.kpi_results['error']}", "error")
        return

    filtered_df = st.session_state.filtered_data
    kpi_results = st.session_state.kpi_results
    kpi_confidence_intervals = st.session_state.get('kpi_confidence_intervals', {})
    plot_theme = st.session_state.get('current_theme', 'dark') # Get theme from session state

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Cannot display overview.", "info")
        logger.info("OverviewPage: filtered_df is empty.")
        return

    # --- Display KPIs using KPIClusterDisplay component ---
    st.subheader("Key Performance Indicators")
    try:
        kpi_cluster = KPIClusterDisplay(
            kpi_results=kpi_results,
            kpi_definitions=KPI_CONFIG, # Imported from config
            kpi_order=DEFAULT_KPI_DISPLAY_ORDER, # Imported from config
            kpi_confidence_intervals=kpi_confidence_intervals,
            cols_per_row=4 # Or make this configurable via session_state/config
        )
        kpi_cluster.render()
    except Exception as e:
        logger.error(f"Error rendering KPI cluster: {e}", exc_info=True)
        display_custom_message(f"An error occurred while displaying KPIs: {e}", "error")


    st.markdown("---")

    # --- Display Equity Curve and Drawdown ---
    st.subheader("Equity Curve & Drawdown")
    try:
        date_col = EXPECTED_COLUMNS.get('date', 'date')
        cum_pnl_col = 'cumulative_pnl' # This should be consistently named from data_processing
        drawdown_pct_col_name = 'drawdown_pct' # Expected from calculations or data_processing

        # Ensure necessary columns are present
        if date_col not in filtered_df.columns:
            display_custom_message(f"Date column ('{date_col}') not found in data for equity curve.", "error")
            return
        if cum_pnl_col not in filtered_df.columns:
            # Attempt to calculate if missing (basic fallback)
            pnl_col_orig = EXPECTED_COLUMNS.get('pnl', 'pnl')
            if pnl_col_orig in filtered_df.columns:
                filtered_df[cum_pnl_col] = filtered_df[pnl_col_orig].cumsum()
                logger.info("OverviewPage: 'cumulative_pnl' was missing, calculated on-the-fly.")
            else:
                display_custom_message(f"Cumulative PnL column ('{cum_pnl_col}') and PnL column ('{pnl_col_orig}') not found.", "error")
                return

        # Determine drawdown column for plotting
        actual_drawdown_col = None
        if drawdown_pct_col_name in filtered_df.columns:
            actual_drawdown_col = drawdown_pct_col_name
        elif 'drawdown_pct_for_plot' in filtered_df.columns: # Fallback if app.py added it
            actual_drawdown_col = 'drawdown_pct_for_plot'
        else:
            # If drawdown percentage is not pre-calculated, we might need to calculate it here
            # or ensure it's part of the main data processing pipeline.
            # For simplicity, we assume it might be missing and the plot function handles it.
            logger.warning(f"OverviewPage: Drawdown percentage column ('{drawdown_pct_col_name}') not found. Plot may not show drawdown.")


        equity_fig = plot_equity_curve_and_drawdown(
            filtered_df,
            date_col=date_col,
            cumulative_pnl_col=cum_pnl_col,
            drawdown_pct_col=actual_drawdown_col, # Pass the determined column name
            theme=plot_theme
        )
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True)
        else:
            display_custom_message("Could not generate the equity curve and drawdown chart. Data might be insufficient or in an unexpected format.", "warning")
    except Exception as e:
        logger.error(f"Error displaying equity curve: {e}", exc_info=True)
        display_custom_message(f"An error occurred while displaying the equity curve: {e}", "error")

# --- Main execution for the page ---
# This ensures that the page's content is rendered when the page is selected.
if __name__ == "__main__":
    # This block is useful for testing the page directly if needed,
    # but Streamlit runs it by importing and calling functions or rendering top-down.
    # For multi-page apps, Streamlit executes the script from top to bottom.
    # We can put the main page rendering logic in a function and call it.
    if 'app_initialized' not in st.session_state: # Basic check if main app.py has run
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
        # Potentially initialize minimal session state for standalone testing:
        # st.session_state.filtered_data = pd.DataFrame(...)
        # st.session_state.kpi_results = {...}
        # st.session_state.current_theme = 'dark'
        # st.session_state.app_initialized = True # Mark as initialized for this test run
    
    # The actual rendering logic for the page:
    show_overview_page()

