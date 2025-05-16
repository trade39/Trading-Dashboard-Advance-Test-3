"""
app.py - Main Entry Point for Multi-Page Trading Performance Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
import datetime # For date operations

# --- Utility Modules ---
try:
    from utils.logger import setup_logger
    from utils.common_utils import load_css, display_custom_message
except ImportError as e:
    st.error(f"Fatal Error: Could not import utility modules. App cannot start. Details: {e}")
    # Basic logging if main logger setup fails
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Fatal Error importing utils: {e}", exc_info=True)
    st.stop()

# --- Component Modules ---
try:
    from components.sidebar_manager import SidebarManager
except ImportError as e:
    st.error(f"Fatal Error: Could not import SidebarManager. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing SidebarManager: {e}", exc_info=True)
    st.stop()

# --- Service Modules ---
try:
    from services.data_service import DataService
    from services.analysis_service import AnalysisService
except ImportError as e:
    st.error(f"Fatal Error: Could not import service modules. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing services: {e}", exc_info=True)
    st.stop()

# --- Core Application Modules (Configs) ---
try:
    from config import (
        APP_TITLE, EXPECTED_COLUMNS, RISK_FREE_RATE,
        LOG_FILE, LOG_LEVEL, LOG_FORMAT, DEFAULT_BENCHMARK_TICKER # Added DEFAULT_BENCHMARK_TICKER
    )
except ImportError as e:
    st.error(f"Fatal Error: Could not import configuration. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing config: {e}", exc_info=True)
    # Define fallbacks for essential config items if import fails, to allow logger setup
    APP_TITLE = "TradingDashboard_Error"
    LOG_FILE = "logs/error_app.log"
    LOG_LEVEL = "ERROR"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    RISK_FREE_RATE = 0.02
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}
    DEFAULT_BENCHMARK_TICKER = "SPY"
    # st.stop() # Don't stop here, allow logger to be set up for further error reporting

# --- Initialize Centralized Logger ---
# This must happen after config constants for logging are potentially defined/fallback.
logger = setup_logger(
    logger_name=APP_TITLE, log_file=LOG_FILE, level=LOG_LEVEL, log_format=LOG_FORMAT
)
logger.info(f"Application '{APP_TITLE}' starting. Logger initialized.")
logger.debug(f"CWD at app start: {os.getcwd()}, sys.path: {sys.path}")


# --- Page Configuration (Global for all pages) ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈", # Changed to a more generic chart icon
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/trading-dashboard-tmh', # Replace with your repo
        'Report a bug': "https://github.com/your-repo/trading-dashboard-tmh/issues", # Replace
        'About': f"## {APP_TITLE}\n\nA comprehensive dashboard for trading performance analysis."
    }
)
logger.debug("Streamlit page_config set.")

# --- Load Custom CSS ---
try:
    load_css("style.css") # Assuming style.css is in the root directory
    logger.debug("Custom CSS loaded.")
except Exception as e:
    logger.error(f"Failed to load style.css: {e}", exc_info=True)
    st.warning("Could not load custom styles. The app may not appear as intended.")

# --- Initialize Session State ---
# Define default values for session state keys
default_session_state = {
    'app_initialized': True,
    'processed_data': None,
    'filtered_data': None,
    'kpi_results': None,
    'kpi_confidence_intervals': {},
    'risk_free_rate': RISK_FREE_RATE, # From config.py
    'current_theme': "dark", # Default theme
    'uploaded_file_name': None,
    'last_processed_file_id': None, # To track if the same file is re-uploaded
    'last_filtered_data_shape': None, # To track if filtered data has changed
    'sidebar_filters': None, # Stores the dictionary of filter values from SidebarManager
    'active_tab': "📈 Overview", # Example for main page tab state if needed
    'selected_benchmark_ticker': DEFAULT_BENCHMARK_TICKER, # Initialize with default
    'selected_benchmark_display_name': "S&P 500 (SPY)", # Default display name
    'benchmark_daily_returns': None, # Stores pd.Series of benchmark returns
    'initial_capital': 100000.0 # Default initial capital
}

# Initialize session state if keys don't exist
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value
logger.debug("Global session state initialized/checked.")


# --- Instantiate Services ---
data_service = DataService()
analysis_service = AnalysisService()
logger.debug("DataService and AnalysisService instantiated.")


# --- Sidebar Rendering and Filter Management ---
st.sidebar.title(f"📊 {APP_TITLE} Controls")
st.sidebar.markdown("---")

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Trading Journal (CSV)", type=["csv"],
    help=f"Expected columns include: {', '.join(EXPECTED_COLUMNS.values())}",
    key="app_wide_file_uploader" # Unique key for the widget
)

# Instantiate SidebarManager with current processed_data (can be None initially)
# SidebarManager will now also handle benchmark selection and initial capital
sidebar_manager = SidebarManager(st.session_state.processed_data)
current_sidebar_filters = sidebar_manager.render_sidebar_controls() # This call renders the sidebar

# Store the returned filter values (including benchmark and initial capital)
st.session_state.sidebar_filters = current_sidebar_filters

# Update session state for RFR, benchmark, and initial capital if changed in sidebar
if current_sidebar_filters:
    if st.session_state.risk_free_rate != current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE):
        st.session_state.risk_free_rate = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
        logger.info(f"Global risk-free rate updated to: {st.session_state.risk_free_rate:.4f}")
        st.session_state.kpi_results = None # Force KPI recalculation

    if st.session_state.selected_benchmark_ticker != current_sidebar_filters.get('selected_benchmark_ticker', ""):
        st.session_state.selected_benchmark_ticker = current_sidebar_filters.get('selected_benchmark_ticker', "")
        # Find display name from config for consistency if needed by other parts of app
        st.session_state.selected_benchmark_display_name = next(
            (name for name, ticker in default_session_state.get('AVAILABLE_BENCHMARKS', {}).items() 
             if ticker == st.session_state.selected_benchmark_ticker), "None"
        )
        logger.info(f"Benchmark ticker updated to: {st.session_state.selected_benchmark_ticker}")
        st.session_state.benchmark_daily_returns = None # Force refetch
        st.session_state.kpi_results = None # Force KPI recalculation

    if st.session_state.initial_capital != current_sidebar_filters.get('initial_capital', 100000.0):
        st.session_state.initial_capital = current_sidebar_filters.get('initial_capital', 100000.0)
        logger.info(f"Initial capital updated to: {st.session_state.initial_capital:.2f}")
        st.session_state.kpi_results = None # Force KPI recalculation if PnL is absolute


# --- Data Loading and Processing ---
if uploaded_file is not None:
    # Create a unique ID for the current file to avoid reprocessing if it hasn't changed
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
    if st.session_state.last_processed_file_id != current_file_id or st.session_state.processed_data is None:
        logger.info(f"File '{uploaded_file.name}' (ID: {current_file_id}) selected. Initiating processing.")
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            st.session_state.processed_data = data_service.get_processed_trading_data(uploaded_file)
        
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.last_processed_file_id = current_file_id
        # Reset dependent states on new file upload
        st.session_state.kpi_results = None
        st.session_state.kpi_confidence_intervals = {}
        st.session_state.filtered_data = st.session_state.processed_data # Initially, filtered is same as processed
        st.session_state.benchmark_daily_returns = None # Reset benchmark data on new file

        if st.session_state.processed_data is not None:
            logger.info(f"DataService processed '{uploaded_file.name}'. Shape: {st.session_state.processed_data.shape}")
            display_custom_message(f"Successfully processed '{uploaded_file.name}'. Navigate pages to see analysis.", "success", icon="✅")
            # Update sidebar manager with new data if it affects filter options (e.g., date range)
            # This might require re-rendering sidebar if options change dynamically, or handle in SidebarManager
        else:
            logger.error(f"DataService failed to process file: {uploaded_file.name}.")
            display_custom_message(f"Failed to process '{uploaded_file.name}'. Check logs and file format.", "error")
            # Clear potentially problematic states
            st.session_state.processed_data = None 
            st.session_state.filtered_data = None
            st.session_state.kpi_results = None
            st.session_state.kpi_confidence_intervals = {}
            st.session_state.benchmark_daily_returns = None


# --- Data Filtering (Applied after data is processed or filters change) ---
if st.session_state.processed_data is not None and st.session_state.sidebar_filters:
    # Check if filters have changed or if filtered_data is not yet set
    if st.session_state.filtered_data is None or \
       st.session_state.get('last_applied_filters') != st.session_state.sidebar_filters:
        
        logger.info("Applying global filters via DataService...")
        st.session_state.filtered_data = data_service.filter_data(
            st.session_state.processed_data,
            st.session_state.sidebar_filters, # Pass the full filter dict
            column_map=EXPECTED_COLUMNS # Ensure DataService uses the correct column map
        )
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy() # Store applied filters
        logger.debug(f"Data filtered. New shape: {st.session_state.filtered_data.shape if st.session_state.filtered_data is not None else 'None'}")
        # Reset KPIs and benchmark data as filtered data has changed
        st.session_state.kpi_results = None
        st.session_state.kpi_confidence_intervals = {}
        st.session_state.benchmark_daily_returns = None # Force refetch if benchmark is selected


# --- Benchmark Data Fetching (Applied after data is filtered or benchmark selection changes) ---
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')
    if selected_ticker and selected_ticker != "": # A benchmark is selected
        # Check if benchmark data needs to be fetched/refetched
        # This condition ensures fetching only if ticker changes or data is not yet loaded
        # or if filtered_data (which defines date range) has changed.
        if st.session_state.benchmark_daily_returns is None or \
           st.session_state.get('last_fetched_benchmark_ticker') != selected_ticker or \
           st.session_state.get('last_benchmark_data_filter_shape') != st.session_state.filtered_data.shape:

            date_col_name = EXPECTED_COLUMNS.get('date')
            if date_col_name and date_col_name in st.session_state.filtered_data.columns:
                min_date = st.session_state.filtered_data[date_col_name].min()
                max_date = st.session_state.filtered_data[date_col_name].max()
                
                if pd.notna(min_date) and pd.notna(max_date):
                    logger.info(f"Fetching benchmark data for {selected_ticker} from {min_date} to {max_date}.")
                    st.session_state.benchmark_daily_returns = analysis_service.get_benchmark_data(
                        selected_ticker, min_date, max_date
                    )
                    st.session_state.last_fetched_benchmark_ticker = selected_ticker
                    st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape

                    if st.session_state.benchmark_daily_returns is None:
                        display_custom_message(f"Could not fetch benchmark data for {selected_ticker}. Proceeding without benchmark comparison.", "warning")
                    else:
                        logger.info(f"Benchmark data for {selected_ticker} fetched successfully.")
                else:
                    logger.warning("Min/max dates from filtered_data are NaT, cannot fetch benchmark data.")
                    st.session_state.benchmark_daily_returns = None # Ensure it's None
            else:
                logger.warning(f"Date column '{date_col_name}' not found in filtered_data for benchmark date range.")
                st.session_state.benchmark_daily_returns = None
            st.session_state.kpi_results = None # Force KPI recalculation after benchmark fetch attempt
    elif st.session_state.benchmark_daily_returns is not None : # If benchmark was "None" or deselected
        logger.info("No benchmark selected or deselected. Clearing benchmark data.")
        st.session_state.benchmark_daily_returns = None
        st.session_state.kpi_results = None # Force KPI recalculation


# --- KPI Calculation (Applied if filtered data exists and KPIs are not yet calculated for current state) ---
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    # Check if KPIs need recalculation
    # This happens if: kpi_results is None, or relevant inputs (RFR, benchmark, initial_capital, filtered_data shape) changed
    
    # Construct a unique ID for the current state affecting KPIs
    current_kpi_calc_state_id = (
        st.session_state.filtered_data.shape,
        st.session_state.risk_free_rate,
        st.session_state.initial_capital,
        st.session_state.selected_benchmark_ticker, # Ticker itself
        # Hash of benchmark data if it exists, to detect changes in actual benchmark values
        pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns, index=True).sum() if st.session_state.benchmark_daily_returns is not None else None
    )

    if st.session_state.kpi_results is None or \
       st.session_state.get('last_kpi_calc_state_id') != current_kpi_calc_state_id:
        
        logger.info("Recalculating global KPIs and CIs via AnalysisService...")
        with st.spinner("Calculating performance metrics & CIs..."):
            # Pass benchmark returns and initial capital to the service
            kpi_service_result = analysis_service.get_core_kpis(
                st.session_state.filtered_data,
                st.session_state.risk_free_rate,
                benchmark_daily_returns=st.session_state.get('benchmark_daily_returns'),
                initial_capital=st.session_state.get('initial_capital')
            )
            
            if kpi_service_result and 'error' not in kpi_service_result:
                st.session_state.kpi_results = kpi_service_result
                st.session_state.last_kpi_calc_state_id = current_kpi_calc_state_id # Store the state ID
                logger.info("Global KPIs calculated.")

                # Bootstrap CIs (consider if these also need benchmark context, currently they don't)
                ci_service_result = analysis_service.get_bootstrapped_kpi_cis(
                    st.session_state.filtered_data,
                    kpis_to_bootstrap=['avg_trade_pnl', 'win_rate', 'sharpe_ratio'] # Add more if needed
                )
                if ci_service_result and 'error' not in ci_service_result:
                    st.session_state.kpi_confidence_intervals = ci_service_result
                    logger.info(f"Global KPI CIs calculated: {list(ci_service_result.keys())}")
                else:
                    error_msg_ci = ci_service_result.get('error', 'Unknown error') if ci_service_result else "CI calculation failed"
                    display_custom_message(f"Warning: CIs calculation error: {error_msg_ci}", "warning")
                    st.session_state.kpi_confidence_intervals = {} # Reset on error
            else:
                error_msg_kpi = kpi_service_result.get('error', 'Unknown error') if kpi_service_result else "KPI calculation failed"
                display_custom_message(f"Error calculating KPIs: {error_msg_kpi}", "error")
                st.session_state.kpi_results = None # Reset on error
                st.session_state.kpi_confidence_intervals = {}
elif st.session_state.filtered_data is not None and st.session_state.filtered_data.empty:
    # If filters result in no data, clear KPIs
    if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
        display_custom_message("No data matches current filters. Adjust filters or check data.", "info")
    st.session_state.kpi_results = None
    st.session_state.kpi_confidence_intervals = {}


# --- Initial Welcome Message or No Data Message ---
if st.session_state.processed_data is None and not uploaded_file:
    # This is the initial state before any upload
    st.markdown("### Welcome to the Trading Performance Dashboard!")
    st.markdown("Use the sidebar to upload your trading journal (CSV file) and select analysis options to get started.")
    logger.info("Displaying welcome message as no data is loaded.")
elif st.session_state.processed_data is not None and \
     (st.session_state.filtered_data is None or st.session_state.filtered_data.empty) and \
     not (st.session_state.kpi_results and 'error' not in st.session_state.kpi_results):
     # This case handles when data is uploaded but filters make it empty
     if st.session_state.sidebar_filters and uploaded_file: # Only show if filters are active after an upload
        display_custom_message(
            "No data matches the current filter selection. Please adjust your filters in the sidebar or verify the uploaded data content.", 
            "info"
        )

logger.info(f"Application '{APP_TITLE}' main script execution finished for this run cycle.")
# Streamlit automatically routes to pages in the pages/ directory.
# The first page alphabetically (e.g., 0_❓_User_Guide.py or 1_📈_Overview.py) will be shown by default.
