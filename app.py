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
        LOG_FILE, LOG_LEVEL, LOG_FORMAT, DEFAULT_BENCHMARK_TICKER, AVAILABLE_BENCHMARKS
    )
except ImportError as e:
    st.error(f"Fatal Error: Could not import configuration. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing config: {e}", exc_info=True)
    APP_TITLE = "TradingDashboard_Error"
    LOG_FILE = "logs/error_app.log"; LOG_LEVEL = "ERROR"; LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    RISK_FREE_RATE = 0.02; EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}; DEFAULT_BENCHMARK_TICKER = "SPY"
    AVAILABLE_BENCHMARKS = {"S&P 500 (SPY)": "SPY", "None": ""}


# --- Initialize Centralized Logger ---
logger = setup_logger(
    logger_name=APP_TITLE, log_file=LOG_FILE, level=LOG_LEVEL, log_format=LOG_FORMAT
)
logger.info(f"Application '{APP_TITLE}' starting. Logger initialized.")
logger.debug(f"CWD at app start: {os.getcwd()}, sys.path: {sys.path}")


# --- Page Configuration (Global for all pages) ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/trading-dashboard-tmh', 
        'Report a bug': "https://github.com/your-repo/trading-dashboard-tmh/issues", 
        'About': f"## {APP_TITLE}\n\nA comprehensive dashboard for trading performance analysis."
    }
)
logger.debug("Streamlit page_config set.")

# --- Load Custom CSS ---
try:
    load_css("style.css") 
    logger.debug("Custom CSS loaded.")
except Exception as e:
    logger.error(f"Failed to load style.css: {e}", exc_info=True)
    st.warning("Could not load custom styles. The app may not appear as intended.")

# --- Initialize Session State ---
default_session_state = {
    'app_initialized': True, 'processed_data': None, 'filtered_data': None,
    'kpi_results': None, 'kpi_confidence_intervals': {},
    'risk_free_rate': RISK_FREE_RATE, 'current_theme': "dark",
    'uploaded_file_name': None, 'last_processed_file_id': None,
    'last_filtered_data_shape': None, 'sidebar_filters': None,
    'active_tab': "📈 Overview",
    'selected_benchmark_ticker': DEFAULT_BENCHMARK_TICKER,
    'selected_benchmark_display_name': next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == DEFAULT_BENCHMARK_TICKER), "None"),
    'benchmark_daily_returns': None,
    'initial_capital': 100000.0,
    'last_applied_filters': None, 
    'last_fetched_benchmark_ticker': None, 
    'last_benchmark_data_filter_shape': None, 
    'last_kpi_calc_state_id': None 
}

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

uploaded_file = st.sidebar.file_uploader(
    "Upload Trading Journal (CSV)", type=["csv"],
    help=f"Expected columns include: {', '.join(EXPECTED_COLUMNS.values())}",
    key="app_wide_file_uploader"
)

sidebar_manager = SidebarManager(st.session_state.processed_data)
current_sidebar_filters = sidebar_manager.render_sidebar_controls() 

st.session_state.sidebar_filters = current_sidebar_filters

# Update session state from sidebar if changes occurred
if current_sidebar_filters:
    rfr_from_sidebar = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
    if st.session_state.risk_free_rate != rfr_from_sidebar:
        st.session_state.risk_free_rate = rfr_from_sidebar
        logger.info(f"Global risk-free rate updated to: {st.session_state.risk_free_rate:.4f}")
        st.session_state.kpi_results = None 

    benchmark_ticker_from_sidebar = current_sidebar_filters.get('selected_benchmark_ticker', "")
    if st.session_state.selected_benchmark_ticker != benchmark_ticker_from_sidebar:
        st.session_state.selected_benchmark_ticker = benchmark_ticker_from_sidebar
        st.session_state.selected_benchmark_display_name = next(
            (name for name, ticker in AVAILABLE_BENCHMARKS.items() 
             if ticker == st.session_state.selected_benchmark_ticker), "None"
        )
        logger.info(f"Benchmark ticker updated to: {st.session_state.selected_benchmark_ticker}")
        st.session_state.benchmark_daily_returns = None 
        st.session_state.kpi_results = None 

    initial_capital_from_sidebar = current_sidebar_filters.get('initial_capital', 100000.0)
    if st.session_state.initial_capital != initial_capital_from_sidebar:
        st.session_state.initial_capital = initial_capital_from_sidebar
        logger.info(f"Initial capital updated to: {st.session_state.initial_capital:.2f}")
        st.session_state.kpi_results = None 


# --- Data Loading and Processing ---
if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
    if st.session_state.last_processed_file_id != current_file_id or st.session_state.processed_data is None:
        logger.info(f"File '{uploaded_file.name}' (ID: {current_file_id}) selected. Initiating processing.")
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            st.session_state.processed_data = data_service.get_processed_trading_data(uploaded_file)
        
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.last_processed_file_id = current_file_id
        st.session_state.kpi_results = None
        st.session_state.kpi_confidence_intervals = {}
        st.session_state.filtered_data = st.session_state.processed_data
        st.session_state.benchmark_daily_returns = None 

        if st.session_state.processed_data is not None:
            logger.info(f"DataService processed '{uploaded_file.name}'. Shape: {st.session_state.processed_data.shape}")
            display_custom_message(f"Successfully processed '{uploaded_file.name}'. Navigate pages to see analysis.", "success", icon="✅")
        else:
            logger.error(f"DataService failed to process file: {uploaded_file.name}.")
            display_custom_message(f"Failed to process '{uploaded_file.name}'. Check logs and file format.", "error")
            st.session_state.processed_data = None 
            st.session_state.filtered_data = None
            st.session_state.kpi_results = None
            st.session_state.kpi_confidence_intervals = {}
            st.session_state.benchmark_daily_returns = None


# --- Data Filtering ---
if st.session_state.processed_data is not None and st.session_state.sidebar_filters:
    if st.session_state.filtered_data is None or \
       st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        
        logger.info("Applying global filters via DataService...")
        st.session_state.filtered_data = data_service.filter_data(
            st.session_state.processed_data,
            st.session_state.sidebar_filters,
            column_map=EXPECTED_COLUMNS
        )
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        logger.debug(f"Data filtered. New shape: {st.session_state.filtered_data.shape if st.session_state.filtered_data is not None else 'None'}")
        st.session_state.kpi_results = None
        st.session_state.kpi_confidence_intervals = {}
        st.session_state.benchmark_daily_returns = None


# --- Benchmark Data Fetching ---
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')
    if selected_ticker and selected_ticker != "": 
        refetch_benchmark = False
        if st.session_state.benchmark_daily_returns is None:
            refetch_benchmark = True
        elif st.session_state.last_fetched_benchmark_ticker != selected_ticker:
            refetch_benchmark = True
        elif st.session_state.last_benchmark_data_filter_shape != st.session_state.filtered_data.shape:
            refetch_benchmark = True

        if refetch_benchmark:
            date_col_name = EXPECTED_COLUMNS.get('date')
            if date_col_name and date_col_name in st.session_state.filtered_data.columns:
                min_date_ts = pd.to_datetime(st.session_state.filtered_data[date_col_name]).min()
                max_date_ts = pd.to_datetime(st.session_state.filtered_data[date_col_name]).max()
                
                # Convert to ISO format string for passing to cached function
                min_date_str = min_date_ts.strftime('%Y-%m-%d') if pd.notna(min_date_ts) else None
                max_date_str = max_date_ts.strftime('%Y-%m-%d') if pd.notna(max_date_ts) else None
                
                if min_date_str and max_date_str:
                    logger.info(f"Fetching benchmark data for {selected_ticker} from {min_date_str} to {max_date_str}.")
                    st.session_state.benchmark_daily_returns = analysis_service.get_benchmark_data(
                        selected_ticker, min_date_str, max_date_str # Pass date strings
                    )
                    st.session_state.last_fetched_benchmark_ticker = selected_ticker
                    st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape

                    if st.session_state.benchmark_daily_returns is None:
                        display_custom_message(f"Could not fetch benchmark data for {selected_ticker}. Proceeding without benchmark comparison.", "warning")
                    else:
                        logger.info(f"Benchmark data for {selected_ticker} fetched successfully.")
                else:
                    logger.warning("Min/max dates from filtered_data are NaT or invalid, cannot fetch benchmark data.")
                    st.session_state.benchmark_daily_returns = None
            else:
                logger.warning(f"Date column '{date_col_name}' not found in filtered_data for benchmark date range.")
                st.session_state.benchmark_daily_returns = None
            st.session_state.kpi_results = None 
    elif st.session_state.benchmark_daily_returns is not None : 
        logger.info("No benchmark selected or deselected. Clearing benchmark data.")
        st.session_state.benchmark_daily_returns = None
        st.session_state.kpi_results = None 


# --- KPI Calculation ---
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    current_kpi_calc_state_id = (
        st.session_state.filtered_data.shape,
        st.session_state.risk_free_rate,
        st.session_state.initial_capital,
        st.session_state.selected_benchmark_ticker,
        pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns, index=True).sum() if st.session_state.benchmark_daily_returns is not None and not st.session_state.benchmark_daily_returns.empty else None
    )

    if st.session_state.kpi_results is None or \
       st.session_state.last_kpi_calc_state_id != current_kpi_calc_state_id:
        
        logger.info("Recalculating global KPIs and CIs via AnalysisService...")
        with st.spinner("Calculating performance metrics & CIs..."):
            kpi_service_result = analysis_service.get_core_kpis(
                st.session_state.filtered_data,
                st.session_state.risk_free_rate,
                benchmark_daily_returns=st.session_state.get('benchmark_daily_returns'),
                initial_capital=st.session_state.get('initial_capital')
            )
            
            if kpi_service_result and 'error' not in kpi_service_result:
                st.session_state.kpi_results = kpi_service_result
                st.session_state.last_kpi_calc_state_id = current_kpi_calc_state_id
                logger.info("Global KPIs calculated.")

                ci_service_result = analysis_service.get_bootstrapped_kpi_cis(
                    st.session_state.filtered_data,
                    kpis_to_bootstrap=['avg_trade_pnl', 'win_rate', 'sharpe_ratio']
                )
                if ci_service_result and 'error' not in ci_service_result:
                    st.session_state.kpi_confidence_intervals = ci_service_result
                    logger.info(f"Global KPI CIs calculated: {list(ci_service_result.keys())}")
                else:
                    error_msg_ci = ci_service_result.get('error', 'Unknown error') if ci_service_result else "CI calculation failed"
                    display_custom_message(f"Warning: CIs calculation error: {error_msg_ci}", "warning")
                    st.session_state.kpi_confidence_intervals = {}
            else:
                error_msg_kpi = kpi_service_result.get('error', 'Unknown error') if kpi_service_result else "KPI calculation failed"
                display_custom_message(f"Error calculating KPIs: {error_msg_kpi}", "error")
                st.session_state.kpi_results = None
                st.session_state.kpi_confidence_intervals = {}
elif st.session_state.filtered_data is not None and st.session_state.filtered_data.empty:
    if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
        display_custom_message("No data matches current filters. Adjust filters or check data.", "info")
    st.session_state.kpi_results = None
    st.session_state.kpi_confidence_intervals = {}


# --- Initial Welcome Message or No Data Message ---
if st.session_state.processed_data is None and not uploaded_file:
    st.markdown("### Welcome to the Trading Performance Dashboard!")
    st.markdown("Use the sidebar to upload your trading journal (CSV file) and select analysis options to get started.")
    logger.info("Displaying welcome message as no data is loaded.")
elif st.session_state.processed_data is not None and \
     (st.session_state.filtered_data is None or st.session_state.filtered_data.empty) and \
     not (st.session_state.kpi_results and 'error' not in st.session_state.kpi_results):
     if st.session_state.sidebar_filters and uploaded_file:
        display_custom_message(
            "No data matches the current filter selection. Please adjust your filters in the sidebar or verify the uploaded data content.", 
            "info"
        )

logger.info(f"Application '{APP_TITLE}' main script execution finished for this run cycle.")
