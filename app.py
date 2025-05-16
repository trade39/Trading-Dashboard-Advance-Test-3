"""
app.py - Main Entry Point for Multi-Page Trading Performance Dashboard
Includes diagnostic prints for debugging import errors.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging # Standard logging module
import sys # For sys.path
import os  # For os.getcwd()

# --- Pre-Import Diagnostics ---
# These prints will execute very early.
print("--- app.py execution started ---")
print(f"Current Working Directory (top of app.py): {os.getcwd()}")
print(f"sys.path (top of app.py): {sys.path}")
# You can also try to list contents of current and services directory
# print(f"Contents of CWD: {os.listdir('.')}")
# if os.path.exists('services'):
#    print(f"Contents of services/: {os.listdir('services')}")
# else:
#    print("Directory 'services/' does NOT exist at CWD.")


# --- Utility Modules ---
try:
    from utils.logger import setup_logger
    from utils.common_utils import load_css, display_custom_message
    print("Successfully imported from utils package.")
except ImportError as e:
    print(f"ERROR importing from utils: {e}")
    st.error(f"Fatal Error: Could not import utility modules. App cannot start. Details: {e}")
    st.stop()


# --- Component Modules (Sidebar is global) ---
try:
    from components.sidebar_manager import SidebarManager
    print("Successfully imported SidebarManager from components.")
except ImportError as e:
    print(f"ERROR importing SidebarManager from components: {e}")
    st.error(f"Fatal Error: Could not import SidebarManager. App cannot start. Details: {e}")
    st.stop()


# --- Service Modules ---
# This is the problematic import area based on user's error.
try:
    from services.data_service import DataService
    print("Successfully imported DataService from services.")
    from services.analysis_service import AnalysisService # Line 28 or around here
    print("Successfully imported AnalysisService from services.")
except ImportError as e:
    print(f"ERROR importing from services (DataService or AnalysisService): {e}")
    # Detailed diagnostics if services import fails
    print(f"Checking existence of services/__init__.py: {os.path.exists('services/__init__.py')}")
    print(f"Checking existence of services/data_service.py: {os.path.exists('services/data_service.py')}")
    print(f"Checking existence of services/analysis_service.py: {os.path.exists('services/analysis_service.py')}")
    st.error(f"Fatal Error: Could not import service modules (DataService or AnalysisService). App cannot start. Details: {e}. Check console for path details.")
    st.stop()


# --- Core Application Modules (Configs) ---
try:
    from config import (
        APP_TITLE, EXPECTED_COLUMNS, RISK_FREE_RATE,
        LOG_FILE, LOG_LEVEL, LOG_FORMAT
    )
    print("Successfully imported from config.")
except ImportError as e:
    print(f"ERROR importing from config: {e}")
    st.error(f"Fatal Error: Could not import configuration. App cannot start. Details: {e}")
    st.stop()

# --- Initialize Centralized Logger ---
# This must happen after config is loaded.
logger = setup_logger(
    logger_name=APP_TITLE, log_file=LOG_FILE, level=LOG_LEVEL, log_format=LOG_FORMAT
)
logger.info(f"Application '{APP_TITLE}' starting. Logger initialized.")
logger.info(f"CWD after imports: {os.getcwd()}") # Log CWD again
logger.info(f"sys.path after imports: {sys.path}") # Log sys.path again


# --- Page Configuration (Global for all pages) ---
# (Rest of app.py from previous version: st.set_page_config, load_css, session state, service instantiation, sidebar, data loading, filtering, KPI calculation, "Home Page" content)
# ... (The rest of your app.py logic would follow here) ...
# For brevity, I'm not repeating the entire app.py content, only the diagnostic part.
# Assume the rest of the app.py (session state, global sidebar, data loading, filtering, KPI logic)
# from the "app.py (Multi-Page Orchestrator)" version follows here.

# --- Page Configuration (Global for all pages) ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/trading-dashboard-tmh',
        'Report a bug': "https://github.com/your-repo/trading-dashboard-tmh/issues",
        'About': f"## {APP_TITLE}\n\nA comprehensive dashboard for trading performance analysis."
    }
)
logger.debug("Streamlit page_config set.")

load_css("style.css")
logger.debug("Custom CSS loaded.")

default_session_state = {
    'app_initialized': True, 'processed_data': None, 'filtered_data': None, 'kpi_results': None,
    'kpi_confidence_intervals': {}, 'risk_free_rate': RISK_FREE_RATE,
    'current_theme': "dark", 'uploaded_file_name': None, 'last_processed_file_id': None,
    'last_filtered_data_shape': None, 'sidebar_filters': None,
    'active_tab': "📈 Overview"
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value
logger.debug("Global session state initialized/checked.")

data_service = DataService()
analysis_service = AnalysisService() # Already imported and checked
logger.debug("DataService and AnalysisService instantiated.")

st.sidebar.title(f"⚙️ {APP_TITLE} Controls")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Trading Journal (CSV)", type=["csv"],
    help=f"Expected columns include: {', '.join(EXPECTED_COLUMNS.values())}",
    key="app_wide_file_uploader"
)

sidebar_manager = SidebarManager(st.session_state.processed_data)
current_sidebar_filters = sidebar_manager.render_sidebar_controls()
st.session_state.sidebar_filters = current_sidebar_filters

if current_sidebar_filters and 'risk_free_rate' in current_sidebar_filters:
    if st.session_state.risk_free_rate != current_sidebar_filters['risk_free_rate']:
        st.session_state.risk_free_rate = current_sidebar_filters['risk_free_rate']
        logger.info(f"Global risk-free rate updated to: {st.session_state.risk_free_rate:.4f}")

if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.last_processed_file_id != current_file_id or st.session_state.processed_data is None:
        logger.info(f"File '{uploaded_file.name}' (ID: {current_file_id}) selected. Initiating processing via DataService.")
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            st.session_state.processed_data = data_service.get_processed_trading_data(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.last_processed_file_id = current_file_id
        st.session_state.kpi_results = None
        st.session_state.kpi_confidence_intervals = {}
        st.session_state.filtered_data = st.session_state.processed_data
        if st.session_state.processed_data is not None:
            logger.info(f"DataService processed '{uploaded_file.name}'. Shape: {st.session_state.processed_data.shape}")
            display_custom_message(f"Successfully processed '{uploaded_file.name}'. Navigate pages to see analysis.", "success", icon="🎉")
            sidebar_manager.processed_data = st.session_state.processed_data
        else:
            logger.error(f"DataService failed to process file: {uploaded_file.name}.")
            display_custom_message(f"Failed to process '{uploaded_file.name}'. Check logs and file format.", "error")
            st.session_state.processed_data = None # Clear potentially problematic state
            st.session_state.filtered_data = None
            st.session_state.kpi_results = None
            st.session_state.kpi_confidence_intervals = {}


if st.session_state.processed_data is not None and st.session_state.sidebar_filters:
    if st.session_state.filtered_data is None or \
       st.session_state.get('last_filter_values') != st.session_state.sidebar_filters:
        logger.info("Applying global filters via DataService...")
        st.session_state.filtered_data = data_service.filter_data(
            st.session_state.processed_data,
            st.session_state.sidebar_filters,
            column_map=EXPECTED_COLUMNS
        )
        st.session_state.last_filter_values = st.session_state.sidebar_filters.copy()
        logger.debug(f"Data filtered. New shape: {st.session_state.filtered_data.shape if st.session_state.filtered_data is not None else 'None'}")
        st.session_state.kpi_results = None
        st.session_state.kpi_confidence_intervals = {}


if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    rfr_for_kpi_calc = st.session_state.risk_free_rate
    if st.session_state.kpi_results is None or \
       st.session_state.get('last_rfr_for_kpi_calc') != rfr_for_kpi_calc:
        logger.info("Recalculating global KPIs and CIs via AnalysisService...")
        with st.spinner("Calculating performance metrics & CIs..."):
            kpi_service_result = analysis_service.get_core_kpis(
                st.session_state.filtered_data, rfr_for_kpi_calc
            )
            if 'error' not in kpi_service_result:
                st.session_state.kpi_results = kpi_service_result
                st.session_state.last_rfr_for_kpi_calc = rfr_for_kpi_calc
                logger.info("Global KPIs calculated.")
                ci_service_result = analysis_service.get_bootstrapped_kpi_cis(
                    st.session_state.filtered_data,
                    kpis_to_bootstrap=['avg_trade_pnl', 'win_rate', 'sharpe_ratio']
                )
                if 'error' not in ci_service_result:
                    st.session_state.kpi_confidence_intervals = ci_service_result
                    logger.info(f"Global KPI CIs calculated: {ci_service_result}")
                else:
                    display_custom_message(f"Warning: CIs calculation error: {ci_service_result['error']}", "warning")
                    st.session_state.kpi_confidence_intervals = {}
            else:
                display_custom_message(f"Error calculating KPIs: {kpi_service_result['error']}", "error")
                st.session_state.kpi_results = None
                st.session_state.kpi_confidence_intervals = {}
        st.session_state.last_filtered_data_shape = st.session_state.filtered_data.shape
elif st.session_state.filtered_data is not None and st.session_state.filtered_data.empty:
    if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
        display_custom_message("No data matches current filters. Adjust filters or check data.", "info")
    st.session_state.kpi_results = None
    st.session_state.kpi_confidence_intervals = {}


if st.session_state.processed_data is None and not uploaded_file:
    st.markdown("### Welcome to the Trading Performance Dashboard!")
    st.markdown("Use the sidebar to upload your trading journal (CSV file) to get started.")
    # st.image("https://placehold.co/600x300/273334/E0E0E0?text=Trading+Analytics", caption="Visualize Your Trading Edge") # Placeholder if you have one
    logger.info("Displaying welcome message as no data is loaded.")
elif st.session_state.processed_data is not None and (st.session_state.filtered_data is None or st.session_state.filtered_data.empty) and not (st.session_state.kpi_results and 'error' not in st.session_state.kpi_results):
     if st.session_state.sidebar_filters and uploaded_file: # Only show if filters are active after an upload
        display_custom_message("No data matches the current filter selection. Please adjust your filters in the sidebar or verify the uploaded data content.", "info")


logger.info(f"Application '{APP_TITLE}' main script execution finished for this run.")
# Streamlit automatically routes to pages in the pages/ directory.
# If no page is explicitly selected, it usually shows the first page by alphabetical order (e.g., 1_📈_Overview.py).
# This app.py now acts as the global setup and state manager.

