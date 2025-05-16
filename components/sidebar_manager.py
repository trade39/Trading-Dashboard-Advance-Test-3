"""
components/sidebar_manager.py

This component encapsulates the logic for creating and managing
sidebar filters and controls for the Trading Performance Dashboard.
It helps to keep the main app.py script cleaner by centralizing sidebar UI.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import datetime # Import datetime for date objects

try:
    # Added AVAILABLE_BENCHMARKS and DEFAULT_BENCHMARK_TICKER
    from config import EXPECTED_COLUMNS, RISK_FREE_RATE, APP_TITLE, AVAILABLE_BENCHMARKS, DEFAULT_BENCHMARK_TICKER
except ImportError:
    print("Warning (sidebar_manager.py): Could not import from root config. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default"
    EXPECTED_COLUMNS = {"date": "date", "symbol": "symbol", "strategy": "strategy"}
    RISK_FREE_RATE = 0.02
    AVAILABLE_BENCHMARKS = {"S&P 500 (SPY)": "SPY", "None": ""} # Fallback
    DEFAULT_BENCHMARK_TICKER = "SPY"


import logging
logger = logging.getLogger(APP_TITLE)

class SidebarManager:
    def __init__(self, processed_data: Optional[pd.DataFrame]):
        self.processed_data = processed_data
        self.filter_values: Dict[str, Any] = {}
        logger.debug("SidebarManager initialized.")

    def _get_date_range_objects(self) -> Optional[Tuple[datetime.date, datetime.date]]:
        date_col_name = EXPECTED_COLUMNS.get('date')
        if self.processed_data is not None and \
           date_col_name and date_col_name in self.processed_data.columns and \
           not self.processed_data[date_col_name].empty:
            try:
                # Ensure the column is datetime before min/max
                # Errors='coerce' will turn unparseable dates into NaT
                df_dates_dt = pd.to_datetime(self.processed_data[date_col_name], errors='coerce').dropna()
                if not df_dates_dt.empty:
                    min_date_obj = df_dates_dt.min().date()
                    max_date_obj = df_dates_dt.max().date()
                    return min_date_obj, max_date_obj
            except Exception as e:
                logger.error(f"Error processing date column ('{date_col_name}') for date range: {e}", exc_info=True)
        return None

    def render_sidebar_controls(self) -> Dict[str, Any]:
        """
        Renders all sidebar controls and returns the selected filter values.
        """
        with st.sidebar:
            # --- Risk-Free Rate ---
            # Ensure 'risk_free_rate' is in session_state before accessing it.
            # Default to RISK_FREE_RATE from config if not in session_state.
            initial_rfr_value = st.session_state.get('risk_free_rate', RISK_FREE_RATE)
            
            rfr_percentage = st.number_input(
                "Annual Risk-Free Rate (%)", min_value=0.0, max_value=100.0,
                value=initial_rfr_value * 100, # Convert to percentage for display
                step=0.01, format="%.2f", key="sidebar_rfr_input_v3", # Unique key
                help="Enter the annualized risk-free rate (e.g., 2 for 2%). Used for Sharpe, Alpha, etc."
            )
            # Store as a decimal (e.g., 0.02 for 2%)
            self.filter_values['risk_free_rate'] = rfr_percentage / 100.0
            # Update session state if changed, to be picked up by app.py
            if st.session_state.get('risk_free_rate') != self.filter_values['risk_free_rate']:
                 st.session_state.risk_free_rate = self.filter_values['risk_free_rate']
                 logger.info(f"Risk-free rate updated to: {st.session_state.risk_free_rate:.4f}")


            st.markdown("---")
            st.subheader("Data Filters")

            # --- Date Range Filter ---
            date_range_objs = self._get_date_range_objects()
            selected_date_val_tuple = None 

            if date_range_objs:
                min_date_data, max_date_data = date_range_objs

                if min_date_data <= max_date_data: # Valid range from data
                    # Get session state default for date range or use full data range
                    session_default_tuple = st.session_state.get('sidebar_date_range_filter_tuple_val')
                    
                    default_start_val = min_date_data
                    default_end_val = max_date_data

                    if session_default_tuple and isinstance(session_default_tuple, tuple) and len(session_default_tuple) == 2:
                        s_start, s_end = session_default_tuple
                        if isinstance(s_start, datetime.datetime): s_start = s_start.date()
                        if isinstance(s_end, datetime.datetime): s_end = s_end.date()

                        if isinstance(s_start, datetime.date) and isinstance(s_end, datetime.date):
                             current_default_start = max(min_date_data, s_start)
                             current_default_end = min(max_date_data, s_end)
                             if current_default_start <= current_default_end:
                                 default_start_val = current_default_start
                                 default_end_val = current_default_end
                        else:
                            logger.warning("Session state for date range was not a tuple of date/datetime. Resetting.")
                    
                    if min_date_data < max_date_data : # More than one day of data
                        selected_date_val_tuple = st.date_input(
                            "Select Date Range",
                            value=(default_start_val, default_end_val),
                            min_value=min_date_data,
                            max_value=max_date_data,
                            key="sidebar_date_range_filter_tuple_input_v3" # Unique key
                        )
                    else: # Single day of data
                        selected_date_val_tuple = (min_date_data, max_date_data) # Range is just that single day
                        st.info(f"Data available for a single date: {min_date_data.strftime('%Y-%m-%d')}")
                    
                    # Store the selected tuple (of datetime.date objects) in session state
                    st.session_state.sidebar_date_range_filter_tuple_val = selected_date_val_tuple
                else:
                    logger.warning("Min date is after max date in sidebar date filter after processing data.")
            else:
                st.info("Upload data with a valid 'date' column for date filtering.")
            self.filter_values['selected_date_range'] = selected_date_val_tuple


            # --- Symbol Filter ---
            actual_symbol_col = EXPECTED_COLUMNS.get('symbol')
            selected_symbol_val = "All" # Default
            if self.processed_data is not None and actual_symbol_col and actual_symbol_col in self.processed_data.columns:
                try:
                    # Ensure column is string, handle NaNs, get unique, sort, prepend "All"
                    unique_symbols = ["All"] + sorted(self.processed_data[actual_symbol_col].astype(str).dropna().unique().tolist())
                    if unique_symbols: # Check if list is not empty (beyond just "All")
                         selected_symbol_val = st.selectbox(
                             "Filter by Symbol", 
                             unique_symbols, 
                             index=0, # Default to "All"
                             key="sidebar_symbol_filter_input_v3" # Unique key
                        )
                except Exception as e:
                    logger.error(f"Error populating symbol filter ('{actual_symbol_col}'): {e}", exc_info=True)
            self.filter_values['selected_symbol'] = selected_symbol_val


            # --- Strategy Filter ---
            actual_strategy_col = EXPECTED_COLUMNS.get('strategy')
            selected_strategy_val = "All" # Default
            if self.processed_data is not None and actual_strategy_col and actual_strategy_col in self.processed_data.columns:
                try:
                    unique_strategies = ["All"] + sorted(self.processed_data[actual_strategy_col].astype(str).dropna().unique().tolist())
                    if unique_strategies:
                        selected_strategy_val = st.selectbox(
                            "Filter by Strategy", 
                            unique_strategies, 
                            index=0, # Default to "All"
                            key="sidebar_strategy_filter_input_v3" # Unique key
                        )
                except Exception as e:
                    logger.error(f"Error populating strategy filter ('{actual_strategy_col}'): {e}", exc_info=True)
            self.filter_values['selected_strategy'] = selected_strategy_val

            st.markdown("---")
            st.subheader("Benchmark Selection")
            
            # --- Benchmark Selection Dropdown ---
            # Get the list of display names for benchmarks from config
            benchmark_display_names = list(AVAILABLE_BENCHMARKS.keys())
            # Get the ticker for the default benchmark
            default_benchmark_display_name = next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == DEFAULT_BENCHMARK_TICKER), "None")

            # Get current selection from session state or default
            current_benchmark_selection_name = st.session_state.get('selected_benchmark_display_name', default_benchmark_display_name)
            
            # Ensure the current selection is valid, otherwise reset to default
            if current_benchmark_selection_name not in benchmark_display_names:
                current_benchmark_selection_name = default_benchmark_display_name
                st.session_state.selected_benchmark_display_name = current_benchmark_selection_name # Update session state

            selected_benchmark_name = st.selectbox(
                "Select Benchmark",
                options=benchmark_display_names,
                index=benchmark_display_names.index(current_benchmark_selection_name), # Set index based on current selection
                key="sidebar_benchmark_select_v1",
                help="Select a market index to compare your strategy against. 'None' disables benchmark comparison."
            )
            
            # Store the selected benchmark *ticker* in filter_values and session_state
            selected_benchmark_ticker = AVAILABLE_BENCHMARKS.get(selected_benchmark_name, "")
            self.filter_values['selected_benchmark_ticker'] = selected_benchmark_ticker
            
            # Update session state for benchmark selection
            if st.session_state.get('selected_benchmark_ticker') != selected_benchmark_ticker:
                st.session_state.selected_benchmark_ticker = selected_benchmark_ticker
                st.session_state.selected_benchmark_display_name = selected_benchmark_name # Also store display name for consistency
                logger.info(f"Benchmark selection updated to: {selected_benchmark_name} ({selected_benchmark_ticker})")

            # --- Initial Capital Input (for % returns if PnL is absolute) ---
            st.markdown("---")
            st.subheader("Strategy Settings")
            initial_capital_input = st.number_input(
                "Initial Capital (for % Returns & Benchmarking)",
                min_value=0.0,
                value=st.session_state.get('initial_capital', 100000.0), # Default to 100k or session state
                step=1000.0,
                format="%.2f",
                key="sidebar_initial_capital_v1",
                help="Enter the initial capital for your strategy. This is used to calculate percentage returns if your PnL data is in absolute currency values, which is important for accurate Alpha/Beta calculation against benchmark percentage returns. If your PnL is already in percentage terms, this might not be strictly necessary or could be set to 100."
            )
            self.filter_values['initial_capital'] = initial_capital_input
            if st.session_state.get('initial_capital') != initial_capital_input:
                st.session_state.initial_capital = initial_capital_input
                logger.info(f"Initial capital updated to: {initial_capital_input:.2f}")


            logger.debug(f"Sidebar controls rendered. Filter values: {self.filter_values}")
            return self.filter_values
