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
    from config import EXPECTED_COLUMNS, RISK_FREE_RATE, APP_TITLE
except ImportError:
    print("Warning (sidebar_manager.py): Could not import from root config. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default"
    EXPECTED_COLUMNS = {"date": "date", "symbol": "symbol", "strategy": "strategy"}
    RISK_FREE_RATE = 0.02

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
                df_dates_dt = pd.to_datetime(self.processed_data[date_col_name], errors='coerce').dropna()
                if not df_dates_dt.empty:
                    min_date_obj = df_dates_dt.min().date()
                    max_date_obj = df_dates_dt.max().date()
                    return min_date_obj, max_date_obj
            except Exception as e:
                logger.error(f"Error processing date column ('{date_col_name}') for date range: {e}", exc_info=True)
        return None

    def render_sidebar_controls(self) -> Dict[str, Any]:
        with st.sidebar:
            # ... (RFR input) ...
            initial_rfr = st.session_state.get('risk_free_rate', RISK_FREE_RATE)
            rfr_percentage = st.number_input(
                "Annual Risk-Free Rate (%)", min_value=0.0, max_value=100.0,
                value=initial_rfr * 100, step=0.01, format="%.2f", key="sidebar_rfr_input_v3"
            )
            self.filter_values['risk_free_rate'] = rfr_percentage / 100.0
            if 'risk_free_rate' not in st.session_state or st.session_state.risk_free_rate != self.filter_values['risk_free_rate']:
                 st.session_state.risk_free_rate = self.filter_values['risk_free_rate']

            st.markdown("---")
            st.subheader("Data Filters")

            date_range_objs = self._get_date_range_objects()
            selected_date_val_tuple = None # Stores tuple of (datetime.date, datetime.date)

            if date_range_objs:
                min_date_data, max_date_data = date_range_objs

                if min_date_data <= max_date_data:
                    session_default_tuple = st.session_state.get('sidebar_date_range_filter_tuple_val')
                    
                    default_start_val = min_date_data
                    default_end_val = max_date_data

                    if session_default_tuple and isinstance(session_default_tuple, tuple) and len(session_default_tuple) == 2:
                        s_start, s_end = session_default_tuple
                        # Ensure s_start and s_end are datetime.date for comparison and use with st.date_input
                        if isinstance(s_start, datetime.datetime): s_start = s_start.date()
                        if isinstance(s_end, datetime.datetime): s_end = s_end.date()

                        if isinstance(s_start, datetime.date) and isinstance(s_end, datetime.date):
                             # Ensure session default is within the new data's bounds
                             current_default_start = max(min_date_data, s_start)
                             current_default_end = min(max_date_data, s_end)
                             if current_default_start <= current_default_end:
                                 default_start_val = current_default_start
                                 default_end_val = current_default_end
                             # else: keep min_date_data, max_date_data if session range is now invalid
                        else:
                            logger.warning("Session state for date range ('sidebar_date_range_filter_tuple_val') was not a tuple of date/datetime objects. Resetting to full data range.")
                    
                    if min_date_data < max_date_data :
                        selected_date_val_tuple = st.date_input(
                            "Select Date Range",
                            value=(default_start_val, default_end_val),
                            min_value=min_date_data,
                            max_value=max_date_data,
                            key="sidebar_date_range_filter_tuple_input_v3"
                        )
                    else: 
                        selected_date_val_tuple = (min_date_data, max_date_data)
                        st.info(f"Data available for a single date: {min_date_data.strftime('%Y-%m-%d')}")
                    
                    st.session_state.sidebar_date_range_filter_tuple_val = selected_date_val_tuple
                else:
                    logger.warning("Min date is after max date in sidebar date filter after processing.")
            else:
                st.info("Upload data with a valid 'date' column for date filtering.")
            self.filter_values['selected_date_range'] = selected_date_val_tuple

            # --- Symbol Filter ---
            # ... (Symbol filter logic as before, ensure keys are unique if needed) ...
            actual_symbol_col = EXPECTED_COLUMNS.get('symbol')
            selected_symbol_val = "All"
            if self.processed_data is not None and actual_symbol_col and actual_symbol_col in self.processed_data.columns:
                try:
                    unique_symbols = ["All"] + sorted(self.processed_data[actual_symbol_col].astype(str).dropna().unique().tolist())
                    if unique_symbols:
                         selected_symbol_val = st.selectbox("Filter by Symbol", unique_symbols, index=0, key="sidebar_symbol_filter_input_v3")
                except Exception as e: logger.error(f"Error populating symbol filter ('{actual_symbol_col}'): {e}", exc_info=True)
            self.filter_values['selected_symbol'] = selected_symbol_val

            # --- Strategy Filter ---
            actual_strategy_col = EXPECTED_COLUMNS.get('strategy')
            selected_strategy_val = "All"
            if self.processed_data is not None and actual_strategy_col and actual_strategy_col in self.processed_data.columns:
                try:
                    unique_strategies = ["All"] + sorted(self.processed_data[actual_strategy_col].astype(str).dropna().unique().tolist())
                    if unique_strategies:
                        selected_strategy_val = st.selectbox("Filter by Strategy", unique_strategies, index=0, key="sidebar_strategy_filter_input_v3")
                except Exception as e: logger.error(f"Error populating strategy filter ('{actual_strategy_col}'): {e}", exc_info=True)
            self.filter_values['selected_strategy'] = selected_strategy_val
            
            logger.debug(f"Sidebar controls rendered. Filter values: {self.filter_values}")
            return self.filter_values
