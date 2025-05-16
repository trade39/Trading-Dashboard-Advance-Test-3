"""
services/data_service.py

This service module is responsible for encapsulating the data loading,
processing, and preparation workflow. It acts as a higher-level interface
over the data_processing.py module and can incorporate caching strategies
or more complex data sourcing logic in the future.
"""
import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict

# Assuming root level modules for config and data_processing
try:
    from config import APP_TITLE
    from data_processing import load_and_process_data
except ImportError:
    # Fallback for potential import issues during standalone testing or if path isn't set up
    print("Warning (data_service.py): Could not import from root config or data_processing. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default_Service"
    # Placeholder for load_and_process_data if import fails
    def load_and_process_data(uploaded_file: Any) -> Optional[pd.DataFrame]:
        if uploaded_file:
            try:
                # Simplified processing for placeholder
                df = pd.read_csv(uploaded_file)
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if 'pnl' in df.columns:
                    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
                return df.dropna(subset=['date', 'pnl']) if 'date' in df.columns and 'pnl' in df.columns else df

            except Exception as e:
                # In a real scenario, logger would be used here
                print(f"Placeholder load_and_process_data error: {e}")
                return None
        return None

import logging
logger = logging.getLogger(APP_TITLE) # Get the main app logger, configured in app.py

class DataService:
    """
    Provides services related to data loading, processing, and management.
    """
    def __init__(self):
        logger.info("DataService initialized.")

    # The @st.cache_data decorator is typically applied at the function level
    # (like in data_processing.load_and_process_data).
    # This service method will call the cached function.
    def get_processed_trading_data(self, uploaded_file: Any) -> Optional[pd.DataFrame]:
        """
        Loads and processes trading data from an uploaded file.
        This method calls the cached `load_and_process_data` function.

        Args:
            uploaded_file: The file object uploaded via Streamlit's file_uploader.

        Returns:
            Optional[pd.DataFrame]: Processed DataFrame if successful, None otherwise.
        """
        if uploaded_file is None:
            logger.debug("DataService: No file provided to get_processed_trading_data.")
            return None

        try:
            logger.info(f"DataService: Attempting to process file: {uploaded_file.name}")
            # `load_and_process_data` from data_processing.py is already cached
            processed_df = load_and_process_data(uploaded_file)

            if processed_df is not None:
                logger.info(f"DataService: File '{uploaded_file.name}' processed successfully. Shape: {processed_df.shape}")
            else:
                logger.warning(f"DataService: Processing of file '{uploaded_file.name}' returned None.")
            return processed_df
        except Exception as e:
            logger.error(f"DataService: Unexpected error during data processing for '{uploaded_file.name}': {e}", exc_info=True)
            # User-facing error should be handled by the caller (e.g., in app.py)
            return None

    def filter_data(
        self,
        df: pd.DataFrame,
        filters: Dict[str, Any],
        column_map: Dict[str, str] = None # Maps filter keys to actual df column names if different
    ) -> pd.DataFrame:
        """
        Applies a set of filters to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            filters (Dict[str, Any]): A dictionary of filters.
                Expected keys: 'selected_date_range', 'selected_symbol', 'selected_strategy'.
            column_map (Dict[str, str], optional): Maps filter keys to DataFrame column names.
                Defaults to using keys from `config.EXPECTED_COLUMNS`.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if df is None or df.empty:
            logger.debug("DataService: DataFrame for filtering is None or empty.")
            return pd.DataFrame() # Return empty DataFrame

        if column_map is None:
            from config import EXPECTED_COLUMNS # Import here if not passed
            column_map = EXPECTED_COLUMNS

        filtered_df = df.copy()
        logger.info(f"DataService: Applying filters. Initial shape: {filtered_df.shape}. Filters: {filters}")

        # Date filter
        date_col = column_map.get('date')
        date_range = filters.get('selected_date_range')
        if date_col and date_col in filtered_df.columns and date_range and len(date_range) == 2:
            try:
                start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                # Ensure the DataFrame's date column is also datetime
                filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
                filtered_df = filtered_df.dropna(subset=[date_col]) # Remove rows where date conversion failed

                filtered_df = filtered_df[
                    (filtered_df[date_col].dt.date >= start_date.date()) &
                    (filtered_df[date_col].dt.date <= end_date.date())
                ]
                logger.debug(f"Applied date filter. Shape after date filter: {filtered_df.shape}")
            except Exception as e:
                logger.error(f"Error applying date filter: {e}", exc_info=True)


        # Symbol filter
        symbol_col = column_map.get('symbol')
        selected_symbol = filters.get('selected_symbol')
        if symbol_col and symbol_col in filtered_df.columns and selected_symbol and selected_symbol != "All":
            try:
                filtered_df = filtered_df[filtered_df[symbol_col].astype(str) == str(selected_symbol)]
                logger.debug(f"Applied symbol filter '{selected_symbol}'. Shape: {filtered_df.shape}")
            except Exception as e:
                logger.error(f"Error applying symbol filter: {e}", exc_info=True)


        # Strategy filter
        strategy_col = column_map.get('strategy')
        selected_strategy = filters.get('selected_strategy')
        if strategy_col and strategy_col in filtered_df.columns and selected_strategy and selected_strategy != "All":
            try:
                filtered_df = filtered_df[filtered_df[strategy_col].astype(str) == str(selected_strategy)]
                logger.debug(f"Applied strategy filter '{selected_strategy}'. Shape: {filtered_df.shape}")
            except Exception as e:
                logger.error(f"Error applying strategy filter: {e}", exc_info=True)

        logger.info(f"DataService: Filtering complete. Final shape: {filtered_df.shape}")
        return filtered_df


if __name__ == "__main__":
    # This block is for testing services/data_service.py independently.
    # It requires creating a dummy CSV file and simulating Streamlit's file_uploader.
    logger.info("--- Testing DataService ---")

    # Create a dummy CSV for testing
    sample_csv_data = """trade_id,date,symbol,entry_price,exit_price,pnl,risk,notes,strategy,signal_confidence
1,2023-01-01 10:00:00,EURUSD,1.1,1.101,10,5,Note 1,Strategy A,0.8
2,2023-01-01 12:00:00,EURUSD,1.102,1.101,-10,5,Note 2,Strategy B,0.7
3,2023-01-02 09:00:00,GBPUSD,1.2,1.202,20,10,Note 3,Strategy A,0.9
4,2023-01-03 10:00:00,EURUSD,1.105,1.107,20,5,Note 4,Strategy A,0.85
"""
    from io import StringIO, BytesIO

    # Simulate Streamlit's UploadedFile
    class MockUploadedFile:
        def __init__(self, name, data_bytes):
            self.name = name
            self._data = BytesIO(data_bytes)
        def read(self): # pandas.read_csv uses read()
            return self._data.read()
        def seek(self, offset, whence=0): # pandas might seek
            return self._data.seek(offset, whence)


    mock_file = MockUploadedFile("test_trades.csv", sample_csv_data.encode('utf-8'))

    data_service = DataService()

    st.subheader("Test Data Processing via Service") # Requires Streamlit context
    processed_df = data_service.get_processed_trading_data(mock_file)

    if processed_df is not None:
        st.write("Processed DataFrame via DataService:")
        st.dataframe(processed_df)
        logger.info(f"Successfully processed data via service. Shape: {processed_df.shape}")

        # Test filtering
        st.subheader("Test Data Filtering via Service")
        test_filters = {
            'selected_date_range': (pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-01')),
            'selected_symbol': 'EURUSD',
            'selected_strategy': 'All' # No strategy filter
        }
        # Need to import EXPECTED_COLUMNS for the default column_map in filter_data
        from config import EXPECTED_COLUMNS as test_expected_cols
        
        filtered_data_from_service = data_service.filter_data(processed_df, test_filters, column_map=test_expected_cols)
        st.write(f"Filtered DataFrame (Filters: {test_filters}):")
        st.dataframe(filtered_data_from_service)
        logger.info(f"Successfully filtered data via service. Shape: {filtered_data_from_service.shape}")

        test_filters_2 = {'selected_strategy': 'Strategy A'}
        filtered_data_strat = data_service.filter_data(processed_df, test_filters_2, column_map=test_expected_cols)
        st.write(f"Filtered DataFrame (Filters: {test_filters_2}):")
        st.dataframe(filtered_data_strat)
        logger.info(f"Successfully filtered data by strategy via service. Shape: {filtered_data_strat.shape}")

    else:
        st.error("Failed to process data via DataService.")
        logger.error("DataService test: Processing returned None.")

    logger.info("--- DataService test complete ---")

