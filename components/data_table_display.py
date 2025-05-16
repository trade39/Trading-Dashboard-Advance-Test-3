"""
components/data_table_display.py

This component provides a standardized way to display pandas DataFrames
within the Streamlit application, potentially with enhanced features like
custom styling, pagination controls, or advanced filtering options beyond
Streamlit's default.
"""
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any

try:
    from config import APP_TITLE # For logger
except ImportError:
    APP_TITLE = "TradingDashboard_Default"

import logging
logger = logging.getLogger(APP_TITLE) # Get the main app logger

class DataTableDisplay:
    """
    A component for displaying data tables.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        title: Optional[str] = None,
        columns_to_display: Optional[List[str]] = None,
        column_config: Optional[Dict[str, Any]] = None, # Streamlit's column_config
        height: Optional[int] = 400, # Default height for scrollable table
        use_container_width: bool = True,
        hide_index: bool = True,
        download_button: bool = True,
        download_file_name: str = "filtered_data.csv"
    ):
        """
        Initialize the DataTableDisplay.

        Args:
            dataframe (pd.DataFrame): The data to display.
            title (Optional[str]): An optional title for the table section.
            columns_to_display (Optional[List[str]]): Specific columns to show. If None, shows all.
            column_config (Optional[Dict[str, Any]]): Configuration for columns (e.g., formatting).
                                                       Refer to st.column_config.
            height (Optional[int]): Height for the scrollable DataFrame.
            use_container_width (bool): Whether the table should use the full container width.
            hide_index (bool): Whether to hide the DataFrame index.
            download_button (bool): Whether to include a download button for the displayed data.
            download_file_name (str): Default file name for the downloaded CSV.
        """
        self.dataframe = dataframe
        self.title = title
        self.columns_to_display = columns_to_display
        self.column_config = column_config if column_config else {}
        self.height = height
        self.use_container_width = use_container_width
        self.hide_index = hide_index
        self.download_button = download_button
        self.download_file_name = download_file_name
        logger.debug(f"DataTableDisplay initialized for title: {title if title else 'Untitled'}")

    def render(self) -> None:
        """
        Renders the data table.
        """
        if self.dataframe is None or self.dataframe.empty:
            if self.title:
                st.subheader(self.title)
            st.info("No data available to display in the table.")
            logger.info(f"DataTableDisplay: No data to render for '{self.title if self.title else 'Untitled'}'.")
            return

        if self.title:
            st.subheader(self.title)

        df_to_show = self.dataframe.copy()

        if self.columns_to_display:
            # Ensure all requested columns exist in the dataframe
            valid_cols = [col for col in self.columns_to_display if col in df_to_show.columns]
            if len(valid_cols) != len(self.columns_to_display):
                missing = set(self.columns_to_display) - set(valid_cols)
                logger.warning(f"DataTableDisplay: Columns {missing} not found in DataFrame. They will be omitted.")
            df_to_show = df_to_show[valid_cols]

        # Apply default formatting for known column types if not specified in column_config
        # For example, format PnL columns as currency, percentage columns appropriately.
        # This can be expanded based on `config.EXPECTED_COLUMNS` or conventions.
        # Example:
        # for col_name, col_props in config.EXPECTED_COLUMNS_PROPERTIES.items(): # Assuming such a config exists
        #     if col_name in df_to_show.columns and col_name not in self.column_config:
        #         if col_props.get('type') == 'currency':
        #             self.column_config[col_name] = st.column_config.NumberColumn(format="$%.2f")
        #         elif col_props.get('type') == 'percentage':
        #             self.column_config[col_name] = st.column_config.NumberColumn(format="%.2f%%")

        st.dataframe(
            df_to_show,
            column_config=self.column_config if self.column_config else None,
            height=self.height,
            use_container_width=self.use_container_width,
            hide_index=self.hide_index
        )

        if self.download_button:
            try:
                csv_data = df_to_show.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Data as CSV",
                    data=csv_data,
                    file_name=self.download_file_name,
                    mime="text/csv",
                    key=f"download_btn_{self.title.replace(' ','_') if self.title else 'data'}" # Unique key
                )
                logger.debug(f"Download button rendered for '{self.title if self.title else 'Untitled'}'.")
            except Exception as e:
                logger.error(f"Error preparing data for download ('{self.title}'): {e}", exc_info=True)
                st.error("Could not prepare data for download.")
        logger.debug(f"DataTableDisplay rendering complete for '{self.title if self.title else 'Untitled'}'.")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Test Data Table Display Component")

    # Sample DataFrame
    sample_df_data = {
        'Trade ID': [101, 102, 103, 104, 105],
        'Date': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 12:30', '2023-01-02 09:15', '2023-01-02 14:00', '2023-01-03 11:45']),
        'Symbol': ['EURUSD', 'GBPUSD', 'EURUSD', 'USDJPY', 'AUDUSD'],
        'PnL ($)': [150.25, -75.50, 220.00, 90.75, -30.20],
        'Win Rate (%)': [None, None, None, None, None], # Placeholder for a column that might need specific formatting
        'Notes': ["Good entry", "Stop loss hit", "Target reached", "Quick scalp", "Reversal"]
    }
    sample_df = pd.DataFrame(sample_df_data)
    sample_df['Win Rate (%)'] = [60.0, 0.0, 100.0, 100.0, 0.0] # Example values

    st.markdown("### Default Table")
    default_table = DataTableDisplay(sample_df.copy(), title="All Trades (Default)")
    default_table.render()

    st.markdown("---")
    st.markdown("### Table with Selected Columns and Custom Config")
    custom_column_config = {
        "Date": st.column_config.DatetimeColumn(
            "Trade Execution Time",
            format="YYYY-MM-DD HH:mm",
        ),
        "PnL ($)": st.column_config.NumberColumn(
            "Profit/Loss",
            format="$%.2f",
            help="Net profit or loss from the trade",
        ),
        "Win Rate (%)": st.column_config.ProgressColumn(
            "Win Rate",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        )
    }
    selected_cols = ['Date', 'Symbol', 'PnL ($)', 'Win Rate (%)', 'Notes']
    custom_table = DataTableDisplay(
        sample_df.copy(),
        title="Key Trade Info (Customized)",
        columns_to_display=selected_cols,
        column_config=custom_column_config,
        height=300,
        download_file_name="key_trade_info.csv"
    )
    custom_table.render()

    st.markdown("---")
    st.markdown("### Empty Table")
    empty_table = DataTableDisplay(pd.DataFrame(), title="Empty Data Set")
    empty_table.render()

    logger.info("DataTableDisplay test complete.")
