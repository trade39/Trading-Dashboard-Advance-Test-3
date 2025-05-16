"""
components/notes_viewer.py

This component provides an interactive interface for viewing, searching,
and filtering trade notes, with improved PnL sorting and display.
"""
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any

try:
    from config import EXPECTED_COLUMNS, APP_TITLE
    from utils.common_utils import format_currency
except ImportError:
    print("Warning (notes_viewer.py): Could not import from root config/utils. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default"
    EXPECTED_COLUMNS = {
        "date": "date", "notes": "notes", "pnl": "pnl", "symbol": "symbol"
    }
    def format_currency(value, currency_symbol="$", decimals=2):
        if pd.isna(value): return "N/A"
        return f"{currency_symbol}{value:.{decimals}f}"

import logging
logger = logging.getLogger(APP_TITLE)

class NotesViewerComponent:
    """
    A component for displaying and interacting with trade notes.
    Ensures PnL is sorted numerically and formatted for display.
    """
    def __init__(
        self,
        notes_dataframe: pd.DataFrame,
        date_col: str = EXPECTED_COLUMNS.get('date', 'date'),
        notes_col: str = EXPECTED_COLUMNS.get('notes', 'notes'),
        pnl_col: str = EXPECTED_COLUMNS.get('pnl', 'pnl'),
        symbol_col: Optional[str] = EXPECTED_COLUMNS.get('symbol'),
        default_sort_by_display: str = "Date", # Default sort by display name
        default_sort_ascending: bool = False
    ):
        self.base_df = notes_dataframe # Original DataFrame with raw data
        self.date_col_orig = date_col
        self.notes_col_orig = notes_col
        self.pnl_col_orig = pnl_col
        self.symbol_col_orig = symbol_col

        # Mapping from original column names to display names
        self.display_col_map = {
            self.date_col_orig: "Date",
            self.notes_col_orig: "Trade Note",
            self.pnl_col_orig: "PnL"
        }
        if self.symbol_col_orig:
            self.display_col_map[self.symbol_col_orig] = "Symbol"

        self.default_sort_by_display = default_sort_by_display
        self.default_sort_ascending = default_sort_ascending

        self.df_prepared = self._prepare_and_filter_df()
        logger.debug("NotesViewerComponent initialized.")

    def _prepare_and_filter_df(self) -> pd.DataFrame:
        """
        Prepares the DataFrame for internal use:
        - Selects necessary original columns.
        - Filters out rows with empty or NaN notes.
        - Converts PnL to numeric if it's not already.
        """
        if self.base_df is None or self.base_df.empty:
            return pd.DataFrame()

        required_original_cols = [self.date_col_orig, self.notes_col_orig, self.pnl_col_orig]
        if self.symbol_col_orig and self.symbol_col_orig in self.base_df.columns:
            required_original_cols.append(self.symbol_col_orig)
        else: # Ensure symbol_col_orig is None if not present, to avoid errors later
            self.symbol_col_orig = None


        missing_cols = [col for col in required_original_cols if col not in self.base_df.columns]
        if missing_cols:
            logger.error(f"NotesViewer: Missing original columns: {missing_cols}.")
            # User feedback should be handled in render()
            return pd.DataFrame()

        # Work with a copy of relevant original columns
        df = self.base_df[required_original_cols].copy()

        # Filter out rows where the notes column is NaN or only whitespace
        df = df.dropna(subset=[self.notes_col_orig])
        df = df[df[self.notes_col_orig].astype(str).str.strip() != '']

        # Ensure PnL column is numeric for sorting
        try:
            df[self.pnl_col_orig] = pd.to_numeric(df[self.pnl_col_orig], errors='coerce')
            # Drop rows where PnL could not be converted if that's desired, or handle NaNs in sorting
            # df = df.dropna(subset=[self.pnl_col_orig]) # Optional: remove trades with non-numeric PnL
        except Exception as e:
            logger.error(f"NotesViewer: Could not convert PnL column '{self.pnl_col_orig}' to numeric: {e}")
            # Potentially return empty or raise, depending on desired strictness
            return pd.DataFrame()

        return df

    def render(self) -> None:
        st.subheader("Trade Notes Viewer")

        if self.df_prepared.empty:
            st.info("No trade notes available to display (data might be empty, missing required columns, or all notes are blank).")
            logger.info("NotesViewer: No prepared data to render.")
            return

        # --- Controls: Search and Sort ---
        col1, col2 = st.columns([3, 2])
        with col1:
            search_term = st.text_input(
                "Search Notes:",
                placeholder="Enter keyword(s)...",
                key="notes_viewer_search"
            ).strip()
        with col2:
            # Sort options use display names
            sort_options_display = list(self.display_col_map.values())
            # Ensure default sort is valid
            default_sort_idx = 0
            if self.default_sort_by_display in sort_options_display:
                default_sort_idx = sort_options_display.index(self.default_sort_by_display)

            sort_by_col_display = st.selectbox(
                "Sort by:",
                options=sort_options_display,
                index=default_sort_idx,
                key="notes_viewer_sort_col"
            )
            sort_ascending = st.checkbox("Ascending", value=self.default_sort_ascending, key="notes_viewer_sort_asc")

        # --- Filtering and Sorting ---
        df_to_display = self.df_prepared.copy() # Start with the prepared (numeric PnL) DataFrame

        if search_term:
            try:
                df_to_display = df_to_display[
                    df_to_display[self.notes_col_orig].astype(str).str.contains(search_term, case=False, na=False)
                ]
            except Exception as e:
                logger.error(f"Error during notes search: {e}", exc_info=True)
                st.error("An error occurred while searching notes.")
                return # Stop rendering if search fails badly

        # Map display sort column back to original column name for sorting
        original_sort_col = None
        for orig_col, disp_col in self.display_col_map.items():
            if disp_col == sort_by_col_display:
                original_sort_col = orig_col
                break
        
        if original_sort_col and original_sort_col in df_to_display.columns:
            try:
                # Date column needs special handling for sorting if it's not already datetime
                if original_sort_col == self.date_col_orig:
                    df_to_display[original_sort_col] = pd.to_datetime(df_to_display[original_sort_col], errors='coerce')

                df_to_display = df_to_display.sort_values(
                    by=original_sort_col,
                    ascending=sort_ascending,
                    na_position='last' # Ensure NaNs in sort column are handled consistently
                )
            except Exception as e:
                logger.error(f"Error sorting notes table by {original_sort_col} (display: {sort_by_col_display}): {e}", exc_info=True)
                st.error(f"Could not sort table by {sort_by_col_display}.")
        elif original_sort_col:
             logger.warning(f"Sort column '{original_sort_col}' not found in prepared DataFrame for notes viewer.")


        # --- Prepare final DataFrame for display (rename columns, format) ---
        final_display_df = df_to_display.copy()

        # Format Date for display AFTER sorting
        if self.date_col_orig in final_display_df.columns:
            try:
                final_display_df[self.date_col_orig] = pd.to_datetime(final_display_df[self.date_col_orig]).dt.strftime('%Y-%m-%d %H:%M')
            except: # Keep original if formatting fails
                pass
        
        # Format PnL for display AFTER sorting
        if self.pnl_col_orig in final_display_df.columns:
            final_display_df[self.pnl_col_orig] = final_display_df[self.pnl_col_orig].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")

        # Rename columns to display names
        final_display_df = final_display_df.rename(columns=self.display_col_map)
        
        # Select only display columns in the desired order
        ordered_display_cols = [self.display_col_map[self.date_col_orig]]
        if self.symbol_col_orig: # Check if symbol_col_orig was valid and added
            ordered_display_cols.append(self.display_col_map[self.symbol_col_orig])
        ordered_display_cols.extend([self.display_col_map[self.pnl_col_orig], self.display_col_map[self.notes_col_orig]])
        
        # Filter out columns not in the map (e.g. if a col was dropped)
        final_display_df = final_display_df[[col for col in ordered_display_cols if col in final_display_df.columns]]


        if not final_display_df.empty:
            st.dataframe(
                final_display_df.reset_index(drop=True),
                use_container_width=True,
                height=500,
                hide_index=True,
                column_config={
                    "Trade Note": st.column_config.TextColumn(width="large")
                }
            )
        else:
            st.info("No notes match your search or filter criteria.")
        logger.debug("NotesViewer rendering complete.")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Test Notes Viewer Component (Refined PnL)")

    mock_notes_data = {
        EXPECTED_COLUMNS['date']: pd.to_datetime(['2023-01-02 14:00', '2023-01-01 10:00', '2023-01-03 11:45', '2023-01-01 12:30', '2023-01-02 09:15']),
        EXPECTED_COLUMNS['symbol']: ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'EURUSD'],
        EXPECTED_COLUMNS['pnl']: [90.75, 150.25, -30.20, -75.50, 220.00],
        EXPECTED_COLUMNS['notes']: ["Quick scalp", "Good entry", "Small loss", "Stop loss hit", "Target reached"]
    }
    mock_notes_df = pd.DataFrame(mock_notes_data)
    mock_notes_df.loc[len(mock_notes_df)] = [pd.to_datetime('2023-01-04 10:00'), 'NZDUSD', "non_numeric_pnl", "Note for non-numeric PnL"]


    st.write("### Default Sort (Date, Descending)")
    notes_viewer_default = NotesViewerComponent(mock_notes_df.copy())
    notes_viewer_default.render()

    st.write("### Sort by PnL, Ascending")
    notes_viewer_pnl_asc = NotesViewerComponent(
        mock_notes_df.copy(),
        default_sort_by_display="PnL", # Use display name
        default_sort_ascending=True
    )
    notes_viewer_pnl_asc.render()

    logger.info("NotesViewerComponent (Refined PnL) test complete.")
