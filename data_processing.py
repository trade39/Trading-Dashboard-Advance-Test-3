"""
data_processing.py

Handles data loading, cleaning, validation, and feature engineering.
Includes URL/text cleaning and robust duration handling.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
import re

from config import EXPECTED_COLUMNS, APP_TITLE

logger = logging.getLogger(APP_TITLE)

def clean_text_column(text_series: pd.Series) -> pd.Series:
    if not isinstance(text_series, pd.Series):
        return pd.Series(text_series, dtype=str)
    
    processed_series = text_series.astype(str).fillna('').str.strip()
    url_pattern = r"\(?https?://[^\s\)\"]+\)?|www\.[^\s\)\"]+"
    notion_link_pattern = r"\(https://www\.notion\.so/[^)]+\)"
    empty_parens_pattern = r"^\(''\)$" # Matches exactly "('')"

    def clean_element(text: str) -> Any:
        if pd.isna(text) or text.lower() == 'nan': return pd.NA
        cleaned_text = text
        cleaned_text = re.sub(notion_link_pattern, '', cleaned_text)
        cleaned_text = re.sub(url_pattern, '', cleaned_text)
        cleaned_text = re.sub(empty_parens_pattern, '', cleaned_text) # Remove "('')"
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text if cleaned_text else pd.NA

    return processed_series.apply(clean_element)


@st.cache_data(ttl=3600, show_spinner="Loading and processing trade data...")
def load_and_process_data(uploaded_file: Any) -> Optional[pd.DataFrame]:
    if uploaded_file is None: logger.info("No file uploaded."); return None
    try:
        df = pd.read_csv(uploaded_file)
        logger.info(f"Successfully loaded CSV. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}", exc_info=True); st.error(f"Error reading CSV: {e}"); return None

    original_columns = df.columns.tolist()
    # Clean column headers: strip, lower, replace space with underscore, remove problematic chars
    df.columns = [str(col).strip().lower().replace(' ', '_').replace('\t', '').replace('(', '').replace(')', '').replace('%', 'pct').replace(':', 'rr') for col in original_columns]
    cleaned_columns = df.columns.tolist()
    logger.info(f"Original CSV headers: {original_columns}")
    logger.info(f"Cleaned DataFrame headers: {cleaned_columns}")

    # --- Column Validation ---
    critical_expected_keys = ['date', 'pnl']
    missing_critical_info = []
    for key in critical_expected_keys:
        col_name = EXPECTED_COLUMNS.get(key) # This is the *cleaned* name expected
        if not col_name: missing_critical_info.append(f"config for '{key}' is missing")
        elif col_name not in df.columns: missing_critical_info.append(f"'{key}' (expected as '{col_name}')")
    if missing_critical_info:
        st.error(f"Critical columns missing/misconfigured: {', '.join(missing_critical_info)}. Available columns: {df.columns.tolist()}"); return None

    # --- Data Type Conversion and Cleaning ---
    try:
        date_col = EXPECTED_COLUMNS['date']
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isnull().sum() > 0:
            logger.warning(f"{df[date_col].isnull().sum()} invalid date formats in '{date_col}'. Dropping rows.")
            df.dropna(subset=[date_col], inplace=True)
        if df.empty: logger.error("DataFrame empty after dropping invalid dates."); return None

        pnl_col = EXPECTED_COLUMNS['pnl']
        df[pnl_col] = pd.to_numeric(df[pnl_col], errors='coerce')
        if df[pnl_col].isnull().sum() > 0: logger.warning(f"{df[pnl_col].isnull().sum()} PnL values NaN after conversion.")

        for key, expected_name_from_config in EXPECTED_COLUMNS.items():
            if key in ['date', 'pnl']: continue
            
            # This is the name we expect to find in df.columns after cleaning
            actual_col_name_in_df = expected_name_from_config 
            
            if actual_col_name_in_df not in df.columns:
                logger.warning(f"Column for '{key}' (configured as '{actual_col_name_in_df}') not found in DataFrame. Skipping its specific processing.")
                # Create placeholder columns if they are essential for schema but missing
                if key == 'risk': df['risk_numeric_internal'] = 0.0
                elif key == 'duration_minutes': df['duration_minutes_numeric'] = pd.NA
                # else: df[actual_col_name_in_df] = pd.NA # Avoid creating if not strictly needed
                continue

            series = df[actual_col_name_in_df]
            if key in ['entry_price', 'exit_price', 'risk', 'signal_confidence']:
                df[actual_col_name_in_df] = pd.to_numeric(series, errors='coerce')
            elif key == 'duration_minutes': # Specifically 'duration_minutes' from config
                df[actual_col_name_in_df] = pd.to_numeric(series, errors='coerce')
                logger.info(f"Processed duration column '{actual_col_name_in_df}' as numeric.")
            elif key == 'notes' or key == 'symbol' or key == 'strategy' or key == 'trade_id':
                df[actual_col_name_in_df] = clean_text_column(series).fillna('N/A')
            else: # Default to string for other configured columns if not specifically handled
                 df[actual_col_name_in_df] = series.astype(str).fillna('N/A')
        
        # Ensure 'risk_numeric_internal' column exists
        risk_col_mapped = EXPECTED_COLUMNS.get('risk')
        if risk_col_mapped and risk_col_mapped in df.columns and pd.api.types.is_numeric_dtype(df[risk_col_mapped]):
            df['risk_numeric_internal'] = df[risk_col_mapped].fillna(0.0)
        else:
            df['risk_numeric_internal'] = 0.0
            if risk_col_mapped: logger.info(f"Risk column '{risk_col_mapped}' used for 'risk_numeric_internal' was not numeric or not found; defaulted to 0.0.")
            else: logger.info("'risk_numeric_internal' column created and set to 0.0 as 'risk' not configured/found.")

    except Exception as e:
        logger.error(f"Error during type conversion/cleaning: {e}", exc_info=True); st.error(f"Type conversion error: {e}"); return None

    df.sort_values(by=EXPECTED_COLUMNS['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Feature Engineering ---
    try:
        df['cumulative_pnl'] = df[EXPECTED_COLUMNS['pnl']].cumsum()
        df['win'] = df[EXPECTED_COLUMNS['pnl']] > 0
        date_col_fe = EXPECTED_COLUMNS['date']
        df['trade_hour'] = df[date_col_fe].dt.hour
        df['trade_day_of_week'] = df[date_col_fe].dt.day_name()
        df['trade_month'] = df[date_col_fe].dt.month
        df['trade_year'] = df[date_col_fe].dt.year
        df['trade_date_only'] = df[date_col_fe].dt.date

        # Standardized numeric duration column for analysis
        # It uses the column name mapped by EXPECTED_COLUMNS['duration_minutes']
        duration_col_from_config = EXPECTED_COLUMNS.get('duration_minutes') # e.g., 'duration_(mins)'
        if duration_col_from_config and duration_col_from_config in df.columns and pd.api.types.is_numeric_dtype(df[duration_col_from_config]):
            df['duration_minutes_numeric'] = df[duration_col_from_config].copy()
            logger.info(f"Using data from column '{duration_col_from_config}' for 'duration_minutes_numeric'.")
        else:
            df['duration_minutes_numeric'] = pd.NA
            logger.warning(f"Configured duration column '{duration_col_from_config}' for 'duration_minutes' not found or not numeric. 'duration_minutes_numeric' is NA.")
        
        pnl_col_fe = EXPECTED_COLUMNS['pnl']
        risk_col_internal = 'risk_numeric_internal'
        if risk_col_internal in df.columns:
            df['reward_risk_ratio'] = df.apply(
                lambda row: row[pnl_col_fe] / abs(row[risk_col_internal]) if pd.notna(row[pnl_col_fe]) and pd.notna(row[risk_col_internal]) and abs(row[risk_col_internal]) > 1e-9 else pd.NA,
                axis=1 )
        else: df['reward_risk_ratio'] = pd.NA

        df['trade_number'] = range(1, len(df) + 1)
        logger.info("Feature engineering complete.")

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}", exc_info=True); st.error(f"Feature engineering error: {e}"); return df

    if df.empty: st.warning("No valid trade data after processing."); return None
    return df
