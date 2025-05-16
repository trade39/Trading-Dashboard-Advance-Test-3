"""
data_processing.py

Handles data loading, cleaning, validation, and feature engineering.
Includes URL/text cleaning and robust duration handling.
Also calculates and adds drawdown series to the DataFrame.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple # Added Tuple
import logging
import re

from config import EXPECTED_COLUMNS, APP_TITLE

logger = logging.getLogger(APP_TITLE)

def _calculate_drawdown_series_for_df(cumulative_pnl: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Helper to calculate absolute and percentage drawdown series.
    To be used within data_processing to add columns to the DataFrame.
    """
    if cumulative_pnl.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    high_water_mark = cumulative_pnl.cummax()
    drawdown_abs_series = high_water_mark - cumulative_pnl
    
    # Replace 0s in HWM with NaN before division to avoid 0/0 or X/0 issues
    hwm_for_pct = high_water_mark.replace(0, np.nan) 
    drawdown_pct_series = (drawdown_abs_series / hwm_for_pct).fillna(0) * 100
    
    # Handle edge case where HWM is always 0 but losses occur (e.g. starts at 0, loses money)
    # If drawdown_abs_series shows loss but drawdown_pct_series is 0 due to HWM being 0.
    if (drawdown_abs_series > 0).any() and (drawdown_pct_series[drawdown_abs_series > 0] == 0).all() and (high_water_mark == 0).all():
        # This scenario implies a 100% drawdown relative to the initial (zero) capital for those points.
        # However, plotting this as 100% can be misleading if capital isn't truly zero.
        # For plotting, it's often better to show the absolute drawdown or cap pct.
        # For now, the fillna(0) handles it, but this logic can be refined if needed.
        pass

    return drawdown_abs_series, drawdown_pct_series

def clean_text_column(text_series: pd.Series) -> pd.Series:
    if not isinstance(text_series, pd.Series):
        return pd.Series(text_series, dtype=str)
    
    processed_series = text_series.astype(str).fillna('').str.strip()
    url_pattern = r"\(?https?://[^\s\)\"]+\)?|www\.[^\s\)\"]+"
    notion_link_pattern = r"\(https://www\.notion\.so/[^)]+\)"
    empty_parens_pattern = r"^\(''\)$" 

    def clean_element(text: str) -> Any:
        if pd.isna(text) or text.lower() == 'nan': return pd.NA
        cleaned_text = text
        cleaned_text = re.sub(notion_link_pattern, '', cleaned_text)
        cleaned_text = re.sub(url_pattern, '', cleaned_text)
        cleaned_text = re.sub(empty_parens_pattern, '', cleaned_text) 
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text if cleaned_text else pd.NA

    return processed_series.apply(clean_element)


@st.cache_data(ttl=3600, show_spinner="Loading and processing trade data...")
def load_and_process_data(uploaded_file: Any) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        logger.info("No file uploaded.")
        return None
    try:
        df = pd.read_csv(uploaded_file)
        logger.info(f"Successfully loaded CSV. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}", exc_info=True)
        st.error(f"Error reading CSV: {e}")
        return None

    original_columns = df.columns.tolist()
    df.columns = [
        str(col).strip().lower().replace(' ', '_').replace('\t', '')
        .replace('(', '').replace(')', '').replace('%', 'pct')
        .replace(':', 'rr') 
        for col in original_columns
    ]
    cleaned_columns = df.columns.tolist()
    logger.info(f"Original CSV headers: {original_columns}")
    logger.info(f"Cleaned DataFrame headers in data_processing: {cleaned_columns}")

    critical_expected_keys = ['date', 'pnl']
    missing_critical_info = []
    for key in critical_expected_keys:
        col_name = EXPECTED_COLUMNS.get(key)
        if not col_name:
            missing_critical_info.append(f"Configuration for '{key}' is missing in EXPECTED_COLUMNS.")
        elif col_name not in df.columns:
            missing_critical_info.append(f"Critical column for '{key}' (expected as '{col_name}') not found after cleaning.")
    
    if missing_critical_info:
        error_message = f"Critical columns missing/misconfigured: {', '.join(missing_critical_info)}. Available columns: {df.columns.tolist()}"
        logger.error(error_message)
        st.error(error_message)
        return None

    try:
        date_col = EXPECTED_COLUMNS['date']
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isnull().sum() > 0:
            logger.warning(f"{df[date_col].isnull().sum()} invalid date formats in '{date_col}'. Rows dropped.")
            df.dropna(subset=[date_col], inplace=True)
        if df.empty:
            logger.error("DataFrame empty after dropping invalid dates."); return None

        pnl_col = EXPECTED_COLUMNS['pnl']
        df[pnl_col] = pd.to_numeric(df[pnl_col], errors='coerce')
        if df[pnl_col].isnull().sum() > 0:
            logger.warning(f"{df[pnl_col].isnull().sum()} PnL values NaN after conversion in '{pnl_col}'.")

        for key, expected_name_from_config in EXPECTED_COLUMNS.items():
            if key in critical_expected_keys: continue
            actual_col_name_in_df = expected_name_from_config 
            if actual_col_name_in_df not in df.columns:
                logger.warning(f"Column for '{key}' ('{actual_col_name_in_df}') not found. Skipping specific processing.")
                if key == 'risk': df['risk_numeric_internal'] = 0.0
                elif key == 'duration_minutes': df['duration_minutes_numeric'] = pd.NA
                continue
            series = df[actual_col_name_in_df]
            if key in ['entry_price', 'exit_price', 'risk', 'signal_confidence']:
                df[actual_col_name_in_df] = pd.to_numeric(series, errors='coerce')
            elif key == 'duration_minutes':
                df[actual_col_name_in_df] = pd.to_numeric(series, errors='coerce')
            elif key in ['notes', 'symbol', 'strategy', 'trade_id']:
                df[actual_col_name_in_df] = clean_text_column(series).fillna('N/A')
            else: 
                 df[actual_col_name_in_df] = series.astype(str).fillna('N/A')
        
        risk_col_mapped = EXPECTED_COLUMNS.get('risk')
        if risk_col_mapped and risk_col_mapped in df.columns and pd.api.types.is_numeric_dtype(df[risk_col_mapped]):
            df['risk_numeric_internal'] = df[risk_col_mapped].fillna(0.0)
        elif 'risk_numeric_internal' not in df.columns:
            df['risk_numeric_internal'] = 0.0
        
        duration_col_mapped = EXPECTED_COLUMNS.get('duration_minutes')
        if duration_col_mapped and duration_col_mapped in df.columns and pd.api.types.is_numeric_dtype(df[duration_col_mapped]):
            df['duration_minutes_numeric'] = df[duration_col_mapped].copy().fillna(pd.NA)
        elif 'duration_minutes_numeric' not in df.columns:
            df['duration_minutes_numeric'] = pd.NA
    except Exception as e:
        logger.error(f"Error during type conversion/cleaning: {e}", exc_info=True); st.error(f"Type conversion error: {e}"); return None

    df.sort_values(by=EXPECTED_COLUMNS['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Feature Engineering ---
    try:
        pnl_col_fe = EXPECTED_COLUMNS['pnl'] # Already validated
        df['cumulative_pnl'] = df[pnl_col_fe].cumsum()
        df['win'] = df[pnl_col_fe] > 0
        
        date_col_fe = EXPECTED_COLUMNS['date'] 
        df['trade_hour'] = df[date_col_fe].dt.hour
        df['trade_day_of_week'] = df[date_col_fe].dt.day_name()
        df['trade_month'] = df[date_col_fe].dt.month
        df['trade_year'] = df[date_col_fe].dt.year
        df['trade_date_only'] = df[date_col_fe].dt.date

        # Calculate and add drawdown series to the DataFrame
        if 'cumulative_pnl' in df.columns and not df['cumulative_pnl'].empty:
            df['drawdown_abs'], df['drawdown_pct'] = _calculate_drawdown_series_for_df(df['cumulative_pnl'])
            logger.info("Added 'drawdown_abs' and 'drawdown_pct' columns to DataFrame.")
        else:
            df['drawdown_abs'] = pd.Series(dtype=float)
            df['drawdown_pct'] = pd.Series(dtype=float)
            logger.warning("'cumulative_pnl' not available or empty, drawdown columns will be empty.")

        risk_col_internal_fe = 'risk_numeric_internal'
        if pd.api.types.is_numeric_dtype(df.get(risk_col_internal_fe)): # Use .get for safety
            df['reward_risk_ratio'] = df.apply(
                lambda row: row[pnl_col_fe] / abs(row[risk_col_internal_fe]) 
                            if pd.notna(row[pnl_col_fe]) and pd.notna(row[risk_col_internal_fe]) and abs(row[risk_col_internal_fe]) > 1e-9 
                            else pd.NA,
                axis=1
            )
        else:
            df['reward_risk_ratio'] = pd.NA

        df['trade_number'] = range(1, len(df) + 1)
        logger.info("Feature engineering complete.")
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}", exc_info=True); st.error(f"Feature engineering error: {e}"); return df 

    if df.empty:
        st.warning("No valid trade data found after processing."); return None
        
    logger.info(f"Data processing complete. Final DataFrame shape: {df.shape}. Columns: {df.columns.tolist()}")
    return df
