"""
utils/common_utils.py

Contains common helper functions used across the Trading Performance Dashboard application.
This includes functions for loading CSS, displaying UI elements like KPI cards,
formatting data, and other general utility functions.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple, Dict
import logging

# Assuming config.py is in the root directory for COLORS
# This import will be resolved when app.py (in root) imports this module.
try:
    from config import COLORS, APP_TITLE # APP_TITLE for logger default
except ImportError:
    # Fallback for standalone testing or if config path is an issue
    print("Warning (common_utils): Could not import from config. Using default UI colors/logger name.", file=sys.stderr)
    APP_TITLE = "TradingDashboard_Default"
    COLORS = {
        "royal_blue": "#4169E1", "green": "#00FF00", "red": "#FF0000",
        "gray": "#808080", "dark_background": "#1C2526",
        "text_dark": "#FFFFFF", "text_muted_color": "#A0A0A0",
        "card_border_dark": "#4169E1"
    }

# Get a logger instance. It will use the configuration provided by app.py
# when setup_logger is called there. If this module is imported before app.py
# fully configures the main logger, it might get a default one.
# Best practice: app.py configures the main logger, other modules retrieve it.
logger = logging.getLogger(APP_TITLE) # Use the app's main logger name


def load_css(file_name: str) -> None:
    """
    Loads a CSS file into the Streamlit app.

    Args:
        file_name (str): The path to the CSS file.
    """
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        logger.info(f"Successfully loaded CSS file: {file_name}")
    except FileNotFoundError:
        logger.warning(f"CSS file not found: {file_name}. Custom styles may not be applied.")
    except Exception as e:
        logger.error(f"Error loading CSS file {file_name}: {e}", exc_info=True)


def display_kpi_card(
    title: str,
    value: Any,
    unit: str = "",
    interpretation: str = "",
    interpretation_desc: str = "",
    color: str = COLORS.get("gray", "#808080"), # Use .get for safety
    confidence_interval: Optional[Tuple[float, float]] = None,
    key_suffix: str = ""
) -> None:
    """
    Displays a single KPI card using Streamlit Markdown with custom HTML/CSS.

    Args:
        title (str): The title of the KPI.
        value (Any): The value of the KPI.
        unit (str): The unit for the KPI value (e.g., "$", "%").
        interpretation (str): Qualitative interpretation of the KPI.
        interpretation_desc (str): Detailed description or threshold for the interpretation.
        color (str): Hex color code for the KPI value, influencing border.
        confidence_interval (Optional[Tuple[float, float]]): Bootstrapped 95% CI.
        key_suffix (str): Unique suffix for Streamlit keys if needed within loops.
    """
    try:
        if pd.isna(value) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            formatted_value = "N/A"
            # color = COLORS.get("gray", "#808080") # Value is NA, color should be neutral
        elif isinstance(value, (int, float)):
            # Use specific formatters if available
            if unit == "$":
                formatted_value = format_currency(value)
            elif unit == "%":
                formatted_value = format_percentage(value / 100 if value > 1 else value) # Assume value is 50 for 50%, not 0.5
            else:
                formatted_value = f"{value:,.2f}{unit}"
        else:
            formatted_value = f"{str(value)}{unit}"

        color_class = "neutral"
        # Ensure COLORS keys exist before comparing
        if color.upper() == COLORS.get("green", "#00FF00").upper():
            color_class = "positive"
        elif color.upper() == COLORS.get("red", "#FF0000").upper():
            color_class = "negative"

        card_html = f"""
        <div class="kpi-card {color_class}" key="kpi-card-{title.replace(' ','_')}-{key_suffix}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value" style="color:{color};">{formatted_value}</div>
            <div class="kpi-interpretation">{interpretation}</div>
            <div class="kpi-interpretation" style="font-size:0.7rem; color:{COLORS.get('text_muted_color', COLORS.get('gray', '#808080'))};">{interpretation_desc}</div>
        """
        if confidence_interval and not any(pd.isna(ci_val) for ci_val in confidence_interval):
            ci_text = f"95% CI: [{confidence_interval[0]:.2f}{unit}, {confidence_interval[1]:.2f}{unit}]"
            card_html += f"""<div class="kpi-confidence-interval">{ci_text}</div>"""
        card_html += "</div>"

        st.markdown(card_html, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error displaying KPI card for '{title}': {e}", exc_info=True)
        # Avoid st.error here if this util is used in non-UI contexts, though for display_kpi_card it's UI-bound.
        # st.error(f"Could not display KPI: {title}") # This could be problematic if called from a non-streamlit thread.


def display_custom_message(
    message: str,
    message_type: str = "info", # "info", "success", "warning", "error"
    icon: Optional[str] = None
) -> None:
    """
    Displays a custom styled message box.
    Relies on .message-box .info, .success etc. classes in style.css.

    Args:
        message (str): The message content.
        message_type (str): Type of message, determines styling.
        icon (Optional[str]): Emoji icon to prepend.
    """
    icon_map = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    display_icon = icon if icon else icon_map.get(message_type, "")

    message_html = f"""
    <div class="message-box {message_type.lower()}">
        {display_icon} {message}
    </div>
    """
    st.markdown(message_html, unsafe_allow_html=True)


def format_currency(value: float, currency_symbol: str = "$", decimals: int = 2) -> str:
    """
    Formats a float value as a currency string.

    Args:
        value (float): The numeric value.
        currency_symbol (str): The currency symbol.
        decimals (int): Number of decimal places.

    Returns:
        str: Formatted currency string. Returns "N/A" if value is NaN or Inf.
    """
    if pd.isna(value) or np.isinf(value):
        return "N/A"
    return f"{currency_symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formats a float value (assumed to be a proportion, e.g., 0.25 for 25%) as a percentage string.

    Args:
        value (float): The numeric value (e.g., 0.25).
        decimals (int): Number of decimal places for the percentage.

    Returns:
        str: Formatted percentage string (e.g., "25.00%"). Returns "N/A" if value is NaN or Inf.
    """
    if pd.isna(value) or np.isinf(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


if __name__ == "__main__":
    logger.info("--- Testing common_utils.py (enhanced) ---")
    logger.info(f"Formatted currency: {format_currency(12345.6789)}")
    logger.info(f"Formatted currency (negative): {format_currency(-987.65)}")
    logger.info(f"Formatted percentage: {format_percentage(0.85678)}")
    logger.info(f"Formatted percentage (small): {format_percentage(0.00123, decimals=3)}")
    logger.info(f"Formatted NaN currency: {format_currency(np.nan)}")
    logger.info(f"Formatted Inf percentage: {format_percentage(np.inf)}")
    logger.info("--- common_utils.py (enhanced) testing complete ---")

