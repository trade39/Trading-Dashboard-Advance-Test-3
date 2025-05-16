"""
pages/0_❓_User_Guide.py

This page provides a user guide for the Trading Performance Dashboard,
explaining its features, data requirements, and KPI interpretations.
"""
import streamlit as st
import logging

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, KPI_CONFIG
except ImportError:
    # Fallback if config is not available (e.g., when viewing this page standalone, though unlikely)
    APP_TITLE = "Trading Performance Dashboard"
    EXPECTED_COLUMNS = {
        "date": "date", "pnl": "pnl", "symbol": "symbol_1", 
        "entry_price": "entry", "exit_price": "exit", "risk": "risk",
        "notes": "lesson_learned", "strategy": "trade_model",
        "duration_minutes": "duration_mins"
        # Add other critical columns if needed for the guide
    }
    KPI_CONFIG = {} # In a real fallback, you might want some default KPI descriptions

# It's good practice to get the logger, even if not heavily used on a static page
# This maintains consistency with other pages.
logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in locals() else "TradingDashboard_Guide")

def show_user_guide_page():
    """
    Renders the content for the User Guide page.
    """
    st.set_page_config(page_title=f"User Guide - {APP_TITLE}", layout="wide") # Ensure wide layout for guide
    st.title("❓ User Guide & Help")
    logger.info("Rendering User Guide Page.")

    st.markdown("""
    Welcome to the Trading Performance Dashboard! This guide will help you understand how to use the application,
    the data it expects, and how to interpret the various analyses and Key Performance Indicators (KPIs).
    """)

    # --- Section 1: Getting Started ---
    st.header("1. Getting Started")
    st.subheader("1.1. Uploading Your Trading Journal")
    st.markdown(f"""
    To begin, you need to upload your trading journal as a CSV file using the file uploader in the sidebar.

    **Expected CSV Format:**
    The dashboard expects your CSV file to have specific column headers. While the application attempts to clean
    common variations (e.g., converting to lowercase, replacing spaces with underscores), adhering to a consistent
    format will ensure the best results.

    The core columns the application looks for (as configured in `config.py` under `EXPECTED_COLUMNS`) are:
    * **`{EXPECTED_COLUMNS.get('date', 'date')}`**: The timestamp of the trade (e.g., "YYYY-MM-DD HH:MM:SS"). This is crucial for time-based analysis.
    * **`{EXPECTED_COLUMNS.get('pnl', 'pnl')}`**: The Profit or Loss for each trade (numeric value). This is the primary metric for performance.
    * **`{EXPECTED_COLUMNS.get('symbol', 'symbol_1')}`**: The trading instrument/symbol (e.g., "EURUSD", "XAUUSD", "AAPL").
    * **`{EXPECTED_COLUMNS.get('strategy', 'trade_model')}`**: The name or identifier of the trading strategy used.
    * **`{EXPECTED_COLUMNS.get('notes', 'lesson_learned')}`**: Any qualitative notes or lessons learned from the trade.
    * **`{EXPECTED_COLUMNS.get('duration_minutes', 'duration_mins')}`**: Duration of the trade in minutes (numeric).
    * **`{EXPECTED_COLUMNS.get('risk', 'risk')}`**: The amount risked on the trade (numeric, used for R:R calculations). If not provided, some risk-related metrics might be unavailable or calculated differently.
    * Other columns like `{EXPECTED_COLUMNS.get('entry_price', 'entry')}`, `{EXPECTED_COLUMNS.get('exit_price', 'exit')}` can also be utilized.

    *Please ensure your column names in the CSV file correspond to these expected names after basic cleaning (lowercase, underscores for spaces).*
    For example, a CSV column "Trade Model " will be cleaned to "trade_model" by the application.
    Check `config.py` for the exact `EXPECTED_COLUMNS` mapping if you encounter issues.
    """)

    st.subheader("1.2. Using Sidebar Filters")
    st.markdown("""
    Once data is uploaded, you can use the filters in the sidebar to refine the dataset for analysis:
    * **Risk-Free Rate:** Set the annual risk-free rate (as a percentage) used in calculations like the Sharpe Ratio.
    * **Date Range:** Select a specific period for analysis.
    * **Symbol:** Filter trades by a specific trading symbol.
    * **Strategy:** Filter trades by a specific strategy.

    Changes to these filters will dynamically update the analyses and visualizations across all pages.
    """)

    # --- Section 2: Understanding the Pages ---
    st.header("2. Navigating the Dashboard Pages")
    st.markdown("Each page in the sidebar offers a different perspective on your trading performance:")

    with st.expander("📈 Overview Page", expanded=False):
        st.markdown("""
        This page provides a high-level summary of your performance.
        * **Key Performance Indicators (KPIs):** A collection of essential metrics like Total PnL, Win Rate, Profit Factor, Sharpe Ratio, Max Drawdown, etc. Each KPI card may show its value, unit, a qualitative interpretation (e.g., "Good", "Poor"), and a 95% confidence interval if calculated.
        * **Equity Curve & Drawdown Chart:** Visualizes your cumulative profit/loss over time and the percentage drawdown from peak equity.
        """)

    with st.expander("📊 Performance Page", expanded=False):
        st.markdown("""
        Delve deeper into specific aspects of your performance.
        * **PnL Distribution:** A histogram showing the frequency of different PnL values per trade.
        * **PnL by Category:** Bar charts showing total PnL grouped by categories like Day of the Week, Month, or Trading Hour.
        * **Win Rate Analysis:** Similar to PnL by category, but shows win rates for different time segments.
        * **Rolling Performance:** Line charts displaying metrics like Rolling PnL Sum or Rolling Win Rate over a defined window of trades/time.
        * **P&L Calendar Heatmap:** A visual heatmap of daily PnL for a selected year, helping to identify patterns on specific days or weeks.
        """)

    with st.expander("📉 Risk and Duration Page", expanded=False):
        st.markdown("""
        Focuses on risk metrics and trade duration.
        * **Key Risk Metrics:** Displays KPIs specifically related to risk, such as Value at Risk (VaR), Conditional VaR (CVaR), Max Drawdown details, and risk-adjusted return ratios like Sortino and Calmar.
        * **Feature Correlation Matrix:** A heatmap showing the correlation between various numeric features in your trading data (e.g., PnL, duration, risk amount).
        * **Trade Duration Analysis (Survival Curve):** A Kaplan-Meier survival curve illustrating the probability of a trade remaining open over time. It also shows the median trade duration.
        """)

    with st.expander("⚖️ Strategy Comparison Page", expanded=False):
        st.markdown("""
        Compare the performance of different trading strategies side-by-side.
        * **Strategy Selection:** Choose two or more strategies identified in your data.
        * **KPI Comparison Table:** A table showing key KPIs for each selected strategy, with highlighting for best/worst values per KPI.
        * **Comparative Equity Curves:** Plots the equity curves of the selected strategies on a single chart.
        """)

    with st.expander("🔬 Advanced Stats Page", expanded=False):
        st.markdown("""
        Explore more sophisticated statistical analyses.
        * **Bootstrap Confidence Intervals:** Calculate and visualize confidence intervals for selected statistics (e.g., Mean PnL, Win Rate) using bootstrapping. This provides a measure of the statistic's stability and range.
        * **Time Series Decomposition:** Decompose a selected time series (e.g., Equity Curve, Daily PnL) into its trend, seasonal, and residual components.
        * *(Other advanced analyses like Distribution Fitting, Change Point Detection may be available here.)*
        """)
    
    with st.expander("🔮 Stochastic Models Page", expanded=False):
        st.markdown("""
        Simulate and analyze trading performance using stochastic process models.
        * **Geometric Brownian Motion (GBM) Simulation:** Simulate future equity paths based on GBM parameters (drift, volatility).
        * **Markov Chain Analysis:** Analyze sequences of trade outcomes (Win/Loss) to identify patterns in transitions between states.
        * *(Other models like Ornstein-Uhlenbeck or Jump-Diffusion may be available.)*
        """)

    with st.expander("🤖 AI and ML Page", expanded=False):
        st.markdown("""
        Leverage Artificial Intelligence and Machine Learning for deeper insights.
        * **Time Series Forecasting:** Forecast future values of selected series (e.g., Daily PnL, Equity Curve) using models like ARIMA or Prophet.
        * *(Other AI/ML tools like Anomaly Detection or Survival Analysis (e.g., Cox PH) may be available.)*
        """)

    with st.expander("📋 Data View Page", expanded=False):
        st.markdown("""
        Inspect the raw (but cleaned and processed) trading data that is currently being analyzed based on your active filters.
        * **Interactive Table:** View your trade log.
        * **Download Data:** Download the currently filtered dataset as a CSV file.
        """)

    with st.expander("📝 Trade Notes Page", expanded=False):
        st.markdown("""
        Review and search through the qualitative notes associated with your trades.
        * **Search Notes:** Filter notes by keywords.
        * **Sort Notes:** Sort notes by Date, PnL, or Symbol.
        """)

    # --- Section 3: Understanding Key Performance Indicators (KPIs) ---
    st.header("3. Understanding Key Performance Indicators (KPIs)")
    st.markdown("The dashboard uses several KPIs to evaluate trading performance. Here are explanations for some of the common ones:")

    # Dynamically generate KPI explanations from KPI_CONFIG if available and detailed enough
    if KPI_CONFIG:
        for kpi_key, kpi_info in KPI_CONFIG.items():
            kpi_name = kpi_info.get("name", kpi_key.replace("_", " ").title())
            # Try to generate a brief description based on the name.
            # For a real user guide, you'd want more detailed, manually written descriptions.
            description = f"Measures the {kpi_name.lower()}." # Basic placeholder
            if kpi_key == "total_pnl":
                description = "The sum of all profits and losses from trades."
            elif kpi_key == "win_rate":
                description = "The percentage of trades that resulted in a profit."
            elif kpi_key == "profit_factor":
                description = "Gross profit divided by gross loss. A value greater than 1 indicates profitability."
            elif kpi_key == "sharpe_ratio":
                description = "Measures risk-adjusted return, considering volatility. Higher is generally better."
            elif kpi_key == "max_drawdown_pct":
                description = "The largest peak-to-trough percentage decline in equity during a specific period."
            # Add more specific descriptions as needed

            with st.expander(f"{kpi_name} ({kpi_info.get('unit', '')})", expanded=False):
                st.markdown(f"**{kpi_name}**: {description}")
                st.markdown(f"*Interpretation Type:* `{kpi_info.get('interpretation_type', 'N/A')}`")
                if kpi_info.get("thresholds"):
                    st.markdown("*Example Thresholds for Interpretation:*")
                    for label, min_val, max_val_excl in kpi_info.get("thresholds", []):
                        min_str = f"{min_val:,.1f}" if isinstance(min_val, (int, float)) and min_val != float('-inf') else "any"
                        max_str = f"{max_val_excl:,.1f}" if isinstance(max_val_excl, (int, float)) and max_val_excl != float('inf') else "any"
                        if max_val_excl == float('inf'):
                            st.markdown(f"  - **{label}**: If value is >= {min_str}")
                        elif min_val == float('-inf'):
                            st.markdown(f"  - **{label}**: If value is < {max_str}")
                        else:
                            st.markdown(f"  - **{label}**: If value is between {min_str} and {max_str} (exclusive of max)")
    else:
        st.markdown("""
        * **Total PnL ($):** Sum of all profits and losses.
        * **Win Rate (%):** Percentage of profitable trades.
        * **Profit Factor:** Gross Profit / Gross Loss. Higher is better.
        * **Sharpe Ratio:** Risk-adjusted return. Higher is better.
        * **Max Drawdown (%):** Largest peak-to-trough decline in equity. Lower is better.
        * *(Refer to standard financial definitions for detailed explanations of each KPI.)*
        """)
    
    # --- Section 4: Troubleshooting & FAQ ---
    st.header("4. Troubleshooting & FAQ")
    with st.expander("Data Upload Issues", expanded=False):
        st.markdown("""
        * **Error reading CSV:** Ensure your file is a valid CSV. Check for encoding issues (UTF-8 is recommended).
        * **Columns missing/misconfigured:** Verify your CSV column names match those expected by the application (see Section 1.1 or `config.py`). The app attempts to clean names (lowercase, underscores for spaces), but significant deviations can cause problems.
        * **Date parsing errors:** Make sure your date column is in a recognizable format (e.g., YYYY-MM-DD HH:MM:SS, MM/DD/YYYY HH:MM).
        """)
    with st.expander("Analysis Not Appearing", expanded=False):
        st.markdown("""
        * **No data after filtering:** If you apply very restrictive filters, there might be no trades matching the criteria. Try adjusting the filters.
        * **Insufficient data for specific analyses:** Some advanced analyses (e.g., certain statistical tests, time series models) require a minimum number of data points. The application usually provides a warning if this is the case.
        """)
    with st.expander("Understanding 'N/A' or 'Inf' in KPIs", expanded=False):
        st.markdown("""
        * **N/A (Not Available):** This typically means the KPI could not be calculated, often due to insufficient data (e.g., trying to calculate standard deviation on a single trade) or missing required input data for that specific KPI.
        * **Inf (Infinity):** This can occur in ratios like Profit Factor if Gross Loss is zero (resulting in division by zero). If Gross Profit is positive, this is a good sign. If both are zero, it might also show as Inf or NaN.
        """)

    st.markdown("---")
    st.markdown("We hope this guide helps you make the most of the Trading Performance Dashboard!")

# --- Main execution for the page ---
if __name__ == "__main__":
    # This ensures that the page's content is rendered when the page is selected in a multi-page app context
    # or when the script is run directly (though direct run is less common for pages).
    if 'app_initialized' not in st.session_state: # Basic check for multi-page context
        st.warning("This page is part of a multi-page app. For full functionality, please run the main `app.py` script.")
        # You could initialize minimal session state here if needed for standalone testing of this page
        # For example: st.session_state.current_theme = 'dark'
    
    show_user_guide_page()
