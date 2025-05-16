"""
pages/1_📈_Overview.py

This page provides a high-level overview of trading performance,
focusing on Key Performance Indicators (KPIs) and the equity curve,
and optionally comparing equity against a selected benchmark.
"""
import streamlit as st
import pandas as pd
import numpy as np # For np.nan
import logging

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, KPI_CONFIG, DEFAULT_KPI_DISPLAY_ORDER, AVAILABLE_BENCHMARKS
    from components.kpi_display import KPIClusterDisplay
    from plotting import plot_equity_curve_and_drawdown, plot_equity_vs_benchmark # Added plot_equity_vs_benchmark
    from utils.common_utils import display_custom_message
except ImportError as e:
    st.error(f"Overview Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error" 
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Overview Page: {e}", exc_info=True)
    # Define fallbacks for essential missing imports
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}
    KPI_CONFIG = {}; DEFAULT_KPI_DISPLAY_ORDER = []; AVAILABLE_BENCHMARKS = {}
    class KPIClusterDisplay:
        def __init__(self, **kwargs): pass
        def render(self): st.warning("KPI Display Component failed to load.")
    def plot_equity_curve_and_drawdown(**kwargs): return None
    def plot_equity_vs_benchmark(**kwargs): return None
    def display_custom_message(msg, type="error"): st.error(msg)
    st.stop()

logger = logging.getLogger(APP_TITLE)

def normalize_series(series: pd.Series, start_value: float = 100.0) -> pd.Series:
    """Normalizes a series to start at a specific value (e.g., 100)."""
    if series.empty:
        return pd.Series(dtype=float)
    # Ensure the first value is not zero for percentage change based normalization
    # If first value is 0, and we want to show growth from 0, it's tricky.
    # A common approach is to shift the series if it starts at 0 or less.
    # For simplicity, if we're plotting returns, they should be relative.
    # If we plot equity, it starts at some base.
    # This normalization assumes series represents values that can be scaled.
    # return (series / series.iloc[0]) * start_value if series.iloc[0] != 0 else series # Avoid division by zero

    # Alternative: additive normalization for PnL based equity
    # If series is cumulative PnL, to start it at 'start_value' conceptually for comparison:
    # return series - series.iloc[0] + start_value

    # For % returns based series (like benchmark cumulative returns)
    # (1 + returns).cumprod() already gives a factor. Multiply by start_value.
    # If series is already cumulative returns as a factor (e.g., 1.05 for 5% return), then:
    return series * start_value


def show_overview_page():
    st.title("📈 Performance Overview")
    logger.info("Rendering Overview Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data to view the overview.", "info")
        return
    if 'kpi_results' not in st.session_state or st.session_state.kpi_results is None:
        display_custom_message("KPI results are not available. Ensure data is processed.", "warning")
        return
    if 'error' in st.session_state.kpi_results:
        display_custom_message(f"Error in KPI calculation: {st.session_state.kpi_results['error']}", "error")
        return

    filtered_df = st.session_state.filtered_data
    kpi_results = st.session_state.kpi_results
    kpi_confidence_intervals = st.session_state.get('kpi_confidence_intervals', {})
    plot_theme = st.session_state.get('current_theme', 'dark')
    benchmark_daily_returns = st.session_state.get('benchmark_daily_returns')
    selected_benchmark_display_name = st.session_state.get('selected_benchmark_display_name', "Benchmark")
    initial_capital = st.session_state.get('initial_capital', 100000.0) # Get initial capital

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Cannot display overview.", "info")
        return

    st.subheader("Key Performance Indicators")
    try:
        kpi_cluster = KPIClusterDisplay(
            kpi_results=kpi_results,
            kpi_definitions=KPI_CONFIG,
            kpi_order=DEFAULT_KPI_DISPLAY_ORDER,
            kpi_confidence_intervals=kpi_confidence_intervals,
            cols_per_row=4
        )
        kpi_cluster.render()
    except Exception as e:
        logger.error(f"Error rendering KPI cluster: {e}", exc_info=True)
        display_custom_message(f"An error occurred while displaying KPIs: {e}", "error")

    st.markdown("---")

    # --- Equity Curve and Drawdown Section ---
    st.subheader("Strategy Equity and Drawdown")
    try:
        date_col = EXPECTED_COLUMNS.get('date', 'date')
        cum_pnl_col = 'cumulative_pnl'
        drawdown_pct_col_name = 'drawdown_pct' # This column is now added in data_processing.py

        if date_col not in filtered_df.columns:
            display_custom_message(f"Date column ('{date_col}') not found for equity curve.", "error"); return
        if cum_pnl_col not in filtered_df.columns:
            pnl_col_orig = EXPECTED_COLUMNS.get('pnl', 'pnl')
            if pnl_col_orig in filtered_df.columns:
                # This should ideally not be needed if data_processing is robust
                filtered_df_copy = filtered_df.copy() # Avoid modifying session state df directly
                filtered_df_copy[cum_pnl_col] = filtered_df_copy[pnl_col_orig].cumsum()
                logger.warning("OverviewPage: 'cumulative_pnl' was missing from filtered_df, calculated on-the-fly.")
                df_for_plot = filtered_df_copy
            else:
                display_custom_message(f"Cumulative PnL ('{cum_pnl_col}') and PnL ('{pnl_col_orig}') columns not found.", "error"); return
        else:
            df_for_plot = filtered_df

        equity_fig = plot_equity_curve_and_drawdown(
            df_for_plot, # Use the DataFrame that definitely has cumulative_pnl
            date_col=date_col,
            cumulative_pnl_col=cum_pnl_col,
            drawdown_pct_col=drawdown_pct_col_name, # Pass the correct column name
            theme=plot_theme
        )
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True)
        else:
            display_custom_message("Could not generate the equity curve and drawdown chart.", "warning")
    except Exception as e:
        logger.error(f"Error displaying equity curve: {e}", exc_info=True)
        display_custom_message(f"An error occurred displaying the equity curve: {e}", "error")

    st.markdown("---")

    # --- Equity vs. Benchmark Section ---
    if benchmark_daily_returns is not None and not benchmark_daily_returns.empty:
        st.subheader(f"Strategy Equity vs. {selected_benchmark_display_name}")
        try:
            date_col = EXPECTED_COLUMNS.get('date')
            cum_pnl_col = 'cumulative_pnl'

            if date_col not in filtered_df.columns or cum_pnl_col not in filtered_df.columns:
                display_custom_message("Required columns for equity vs. benchmark plot are missing.", "error")
            else:
                strategy_equity_series = filtered_df.set_index(date_col)[cum_pnl_col]
                
                # Normalize strategy equity to start at initial_capital (or 100 if PnL is already % based)
                # If cum_pnl is absolute, it already reflects growth from an implicit 0 + first trade.
                # To compare with benchmark (which is % return), we should normalize both.
                # Strategy: Start at `initial_capital` and add cumulative PnL.
                strategy_plot_equity = initial_capital + strategy_equity_series
                # Ensure the first point is indeed initial_capital if trades start later
                if not strategy_plot_equity.empty:
                     first_trade_pnl = filtered_df.iloc[0][EXPECTED_COLUMNS.get('pnl', 'pnl')]
                     strategy_plot_equity = strategy_plot_equity - first_trade_pnl # Adjust so first actual value is initial_capital + first_pnl


                # Benchmark: Calculate cumulative returns and scale by initial_capital
                # (1 + daily_returns).cumprod() gives a factor series starting at 1.
                # Multiply by initial_capital to scale it.
                benchmark_cumulative_factor = (1 + benchmark_daily_returns).cumprod()
                benchmark_plot_equity = benchmark_cumulative_factor * initial_capital
                
                # Align indices for plotting (outer join to keep all dates, then ffill)
                # This ensures both series are plotted over a common, continuous date range.
                combined_index = strategy_plot_equity.index.union(benchmark_plot_equity.index).sort_values()
                
                strategy_plot_equity_aligned = strategy_plot_equity.reindex(combined_index).ffill().fillna(initial_capital) # Fill NaNs with initial capital
                benchmark_plot_equity_aligned = benchmark_plot_equity.reindex(combined_index).ffill().fillna(initial_capital)


                equity_vs_bench_fig = plot_equity_vs_benchmark(
                    strategy_equity=strategy_plot_equity_aligned,
                    benchmark_cumulative_returns=benchmark_plot_equity_aligned,
                    strategy_name="Strategy Equity",
                    benchmark_name=f"{selected_benchmark_display_name} (Scaled to Initial Capital)",
                    theme=plot_theme
                )
                if equity_vs_bench_fig:
                    st.plotly_chart(equity_vs_bench_fig, use_container_width=True)
                else:
                    display_custom_message("Could not generate equity vs. benchmark chart.", "warning")
        except Exception as e:
            logger.error(f"Error displaying equity vs. benchmark chart: {e}", exc_info=True)
            display_custom_message(f"An error occurred displaying equity vs. benchmark: {e}", "error")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_overview_page()
