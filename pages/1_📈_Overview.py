"""
pages/1_📈_Overview.py

This page provides a high-level overview of trading performance,
focusing on Key Performance Indicators (KPIs) and the equity curve,
and optionally comparing equity against a selected benchmark.
KPIs are now grouped for better readability.
"""
import streamlit as st
import pandas as pd
import numpy as np 
import logging

try:
    # Added KPI_GROUPS_OVERVIEW
    from config import APP_TITLE, EXPECTED_COLUMNS, KPI_CONFIG, KPI_GROUPS_OVERVIEW, AVAILABLE_BENCHMARKS
    from components.kpi_display import KPIClusterDisplay
    from plotting import plot_equity_curve_and_drawdown, plot_equity_vs_benchmark
    from utils.common_utils import display_custom_message
except ImportError as e:
    st.error(f"Overview Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error" 
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Overview Page: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}; KPI_CONFIG = {}; KPI_GROUPS_OVERVIEW = {}; AVAILABLE_BENCHMARKS = {}
    class KPIClusterDisplay:
        def __init__(self, **kwargs): pass
        def render(self): st.warning("KPI Display Component failed to load.")
    def plot_equity_curve_and_drawdown(**kwargs): return None
    def plot_equity_vs_benchmark(**kwargs): return None
    def display_custom_message(msg, type="error"): st.error(msg)
    st.stop()

logger = logging.getLogger(APP_TITLE)

def show_overview_page():
    st.title("📈 Performance Overview")
    logger.info("Rendering Overview Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data to view the overview.", "info")
        return
    if 'kpi_results' not in st.session_state or st.session_state.kpi_results is None:
        display_custom_message("KPI results are not available. Ensure data is processed.", "warning")
        return
    if 'error' in st.session_state.kpi_results: # Check if kpi_results itself is an error dict
        display_custom_message(f"Error in KPI calculation: {st.session_state.kpi_results['error']}", "error")
        return


    filtered_df = st.session_state.filtered_data
    kpi_results = st.session_state.kpi_results
    kpi_confidence_intervals = st.session_state.get('kpi_confidence_intervals', {})
    plot_theme = st.session_state.get('current_theme', 'dark')
    benchmark_daily_returns = st.session_state.get('benchmark_daily_returns')
    selected_benchmark_display_name = st.session_state.get('selected_benchmark_display_name', "Benchmark")
    initial_capital = st.session_state.get('initial_capital', 100000.0)

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Cannot display overview.", "info")
        return

    # --- Display KPIs using KPIClusterDisplay component, now grouped ---
    st.header("Key Performance Indicators")
    
    # Determine how many columns per row for KPI cards (can be dynamic or fixed)
    cols_per_row_setting = 4 # You can adjust this

    for group_name, kpi_keys_in_group in KPI_GROUPS_OVERVIEW.items():
        # Filter kpi_results and kpi_order for the current group
        group_kpi_results = {key: kpi_results[key] for key in kpi_keys_in_group if key in kpi_results}
        
        # Special handling for Benchmark Comparison group: only show if benchmark is selected and data exists
        if group_name == "Benchmark Comparison":
            if benchmark_daily_returns is None or benchmark_daily_returns.empty:
                # Check if any benchmark specific KPIs have non-NaN values (e.g. if they were calculated but benchmark was then deselected)
                # A more robust check might be if st.session_state.selected_benchmark_ticker is "None" or empty
                if all(pd.isna(group_kpi_results.get(key, np.nan)) for key in kpi_keys_in_group):
                    logger.info(f"Skipping '{group_name}' KPI group as no benchmark is selected or data available.")
                    continue 
            # Also skip if all relevant benchmark KPIs are NaN (e.g., calculation failed)
            if not group_kpi_results or all(pd.isna(val) for val in group_kpi_results.values()):
                 logger.info(f"Skipping '{group_name}' KPI group as results are NaN or empty.")
                 continue


        if group_kpi_results: # Only render if there are KPIs to show in this group
            st.subheader(group_name)
            try:
                kpi_cluster = KPIClusterDisplay(
                    kpi_results=group_kpi_results,
                    kpi_definitions=KPI_CONFIG,
                    kpi_order=kpi_keys_in_group, # Use the defined order for this group
                    kpi_confidence_intervals=kpi_confidence_intervals,
                    cols_per_row=cols_per_row_setting
                )
                kpi_cluster.render()
                st.markdown("---") # Separator after each group
            except Exception as e:
                logger.error(f"Error rendering KPI cluster for group '{group_name}': {e}", exc_info=True)
                display_custom_message(f"An error occurred while displaying KPIs for {group_name}: {e}", "error")
    
    # --- Equity Curve and Drawdown Section ---
    # (This section remains largely the same as your latest version)
    st.header("Strategy Performance Charts") # Changed from subheader to header
    st.subheader("Strategy Equity and Drawdown")
    try:
        date_col = EXPECTED_COLUMNS.get('date', 'date')
        cum_pnl_col = 'cumulative_pnl'
        drawdown_pct_col_name = 'drawdown_pct'

        if date_col not in filtered_df.columns:
            display_custom_message(f"Date column ('{date_col}') not found for equity curve.", "error"); return
        
        df_for_plot = filtered_df # Assume data_processing adds necessary columns
        if cum_pnl_col not in df_for_plot.columns:
             pnl_col_orig = EXPECTED_COLUMNS.get('pnl', 'pnl')
             if pnl_col_orig in df_for_plot.columns:
                df_for_plot = df_for_plot.copy()
                df_for_plot[cum_pnl_col] = df_for_plot[pnl_col_orig].cumsum()
             else:
                display_custom_message(f"Cumulative PnL and PnL columns not found.", "error"); return
        
        if drawdown_pct_col_name not in df_for_plot.columns:
            logger.warning(f"OverviewPage: Drawdown column '{drawdown_pct_col_name}' not found in df_for_plot. Drawdown chart might be affected.")
            # The plot_equity_curve_and_drawdown function should handle this gracefully

        equity_fig = plot_equity_curve_and_drawdown(
            df_for_plot,
            date_col=date_col,
            cumulative_pnl_col=cum_pnl_col,
            drawdown_pct_col=drawdown_pct_col_name,
            theme=plot_theme
        )
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True)
        else:
            display_custom_message("Could not generate the equity curve and drawdown chart.", "warning")
    except Exception as e:
        logger.error(f"Error displaying equity curve: {e}", exc_info=True)
        display_custom_message(f"An error occurred displaying the equity curve: {e}", "error")

    # --- Equity vs. Benchmark Section ---
    # (This section remains largely the same as your latest version)
    if benchmark_daily_returns is not None and not benchmark_daily_returns.empty:
        st.subheader(f"Strategy Equity vs. {selected_benchmark_display_name}")
        try:
            date_col = EXPECTED_COLUMNS.get('date')
            cum_pnl_col = 'cumulative_pnl'

            if date_col not in filtered_df.columns or cum_pnl_col not in filtered_df.columns:
                display_custom_message("Required columns for equity vs. benchmark plot are missing.", "error")
            else:
                strategy_equity_series = filtered_df.set_index(date_col)[cum_pnl_col]
                
                strategy_plot_equity = initial_capital + strategy_equity_series
                if not strategy_plot_equity.empty and not filtered_df.empty:
                     first_trade_pnl = filtered_df.iloc[0].get(EXPECTED_COLUMNS.get('pnl', 'pnl'), 0)
                     strategy_plot_equity = strategy_plot_equity - first_trade_pnl # Base for comparison

                benchmark_cumulative_factor = (1 + benchmark_daily_returns).cumprod()
                benchmark_plot_equity = benchmark_cumulative_factor * initial_capital
                
                combined_index = strategy_plot_equity.index.union(benchmark_plot_equity.index).sort_values()
                
                strategy_plot_equity_aligned = strategy_plot_equity.reindex(combined_index).ffill().fillna(initial_capital)
                benchmark_plot_equity_aligned = benchmark_plot_equity.reindex(combined_index).ffill().fillna(initial_capital)

                equity_vs_bench_fig = plot_equity_vs_benchmark(
                    strategy_equity=strategy_plot_equity_aligned,
                    benchmark_cumulative_returns=benchmark_plot_equity_aligned,
                    strategy_name="Strategy Equity",
                    benchmark_name=f"{selected_benchmark_display_name} (Scaled)",
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
