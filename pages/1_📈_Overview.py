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
    if 'error' in st.session_state.kpi_results: 
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

    st.header("Key Performance Indicators")
    cols_per_row_setting = 4

    for group_name, kpi_keys_in_group in KPI_GROUPS_OVERVIEW.items():
        group_kpi_results = {key: kpi_results[key] for key in kpi_keys_in_group if key in kpi_results}
        
        if group_name == "Benchmark Comparison":
            if benchmark_daily_returns is None or benchmark_daily_returns.empty:
                if all(pd.isna(group_kpi_results.get(key, np.nan)) for key in kpi_keys_in_group):
                    logger.info(f"Skipping '{group_name}' KPI group as no benchmark is selected or data available.")
                    continue 
            if not group_kpi_results or all(pd.isna(val) for val in group_kpi_results.values()):
                 logger.info(f"Skipping '{group_name}' KPI group as results are NaN or empty.")
                 continue

        if group_kpi_results: 
            st.subheader(group_name)
            try:
                kpi_cluster = KPIClusterDisplay(
                    kpi_results=group_kpi_results,
                    kpi_definitions=KPI_CONFIG,
                    kpi_order=kpi_keys_in_group, 
                    kpi_confidence_intervals=kpi_confidence_intervals,
                    cols_per_row=cols_per_row_setting
                )
                kpi_cluster.render()
                st.markdown("---") 
            except Exception as e:
                logger.error(f"Error rendering KPI cluster for group '{group_name}': {e}", exc_info=True)
                display_custom_message(f"An error occurred while displaying KPIs for {group_name}: {e}", "error")
    
    st.header("Strategy Performance Charts")
    st.subheader("Strategy Equity and Drawdown")
    try:
        date_col = EXPECTED_COLUMNS.get('date', 'date')
        cum_pnl_col = 'cumulative_pnl'
        drawdown_pct_col_name = 'drawdown_pct'

        if date_col not in filtered_df.columns:
            display_custom_message(f"Date column ('{date_col}') not found for equity curve.", "error"); return
        
        df_for_plot = filtered_df 
        if cum_pnl_col not in df_for_plot.columns:
             pnl_col_orig = EXPECTED_COLUMNS.get('pnl', 'pnl')
             if pnl_col_orig in df_for_plot.columns:
                df_for_plot = df_for_plot.copy()
                df_for_plot[cum_pnl_col] = df_for_plot[pnl_col_orig].cumsum()
             else:
                display_custom_message(f"Cumulative PnL and PnL columns not found.", "error"); return
        
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

    if benchmark_daily_returns is not None and not benchmark_daily_returns.empty:
        st.subheader(f"Strategy Equity vs. {selected_benchmark_display_name}")
        try:
            date_col = EXPECTED_COLUMNS.get('date')
            cum_pnl_col = 'cumulative_pnl' # This is absolute PnL over time

            if date_col not in filtered_df.columns or cum_pnl_col not in filtered_df.columns:
                display_custom_message("Required columns for equity vs. benchmark plot are missing.", "error")
            else:
                # 1. Strategy Equity Series (Absolute Value)
                # Starts at initial_capital and grows/shrinks with cumulative PnL
                strategy_cum_pnl_series = filtered_df.set_index(date_col)[cum_pnl_col]
                # To make it start from initial_capital:
                # The first value of cum_pnl_series is the PnL of the first trade.
                # Equity = initial_capital + cum_pnl
                # We need to adjust if cum_pnl doesn't start from 0 relative to the first trade.
                # A simpler way: initial_capital + (cum_pnl - first_cum_pnl_value_if_not_0 + first_pnl_value)
                # Or, more directly:
                strategy_equity_values = initial_capital + strategy_cum_pnl_series
                
                logger.debug(f"Raw strategy equity values (initial_capital + cum_pnl) head:\n{strategy_equity_values.head()}")


                # 2. Benchmark Equity Series (Scaled to Initial Capital)
                # benchmark_daily_returns is already % change
                benchmark_cumulative_growth_factor = (1 + benchmark_daily_returns).cumprod()
                # The first value of cumprod will be (1 + first_return). To make it start at 1:
                if not benchmark_cumulative_growth_factor.empty:
                    # Prepend a "1" at the start date of the benchmark series for normalization
                    first_bm_date = benchmark_daily_returns.index.min()
                    # Create a series that starts with 1, then cumprod
                    # This ensures benchmark growth starts from a factor of 1
                    bm_returns_for_factor = benchmark_daily_returns.copy()
                    # If the first day's return is NaN (from pct_change), cumprod might start with NaN.
                    # Fill first NaN with 0 to ensure cumprod starts correctly.
                    if pd.isna(bm_returns_for_factor.iloc[0]):
                        bm_returns_for_factor.iloc[0] = 0.0
                    
                    benchmark_cumulative_growth_factor = (1 + bm_returns_for_factor).cumprod()
                    benchmark_plot_equity = benchmark_cumulative_growth_factor * initial_capital
                else:
                    benchmark_plot_equity = pd.Series(dtype=float)

                logger.debug(f"Benchmark daily returns head:\n{benchmark_daily_returns.head()}")
                logger.debug(f"Benchmark cumulative growth factor head:\n{benchmark_cumulative_growth_factor.head() if not benchmark_cumulative_growth_factor.empty else 'Empty'}")
                logger.debug(f"Benchmark plot equity (scaled) head:\n{benchmark_plot_equity.head() if not benchmark_plot_equity.empty else 'Empty'}")

                # 3. Align and Plot
                if not strategy_equity_values.empty or not benchmark_plot_equity.empty:
                    # Create a common date index based on the union of both series' indices
                    # This ensures both lines are plotted over the full available range.
                    common_min_date = min(strategy_equity_values.index.min() if not strategy_equity_values.empty else pd.Timestamp.max,
                                          benchmark_plot_equity.index.min() if not benchmark_plot_equity.empty else pd.Timestamp.max)
                    common_max_date = max(strategy_equity_values.index.max() if not strategy_equity_values.empty else pd.Timestamp.min,
                                          benchmark_plot_equity.index.max() if not benchmark_plot_equity.empty else pd.Timestamp.min)

                    if common_min_date > common_max_date: # Should not happen if at least one series has data
                         display_custom_message("Cannot align strategy and benchmark due to date issues.", "warning"); return

                    # Reindex both series to this common index, then forward-fill
                    # For points where one series exists but the other doesn't yet, ffill will carry forward.
                    # For points before a series starts, fill with initial_capital.
                    
                    # Strategy alignment:
                    strategy_plot_equity_aligned = strategy_equity_values.reindex(pd.date_range(start=common_min_date, end=common_max_date, freq='B')) # Use Business Day freq
                    strategy_plot_equity_aligned = strategy_plot_equity_aligned.ffill().fillna(initial_capital)
                    
                    # Benchmark alignment:
                    benchmark_plot_equity_aligned = benchmark_plot_equity.reindex(strategy_plot_equity_aligned.index) # Align to strategy's final index
                    benchmark_plot_equity_aligned = benchmark_plot_equity_aligned.ffill()
                    # If benchmark starts later than strategy, fill initial NaNs with initial_capital
                    if not benchmark_plot_equity_aligned.empty and pd.isna(benchmark_plot_equity_aligned.iloc[0]):
                         benchmark_plot_equity_aligned.iloc[0] = initial_capital
                         benchmark_plot_equity_aligned = benchmark_plot_equity_aligned.ffill() # Re-ffill after setting first point
                    benchmark_plot_equity_aligned = benchmark_plot_equity_aligned.fillna(initial_capital) # Catch any remaining NaNs


                    logger.debug(f"Aligned strategy equity head:\n{strategy_plot_equity_aligned.head()}")
                    logger.debug(f"Aligned benchmark equity head:\n{benchmark_plot_equity_aligned.head()}")
                    logger.debug(f"Aligned strategy equity tail:\n{strategy_plot_equity_aligned.tail()}")
                    logger.debug(f"Aligned benchmark equity tail:\n{benchmark_plot_equity_aligned.tail()}")
                    logger.debug(f"Aligned strategy equity describe:\n{strategy_plot_equity_aligned.describe()}")
                    logger.debug(f"Aligned benchmark equity describe:\n{benchmark_plot_equity_aligned.describe()}")


                    equity_vs_bench_fig = plot_equity_vs_benchmark(
                        strategy_equity=strategy_plot_equity_aligned,
                        benchmark_cumulative_returns=benchmark_plot_equity_aligned, # Pass the scaled equity
                        strategy_name="Strategy Equity",
                        benchmark_name=f"{selected_benchmark_display_name} (Scaled Equity)",
                        theme=plot_theme
                    )
                    if equity_vs_bench_fig:
                        st.plotly_chart(equity_vs_bench_fig, use_container_width=True)
                    else:
                        display_custom_message("Could not generate equity vs. benchmark chart.", "warning")
                else:
                    display_custom_message("Not enough data for strategy or benchmark to plot comparison.", "info")
        except Exception as e:
            logger.error(f"Error displaying equity vs. benchmark chart: {e}", exc_info=True)
            display_custom_message(f"An error occurred displaying equity vs. benchmark: {e}", "error")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_overview_page()
