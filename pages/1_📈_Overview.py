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

    st.markdown("---")

    if benchmark_daily_returns is not None and not benchmark_daily_returns.empty:
        st.subheader(f"Strategy Equity vs. {selected_benchmark_display_name}")
        try:
            date_col = EXPECTED_COLUMNS.get('date')
            cum_pnl_col = 'cumulative_pnl' 

            if date_col not in filtered_df.columns or cum_pnl_col not in filtered_df.columns:
                display_custom_message("Required columns for equity vs. benchmark plot are missing.", "error")
            else:
                strategy_cum_pnl_series = filtered_df.set_index(date_col)[cum_pnl_col]
                strategy_plot_equity = initial_capital + strategy_cum_pnl_series
                
                # Ensure benchmark_daily_returns is a Series
                bm_daily_returns_series = benchmark_daily_returns
                if isinstance(bm_daily_returns_series, pd.DataFrame):
                    bm_daily_returns_series = bm_daily_returns_series.squeeze() # Convert to Series if it's a single-column DataFrame
                if not isinstance(bm_daily_returns_series, pd.Series):
                    logger.error(f"Benchmark daily returns is not a Series. Type: {type(bm_daily_returns_series)}")
                    display_custom_message("Benchmark data is not in the correct format (Series expected).", "error")
                    return # Stop further processing for this plot

                benchmark_plot_equity = pd.Series(dtype=float) # Initialize
                if not bm_daily_returns_series.empty:
                    bm_returns_for_factor = bm_daily_returns_series.copy()
                    if not bm_returns_for_factor.empty and pd.isna(bm_returns_for_factor.iloc[0]):
                        bm_returns_for_factor.iloc[0] = 0.0
                    
                    benchmark_cumulative_growth_factor = (1 + bm_returns_for_factor).cumprod()
                    if not benchmark_cumulative_growth_factor.empty:
                         benchmark_plot_equity = benchmark_cumulative_growth_factor * initial_capital
                
                logger.debug(f"Strategy plot equity head:\n{strategy_plot_equity.head() if not strategy_plot_equity.empty else 'Empty'}")
                logger.debug(f"Benchmark plot equity head:\n{benchmark_plot_equity.head() if not benchmark_plot_equity.empty else 'Empty'}")

                strategy_plot_equity_aligned = pd.Series(dtype=float)
                benchmark_plot_equity_aligned = pd.Series(dtype=float)

                if not strategy_plot_equity.empty or not benchmark_plot_equity.empty:
                    # Determine common date range using min/max of available indices
                    all_dates = pd.Index([])
                    if not strategy_plot_equity.empty:
                        all_dates = all_dates.union(strategy_plot_equity.index)
                    if not benchmark_plot_equity.empty:
                        all_dates = all_dates.union(benchmark_plot_equity.index)
                    
                    if not all_dates.empty:
                        common_min_date = all_dates.min()
                        common_max_date = all_dates.max()
                        
                        # Create a business day index over this common range
                        aligned_index = pd.date_range(start=common_min_date, end=common_max_date, freq='B')

                        if not strategy_plot_equity.empty:
                            strategy_plot_equity_aligned = strategy_plot_equity.reindex(aligned_index).ffill().fillna(initial_capital)
                        else: # If strategy equity is empty, fill with initial capital for alignment
                            strategy_plot_equity_aligned = pd.Series(initial_capital, index=aligned_index)


                        if not benchmark_plot_equity.empty:
                            benchmark_plot_equity_aligned = benchmark_plot_equity.reindex(aligned_index).ffill()
                            # If benchmark starts later, its initial NaNs should be initial_capital
                            if not benchmark_plot_equity_aligned.empty and pd.isna(benchmark_plot_equity_aligned.iloc[0]):
                                benchmark_plot_equity_aligned.iloc[0] = initial_capital
                                benchmark_plot_equity_aligned = benchmark_plot_equity_aligned.ffill()
                            benchmark_plot_equity_aligned = benchmark_plot_equity_aligned.fillna(initial_capital)
                        else: # If benchmark equity is empty, fill with initial capital
                            benchmark_plot_equity_aligned = pd.Series(initial_capital, index=aligned_index)
                    else: # Both are empty, or some other issue
                         display_custom_message("Not enough date information to align strategy and benchmark.", "warning"); return


                    logger.debug(f"Final Aligned strategy equity head:\n{strategy_plot_equity_aligned.head() if not strategy_plot_equity_aligned.empty else 'Empty'}")
                    logger.debug(f"Final Aligned benchmark equity head:\n{benchmark_plot_equity_aligned.head() if not benchmark_plot_equity_aligned.empty else 'Empty'}")

                    if not strategy_plot_equity_aligned.empty or not benchmark_plot_equity_aligned.empty:
                        equity_vs_bench_fig = plot_equity_vs_benchmark(
                            strategy_equity=strategy_plot_equity_aligned,
                            benchmark_cumulative_returns=benchmark_plot_equity_aligned,
                            strategy_name="Strategy Equity",
                            benchmark_name=f"{selected_benchmark_display_name} (Scaled Equity)",
                            theme=plot_theme
                        )
                        if equity_vs_bench_fig:
                            st.plotly_chart(equity_vs_bench_fig, use_container_width=True)
                        else:
                            display_custom_message("Could not generate equity vs. benchmark chart.", "warning")
                    else:
                        display_custom_message("No data to plot for equity vs. benchmark after alignment.", "info")
                else:
                    display_custom_message("Not enough data for strategy or benchmark to plot comparison.", "info")
        except Exception as e:
            logger.error(f"Error displaying equity vs. benchmark chart: {e}", exc_info=True)
            display_custom_message(f"An error occurred displaying equity vs. benchmark: {e}", "error")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_overview_page()
