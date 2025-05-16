"""
pages/2_📊_Performance.py

This page delves into detailed performance metrics and visualizations,
such as PnL distributions, categorical PnL analysis, win rates by time,
and a P&L calendar view.
"""
import streamlit as st
import pandas as pd
import logging

# --- Assuming root-level modules are accessible ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS
    from plotting import (
        plot_pnl_distribution, plot_pnl_by_category,
        plot_win_rate_analysis, plot_rolling_performance
    )
    from components.calendar_view import PnLCalendarComponent
    from utils.common_utils import display_custom_message
    # from services.analysis_service import AnalysisService # If using service for plot generation
except ImportError as e:
    st.error(f"Performance Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error" # Placeholder
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Performance Page: {e}", exc_info=True)
    st.stop()

logger = logging.getLogger(APP_TITLE)

def show_performance_page():
    """
    Renders the content for the Performance Details page.
    """
    st.title("📊 Detailed Performance Analysis")
    logger.info("Rendering Performance Page.")

    # --- Check for necessary data in session state ---
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data in the main application to view performance details.", "info")
        return
    
    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    pnl_col = EXPECTED_COLUMNS.get('pnl', 'pnl') # Get PnL column name from config

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Cannot display performance details.", "info")
        return
    if pnl_col not in filtered_df.columns:
        display_custom_message(f"PnL column ('{pnl_col}') not found in data. Performance analysis cannot proceed.", "error")
        return

    # --- Performance Breakdown Section ---
    st.subheader("Performance Breakdown")
    try:
        col1, col2 = st.columns(2)
        with col1:
            # PnL Distribution
            pnl_dist_fig = plot_pnl_distribution(filtered_df, pnl_col=pnl_col, theme=plot_theme)
            if pnl_dist_fig:
                st.plotly_chart(pnl_dist_fig, use_container_width=True)
            else:
                display_custom_message("Could not generate PnL distribution plot.", "warning")

            # PnL by Day of Week
            if 'trade_day_of_week' in filtered_df.columns:
                pnl_dow_fig = plot_pnl_by_category(filtered_df, 'trade_day_of_week', pnl_col=pnl_col, theme=plot_theme, title_prefix="Total PnL by")
                if pnl_dow_fig:
                    st.plotly_chart(pnl_dow_fig, use_container_width=True)
            else:
                logger.debug("PerformancePage: 'trade_day_of_week' column not found for PnL by DOW plot.")


        with col2:
            # Win Rate by Hour
            if 'trade_hour' in filtered_df.columns and 'win' in filtered_df.columns: # 'win' col from data_processing
                winrate_hour_fig = plot_win_rate_analysis(filtered_df, 'trade_hour', win_col='win', theme=plot_theme, title_prefix="Win Rate by")
                if winrate_hour_fig:
                    st.plotly_chart(winrate_hour_fig, use_container_width=True)
            else:
                logger.debug("PerformancePage: 'trade_hour' or 'win' column not found for Win Rate by Hour plot.")


            # PnL by Month
            if 'trade_month' in filtered_df.columns: # Assuming month number (1-12)
                # Optional: Convert month number to month name for better display if desired
                # df_copy_for_month_plot = filtered_df.copy()
                # df_copy_for_month_plot['month_name_display'] = pd.to_datetime(df_copy_for_month_plot['trade_month'], format='%m').dt.strftime('%b')
                # pnl_month_fig = plot_pnl_by_category(df_copy_for_month_plot, 'month_name_display', pnl_col=pnl_col, theme=plot_theme)
                pnl_month_fig = plot_pnl_by_category(filtered_df, 'trade_month', pnl_col=pnl_col, theme=plot_theme, title_prefix="Total PnL by")
                if pnl_month_fig:
                    st.plotly_chart(pnl_month_fig, use_container_width=True)
            else:
                logger.debug("PerformancePage: 'trade_month' column not found for PnL by Month plot.")

    except Exception as e:
        logger.error(f"Error rendering performance breakdown section: {e}", exc_info=True)
        display_custom_message(f"An error occurred in performance breakdown: {e}", "error")

    st.markdown("---")

    # --- Rolling Performance ---
    st.subheader("Rolling Performance Metrics")
    try:
        if len(filtered_df) >= 30: # Example window for rolling metrics
            # Rolling PnL Sum
            rolling_pnl_sum = filtered_df[pnl_col].rolling(window=30, min_periods=10).sum()
            rolling_pnl_fig = plot_rolling_performance(
                filtered_df,
                date_col=EXPECTED_COLUMNS.get('date', 'date'),
                metric_series=rolling_pnl_sum,
                metric_name="30-Period Rolling PnL Sum",
                theme=plot_theme
            )
            if rolling_pnl_fig:
                st.plotly_chart(rolling_pnl_fig, use_container_width=True)

            # Rolling Win Rate (Example)
            if 'win' in filtered_df.columns:
                rolling_win_rate = filtered_df['win'].rolling(window=50, min_periods=20).mean() * 100 # As percentage
                rolling_wr_fig = plot_rolling_performance(
                    filtered_df,
                    date_col=EXPECTED_COLUMNS.get('date', 'date'),
                    metric_series=rolling_win_rate,
                    metric_name="50-Period Rolling Win Rate (%)",
                    theme=plot_theme
                )
                if rolling_wr_fig:
                    st.plotly_chart(rolling_wr_fig, use_container_width=True)
        else:
            display_custom_message("Not enough data for rolling performance metrics (need at least 30 trades).", "info")
    except Exception as e:
        logger.error(f"Error rendering rolling performance metrics: {e}", exc_info=True)
        display_custom_message(f"An error occurred displaying rolling performance: {e}", "error")

    st.markdown("---")

    # --- P&L Calendar View ---
    # This section now uses the PnLCalendarComponent
    date_col_cal = EXPECTED_COLUMNS.get('date', 'date') # Ensure this is the original date column
    if date_col_cal in filtered_df.columns and pnl_col in filtered_df.columns:
        try:
            # Aggregate data to daily PnL for the calendar component
            daily_pnl_df_agg = filtered_df.groupby(
                filtered_df[date_col_cal].dt.normalize() # Group by date part only
            )[pnl_col].sum().reset_index()
            # Rename columns to what PnLCalendarComponent expects ('date', 'pnl')
            daily_pnl_df_agg = daily_pnl_df_agg.rename(columns={date_col_cal: 'date', pnl_col: 'pnl'})

            available_years = sorted(daily_pnl_df_agg['date'].dt.year.unique(), reverse=True)
            if available_years:
                selected_year = st.selectbox(
                    "Select Year for P&L Calendar:",
                    options=available_years,
                    index=0, # Default to the latest year
                    key="perf_page_calendar_year_select"
                )
                if selected_year:
                    calendar_component = PnLCalendarComponent(
                        daily_pnl_df=daily_pnl_df_agg,
                        year=selected_year,
                        plot_theme=plot_theme
                    )
                    calendar_component.render() # This will render the subheader and the plot
            else:
                display_custom_message("No yearly data available to display P&L calendar.", "info")
        except Exception as e:
            logger.error(f"Error preparing or rendering P&L Calendar: {e}", exc_info=True)
            display_custom_message(f"Could not generate P&L Calendar: {e}", "error")
    else:
        display_custom_message(f"Required columns ('{date_col_cal}', '{pnl_col}') not found for P&L Calendar.", "warning")


# --- Main execution for the page ---
if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_performance_page()
