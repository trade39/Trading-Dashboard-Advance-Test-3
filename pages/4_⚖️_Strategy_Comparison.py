"""
pages/4_⚖️_Strategy_Comparison.py

This page allows users to compare the performance of different trading strategies
side-by-side, based on various KPIs and visualizations.
"""
import streamlit as st
import pandas as pd
import logging
import plotly.graph_objects as go


# --- Assuming root-level modules are accessible ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, DEFAULT_KPI_DISPLAY_ORDER, COLORS
    from utils.common_utils import display_custom_message, format_currency, format_percentage
    from services.analysis_service import AnalysisService
    from plotting import _apply_custom_theme
except ImportError as e:
    st.error(f"Strategy Comparison Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error" # Fallback
    logger = logging.getLogger(APP_TITLE) # Attempt to get a logger
    logger.error(f"CRITICAL IMPORT ERROR in Strategy Comparison Page: {e}", exc_info=True)
    # Define dummy EXPECTED_COLUMNS to prevent further NameErrors if config fails to load
    EXPECTED_COLUMNS = {"strategy": "strategy_fallback"} 
    DEFAULT_KPI_DISPLAY_ORDER = []
    COLORS = {}
    st.stop()

logger = logging.getLogger(APP_TITLE)
analysis_service = AnalysisService() # Instantiate service

def show_strategy_comparison_page():
    """
    Renders the content for the Strategy Comparison page.
    """
    st.title("⚖️ Strategy Performance Comparison")
    logger.info("Rendering Strategy Comparison Page.")

    # --- Check for necessary data in session state ---
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data in the main application to compare strategies.", "info")
        logger.info("StrategyComparisonPage: No filtered_data in session_state.")
        return

    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    risk_free_rate = st.session_state.get('risk_free_rate', 0.02) 
    
    logger.debug(f"StrategyComparisonPage: Columns in st.session_state.filtered_data: {filtered_df.columns.tolist()}")
    strategy_col_from_config = EXPECTED_COLUMNS.get('strategy')
    logger.debug(f"StrategyComparisonPage: Expected strategy column from config (EXPECTED_COLUMNS['strategy']): '{strategy_col_from_config}'")

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Cannot perform strategy comparison.", "info")
        logger.info("StrategyComparisonPage: filtered_df is empty.")
        return

    actual_strategy_col_name = strategy_col_from_config 

    if not actual_strategy_col_name or actual_strategy_col_name not in filtered_df.columns:
        err_msg = (
            f"Strategy column ('{actual_strategy_col_name}') not found in the data. Comparison is not possible. "
            f"Available columns in the filtered data: {filtered_df.columns.tolist()}"
        )
        display_custom_message(err_msg, "warning")
        logger.warning(err_msg)
        return

    # --- Strategy Selection ---
    try:
        available_strategies = sorted(filtered_df[actual_strategy_col_name].astype(str).dropna().unique())
        
        if not available_strategies:
            display_custom_message("No distinct strategies found in the data to compare.", "info")
            logger.info("StrategyComparisonPage: No distinct strategies found.")
            return
        if len(available_strategies) < 2:
            display_custom_message(f"Only one strategy ('{available_strategies[0]}') found. At least two strategies are needed for comparison.", "info")
            logger.info(f"StrategyComparisonPage: Only one strategy found: {available_strategies[0]}")

        st.subheader("Select Strategies to Compare")
        default_selection = available_strategies[:2] if len(available_strategies) >= 2 else available_strategies

        selected_strategies = st.multiselect(
            "Choose strategies:",
            options=available_strategies,
            default=default_selection,
            key="strategy_comp_select_v2"
        )

        if not selected_strategies:
            display_custom_message("Please select at least one strategy to view its performance, or two or more to compare.", "info")
            return
        
        if len(selected_strategies) == 1:
            st.info(f"Displaying performance for strategy: **{selected_strategies[0]}**. Select more strategies for comparison.")

    except Exception as e:
        logger.error(f"Error during strategy selection setup using column '{actual_strategy_col_name}': {e}", exc_info=True)
        display_custom_message(f"An error occurred setting up strategy selection: {e}", "error")
        return

    # --- Comparative KPI Table ---
    st.subheader("Key Performance Indicator Comparison")
    comparison_kpi_data = []
    try:
        for strat_name in selected_strategies:
            strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)]
            if not strat_df.empty:
                kpis = analysis_service.get_core_kpis(strat_df, risk_free_rate)
                if kpis and 'error' not in kpis:
                    comparison_kpi_data.append({"Strategy": strat_name, **kpis})
                else:
                    logger.warning(f"Could not calculate KPIs for strategy '{strat_name}'. Error: {kpis.get('error') if kpis else 'Unknown'}")
            else:
                logger.info(f"No data found for strategy '{strat_name}' within the current filters.")
        
        if comparison_kpi_data:
            comp_df = pd.DataFrame(comparison_kpi_data).set_index("Strategy")
            
            kpis_to_show_in_table = [
                kpi for kpi in DEFAULT_KPI_DISPLAY_ORDER
                if kpi in comp_df.columns and kpi not in ['trading_days', 'risk_free_rate_used']
            ]
            if not kpis_to_show_in_table and comp_df.columns.any(): 
                kpis_to_show_in_table = comp_df.columns.tolist()

            if kpis_to_show_in_table:
                # Define a dark text color for highlighted cells
                dark_text_color_for_highlight = 'color: #FFFFFF;' # A very dark gray

                st.dataframe(
                    comp_df[kpis_to_show_in_table].style.format("{:,.2f}", na_rep="-")
                    .highlight_max(axis=0, color='lightgreen', props=dark_text_color_for_highlight)
                    .highlight_min(axis=0, color='#FFCCCB', props=dark_text_color_for_highlight)
                )
            else:
                display_custom_message("No common KPIs found to display for selected strategies.", "warning")
        elif selected_strategies: 
            display_custom_message(f"No performance data could be calculated for the selected strategies: {', '.join(selected_strategies)}.", "warning")

    except Exception as e:
        logger.error(f"Error generating KPI comparison table: {e}", exc_info=True)
        display_custom_message(f"An error occurred generating the KPI comparison: {e}", "error")

    st.markdown("---")

    # --- Comparative Visualizations (Example: Equity Curves) ---
    st.subheader("Comparative Equity Curves")
    if len(selected_strategies) > 0 :
        equity_comp_fig = go.Figure()
        has_data_for_plot = False
        
        date_col_name = EXPECTED_COLUMNS.get('date')
        pnl_col_name = EXPECTED_COLUMNS.get('pnl')

        if not date_col_name or not pnl_col_name:
            display_custom_message("Date or PnL column configuration is missing. Cannot plot equity curves.", "error")
            logger.error("StrategyComparisonPage: Date or PnL column not found in EXPECTED_COLUMNS for equity plot.")
            # Return or stop further execution in this block if essential columns are missing
            # For example, by not proceeding with the loop below.
        else:
            for i, strat_name in enumerate(selected_strategies):
                strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)]
                
                if not strat_df.empty and date_col_name in strat_df.columns and pnl_col_name in strat_df.columns:
                    strat_df = strat_df.sort_values(by=date_col_name)
                    if pd.api.types.is_numeric_dtype(strat_df[pnl_col_name]):
                        strat_df['cumulative_pnl'] = strat_df[pnl_col_name].cumsum()
                        equity_comp_fig.add_trace(go.Scatter(
                            x=strat_df[date_col_name],
                            y=strat_df['cumulative_pnl'],
                            mode='lines',
                            name=strat_name,
                            # Example of using a color cycle from your COLORS config if you define a list
                            # line=dict(color=COLORS.get(f"plot_color_{i % N_PLOT_COLORS}", COLORS.get("royal_blue")))
                        ))
                        has_data_for_plot = True
                    else:
                        logger.warning(f"StrategyComparisonPage: PnL column '{pnl_col_name}' for strategy '{strat_name}' is not numeric. Skipping for equity curve.")
                else:
                    logger.info(f"StrategyComparisonPage: No data or missing PnL/Date columns for strategy '{strat_name}' for equity plot.")
            
            if has_data_for_plot:
                equity_comp_fig.update_layout(
                    title="Equity Curve Comparison by Strategy",
                    xaxis_title="Date",
                    yaxis_title="Cumulative PnL",
                    hovermode="x unified"
                )
                st.plotly_chart(_apply_custom_theme(equity_comp_fig, plot_theme), use_container_width=True)
            elif selected_strategies:
                display_custom_message("Not enough data to plot comparative equity curves for selected strategies.", "info")

# --- Main execution for the page ---
if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_strategy_comparison_page()
