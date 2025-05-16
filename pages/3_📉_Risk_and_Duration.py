"""
pages/3_📉_Risk_and_Duration.py

This page focuses on risk metrics, correlation analysis, and trade duration analysis.
KPIs are now grouped for better readability.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

try:
    # Added KPI_GROUPS_RISK_DURATION
    from config import APP_TITLE, EXPECTED_COLUMNS, COLORS, KPI_CONFIG, KPI_GROUPS_RISK_DURATION, AVAILABLE_BENCHMARKS
    from utils.common_utils import display_custom_message # format_currency, format_percentage removed as not directly used
    from plotting import plot_correlation_matrix, _apply_custom_theme
    from services.analysis_service import AnalysisService
    from ai_models import LIFELINES_AVAILABLE
    from components.kpi_display import KPIClusterDisplay
except ImportError as e:
    st.error(f"Risk & Duration Page Error: Critical module import failed: {e}.")
    APP_TITLE = "TradingDashboard_Error"; logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in 3_📉_Risk_and_Duration.py: {e}", exc_info=True)
    LIFELINES_AVAILABLE = False; COLORS = {}; KPI_CONFIG = {}; KPI_GROUPS_RISK_DURATION = {}; AVAILABLE_BENCHMARKS = {}
    def display_custom_message(msg, type="error"): st.error(msg)
    class KPIClusterDisplay:
        def __init__(self, **kwargs): pass
        def render(self): st.warning("KPI Display Component failed to load.")
    def plot_correlation_matrix(**kwargs): return None
    def _apply_custom_theme(fig, theme): return fig
    st.stop()

logger = logging.getLogger(APP_TITLE)
analysis_service = AnalysisService() # Instance for service calls

def show_risk_duration_page():
    st.title("📉 Risk & Duration Analysis")
    logger.info("Rendering Risk & Duration Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Upload and process data to view this page.", "info"); return
    if 'kpi_results' not in st.session_state or st.session_state.kpi_results is None:
        display_custom_message("KPI results are not available. Ensure data is processed.", "warning"); return
    if 'error' in st.session_state.kpi_results: # Check if kpi_results itself is an error dict
        display_custom_message(f"Error in KPI calculation: {st.session_state.kpi_results['error']}", "error"); return

    filtered_df = st.session_state.filtered_data
    kpi_results = st.session_state.kpi_results
    kpi_confidence_intervals = st.session_state.get('kpi_confidence_intervals', {})
    plot_theme = st.session_state.get('current_theme', 'dark')
    benchmark_daily_returns = st.session_state.get('benchmark_daily_returns') # For benchmark-related risk KPIs

    if filtered_df.empty:
        display_custom_message("No data matches filters for risk and duration analysis.", "info"); return

    # --- Key Risk Metrics Section ---
    st.header("Key Risk Metrics") # Changed to header
    
    cols_per_row_setting = 3 # Adjust as needed for risk page

    for group_name, kpi_keys_in_group in KPI_GROUPS_RISK_DURATION.items():
        group_kpi_results = {key: kpi_results[key] for key in kpi_keys_in_group if key in kpi_results}

        # Special handling for Benchmark-related risk group
        if group_name == "Market Risk & Relative Performance":
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
                kpi_cluster_risk = KPIClusterDisplay(
                    kpi_results=group_kpi_results,
                    kpi_definitions=KPI_CONFIG,
                    kpi_order=kpi_keys_in_group,
                    kpi_confidence_intervals=kpi_confidence_intervals,
                    cols_per_row=cols_per_row_setting
                )
                kpi_cluster_risk.render()
                st.markdown("---") # Separator
            except Exception as e:
                logger.error(f"Error rendering Key Risk Metrics for group '{group_name}': {e}", exc_info=True)
                display_custom_message(f"An error occurred while displaying Key Risk Metrics for {group_name}: {e}", "error")

    # --- Feature Correlation Matrix Section ---
    # (This section remains largely the same as your latest version)
    st.header("Advanced Risk Visualizations") # Changed to header
    st.subheader("Feature Correlation Matrix")
    try:
        pnl_col_name = EXPECTED_COLUMNS.get('pnl')
        numeric_cols_for_corr = []
        if pnl_col_name and pnl_col_name in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[pnl_col_name]):
            numeric_cols_for_corr.append(pnl_col_name)
        duration_numeric_col = 'duration_minutes_numeric'
        if duration_numeric_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[duration_numeric_col]):
            numeric_cols_for_corr.append(duration_numeric_col)
        risk_numeric_col = 'risk_numeric_internal'
        if risk_numeric_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[risk_numeric_col]):
            numeric_cols_for_corr.append(risk_numeric_col)
        rrr_col = 'reward_risk_ratio'
        if rrr_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[rrr_col]):
            numeric_cols_for_corr.append(rrr_col)
        signal_conf_col = EXPECTED_COLUMNS.get('signal_confidence')
        if signal_conf_col and signal_conf_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[signal_conf_col]):
            numeric_cols_for_corr.append(signal_conf_col)

        if len(numeric_cols_for_corr) >= 2:
            correlation_fig = plot_correlation_matrix(
                filtered_df, numeric_cols=list(set(numeric_cols_for_corr)), theme=plot_theme
            )
            if correlation_fig:
                st.plotly_chart(correlation_fig, use_container_width=True)
            else:
                display_custom_message("Could not generate the correlation matrix.", "warning")
        else:
            display_custom_message(f"Not enough numeric features (need at least 2, found {len(numeric_cols_for_corr)}) for correlation matrix.", "info")
    except Exception as e:
        logger.error(f"Error rendering Feature Correlation Matrix: {e}", exc_info=True)
        display_custom_message(f"An error displaying Feature Correlation Matrix: {e}", "error")

    st.markdown("---")
    # --- Trade Duration Analysis (Survival Curve) ---
    # (This section remains largely the same as your latest version)
    st.subheader("Trade Duration Analysis (Survival Curve)")
    if not LIFELINES_AVAILABLE:
        display_custom_message("Survival analysis tools (Lifelines library) are not available.", "warning")
    else:
        duration_col_for_analysis = 'duration_minutes_numeric'
        if duration_col_for_analysis in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[duration_col_for_analysis]):
            durations = filtered_df[duration_col_for_analysis].dropna()
            if not durations.empty and len(durations) >= 5:
                event_observed = pd.Series([True] * len(durations), index=durations.index)
                with st.spinner("Performing Kaplan-Meier survival analysis..."):
                    km_service_results = analysis_service_instance.perform_kaplan_meier_analysis(durations, event_observed) # Use instance

                if km_service_results and 'error' not in km_service_results and 'survival_function_df' in km_service_results:
                    survival_df = km_service_results['survival_function_df']
                    km_plot_fig = go.Figure()
                    km_plot_fig.add_trace(go.Scatter(
                        x=survival_df.index, y=survival_df['KM_estimate'],
                        mode='lines', name='Survival Probability (KM Estimate)', line_shape='hv',
                        line=dict(color=COLORS.get('royal_blue', 'blue'))
                    ))
                    if 'confidence_interval_df' in km_service_results and not km_service_results['confidence_interval_df'].empty:
                        ci_df = km_service_results['confidence_interval_df']
                        conf_level = km_service_results.get("confidence_level", 0.95)
                        lower_ci_col = f'KM_estimate_lower_{conf_level:.2f}'.replace('0.', '') if conf_level != 0.95 else 'KM_estimate_lower_0.95'
                        upper_ci_col = f'KM_estimate_upper_{conf_level:.2f}'.replace('0.', '') if conf_level != 0.95 else 'KM_estimate_upper_0.95'
                        # Lifelines might use slightly different naming, e.g. KM_estimate_lower_0.95
                        # Check common variations
                        if lower_ci_col not in ci_df.columns and 'KM_estimate_lower_0.95' in ci_df.columns: lower_ci_col = 'KM_estimate_lower_0.95'
                        if upper_ci_col not in ci_df.columns and 'KM_estimate_upper_0.95' in ci_df.columns: upper_ci_col = 'KM_estimate_upper_0.95'
                        
                        if lower_ci_col in ci_df.columns and upper_ci_col in ci_df.columns:
                            km_plot_fig.add_trace(go.Scatter(
                                x=ci_df.index, y=ci_df[lower_ci_col], mode='lines',
                                line=dict(width=0), showlegend=False, line_shape='hv'
                            ))
                            km_plot_fig.add_trace(go.Scatter(
                                x=ci_df.index, y=ci_df[upper_ci_col], mode='lines',
                                line=dict(width=0), fill='tonexty', fillcolor='rgba(65,105,225,0.2)',
                                name=f'{int(conf_level*100)}% Confidence Interval', 
                                showlegend=True, line_shape='hv'
                            ))
                    
                    duration_display_name = EXPECTED_COLUMNS.get('duration_minutes', 'duration_minutes').replace('_', ' ').title()
                    km_plot_fig.update_layout(
                        title_text=f"Trade Survival Curve for {duration_display_name}",
                        xaxis_title=f"Duration ({duration_display_name})",
                        yaxis_title="Probability of Trade Still Being Open", yaxis_range=[0, 1.05]
                    )
                    st.plotly_chart(_apply_custom_theme(km_plot_fig, plot_theme), use_container_width=True)
                    median_survival = km_service_results.get('median_survival_time')
                    st.metric(
                        label=f"Median Trade Duration ({duration_display_name})",
                        value=f"{median_survival:.2f} mins" if pd.notna(median_survival) else "N/A",
                        help="The time at which 50% of trades are expected to have closed."
                    )
                elif km_service_results and 'error' in km_service_results:
                    display_custom_message(f"Kaplan-Meier Analysis Error: {km_service_results['error']}", "error")
                else:
                    display_custom_message("Survival analysis for trade duration did not return expected results.", "warning")
            else:
                display_custom_message(f"Not enough valid data in '{duration_col_for_analysis}' for survival analysis (need at least 5 observations). Current valid count: {len(durations)}", "info")
        else:
            duration_config_key = 'duration_minutes'; expected_duration_col_name = EXPECTED_COLUMNS.get(duration_config_key)
            available_numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
            error_msg = (f"The standardized numeric duration column ('{duration_col_for_analysis}') was not found or is not numeric. Check CSV and `config.EXPECTED_COLUMNS['{duration_config_key}']` (expected as '{expected_duration_col_name}'). Available numeric columns: {available_numeric_cols}")
            display_custom_message(error_msg, "warning")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_risk_duration_page()
