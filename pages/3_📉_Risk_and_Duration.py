"""
pages/3_📉_Risk_and_Duration.py

This page focuses on risk metrics, correlation analysis, and trade duration analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, COLORS, KPI_CONFIG, DEFAULT_KPI_DISPLAY_ORDER
    from utils.common_utils import display_custom_message, format_currency, format_percentage
    from plotting import plot_correlation_matrix, _apply_custom_theme
    from services.analysis_service import AnalysisService
    from ai_models import LIFELINES_AVAILABLE
    from components.kpi_display import KPIClusterDisplay # For displaying risk KPIs
except ImportError as e:
    st.error(f"Risk & Duration Page Error: Critical module import failed: {e}.")
    APP_TITLE = "TradingDashboard_Error"; logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in 3_📉_Risk_and_Duration.py: {e}", exc_info=True)
    # Define fallbacks for critical missing imports to allow the page to at least load an error message
    LIFELINES_AVAILABLE = False; COLORS = {}; KPI_CONFIG = {}; DEFAULT_KPI_DISPLAY_ORDER = []
    # Dummy display_custom_message if common_utils fails
    def display_custom_message(msg, type="error"): st.error(msg)
    # Dummy KPIClusterDisplay
    class KPIClusterDisplay:
        def __init__(self, **kwargs): pass
        def render(self): st.warning("KPI Display Component failed to load.")
    # Dummy plot_correlation_matrix
    def plot_correlation_matrix(**kwargs): return None
    def _apply_custom_theme(fig, theme): return fig

    st.stop()


logger = logging.getLogger(APP_TITLE)
analysis_service = AnalysisService()

def show_risk_duration_page():
    """
    Renders the content for the Risk & Duration Analysis page.
    """
    st.title("📉 Risk & Duration Analysis")
    logger.info("Rendering Risk & Duration Page.")

    # --- Check for necessary data in session state ---
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data to view risk and duration analysis.", "info")
        logger.info("RiskDurationPage: No filtered_data in session_state.")
        return

    if 'kpi_results' not in st.session_state or st.session_state.kpi_results is None:
        display_custom_message("KPI results are not available. Please ensure data is processed before viewing this page.", "warning")
        logger.info("RiskDurationPage: No kpi_results in session_state.")
        return

    if 'error' in st.session_state.kpi_results:
        display_custom_message(f"Error in KPI calculation: {st.session_state.kpi_results['error']}", "error")
        logger.warning("RiskDurationPage: Error found in kpi_results.")
        return

    filtered_df = st.session_state.filtered_data
    kpi_results = st.session_state.kpi_results
    kpi_confidence_intervals = st.session_state.get('kpi_confidence_intervals', {})
    plot_theme = st.session_state.get('current_theme', 'dark')

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Cannot perform risk and duration analysis.", "info")
        logger.info("RiskDurationPage: filtered_df is empty.")
        return

    # --- Key Risk Metrics Section ---
    st.subheader("Key Risk Metrics")
    try:
        # Define which KPIs are considered "risk metrics"
        risk_kpi_keys = [
            "max_drawdown_abs", "max_drawdown_pct",
            "var_95_loss", "cvar_95_loss",
            "var_99_loss", "cvar_99_loss",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", # Risk-adjusted returns
            "pnl_skewness", "pnl_kurtosis" # Distributional risk indicators
        ]
        
        # Filter the main kpi_results and kpi_order for only risk KPIs
        risk_kpis_to_display = {key: kpi_results[key] for key in risk_kpi_keys if key in kpi_results}
        
        # Maintain a sensible order for risk KPIs
        ordered_risk_kpis = [key for key in DEFAULT_KPI_DISPLAY_ORDER if key in risk_kpis_to_display]
        # Add any risk KPIs that might not be in DEFAULT_KPI_DISPLAY_ORDER but are in risk_kpi_keys
        for key in risk_kpi_keys:
            if key in risk_kpis_to_display and key not in ordered_risk_kpis:
                ordered_risk_kpis.append(key)

        if risk_kpis_to_display:
            kpi_cluster_risk = KPIClusterDisplay(
                kpi_results=risk_kpis_to_display,
                kpi_definitions=KPI_CONFIG,
                kpi_order=ordered_risk_kpis,
                kpi_confidence_intervals=kpi_confidence_intervals, # Pass all CIs, component will pick relevant ones
                cols_per_row=3 # Adjust as needed for risk metrics
            )
            kpi_cluster_risk.render()
        else:
            display_custom_message("No specific risk metrics could be calculated or displayed.", "info")
            logger.info("RiskDurationPage: No risk_kpis_to_display.")

    except Exception as e:
        logger.error(f"Error rendering Key Risk Metrics: {e}", exc_info=True)
        display_custom_message(f"An error occurred while displaying Key Risk Metrics: {e}", "error")

    st.markdown("---")

    # --- Feature Correlation Matrix Section ---
    st.subheader("Feature Correlation Matrix")
    try:
        # Select numeric columns for correlation.
        # These could be predefined or dynamically selected.
        # Example: PnL, duration, risk (if numeric), entry/exit prices (if meaningful)
        # Ensure these columns exist and are numeric in filtered_df
        
        # Start with PnL
        pnl_col_name = EXPECTED_COLUMNS.get('pnl')
        numeric_cols_for_corr = []
        if pnl_col_name and pnl_col_name in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[pnl_col_name]):
            numeric_cols_for_corr.append(pnl_col_name)

        # Add 'duration_minutes_numeric' if it exists and is numeric
        duration_numeric_col = 'duration_minutes_numeric' # Standardized in data_processing.py
        if duration_numeric_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[duration_numeric_col]):
            numeric_cols_for_corr.append(duration_numeric_col)
        
        # Add 'risk_numeric_internal' if it exists and is numeric
        risk_numeric_col = 'risk_numeric_internal' # Standardized in data_processing.py
        if risk_numeric_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[risk_numeric_col]):
            numeric_cols_for_corr.append(risk_numeric_col)

        # Add 'reward_risk_ratio' if it exists and is numeric
        rrr_col = 'reward_risk_ratio' # Engineered in data_processing.py
        if rrr_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[rrr_col]):
            numeric_cols_for_corr.append(rrr_col)

        # Add other potentially interesting numeric columns if they exist
        # For example, if 'signal_confidence' is numeric and configured:
        signal_conf_col = EXPECTED_COLUMNS.get('signal_confidence')
        if signal_conf_col and signal_conf_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[signal_conf_col]):
            numeric_cols_for_corr.append(signal_conf_col)

        if len(numeric_cols_for_corr) >= 2:
            correlation_fig = plot_correlation_matrix(
                filtered_df,
                numeric_cols=list(set(numeric_cols_for_corr)), # Ensure unique columns
                theme=plot_theme
            )
            if correlation_fig:
                st.plotly_chart(correlation_fig, use_container_width=True)
            else:
                display_custom_message("Could not generate the correlation matrix. Ensure there are at least two numeric columns available for comparison.", "warning")
        else:
            display_custom_message(f"Not enough numeric features (need at least 2, found {len(numeric_cols_for_corr)}) to display a correlation matrix. Found: {numeric_cols_for_corr}", "info")
            logger.info(f"RiskDurationPage: Not enough numeric columns for correlation matrix. Available for correlation: {numeric_cols_for_corr}")

    except Exception as e:
        logger.error(f"Error rendering Feature Correlation Matrix: {e}", exc_info=True)
        display_custom_message(f"An error occurred while displaying the Feature Correlation Matrix: {e}", "error")

    st.markdown("---")

    # --- Trade Duration Analysis (Survival Curve) ---
    st.subheader("Trade Duration Analysis (Survival Curve)")
    if not LIFELINES_AVAILABLE:
        display_custom_message("Survival analysis tools (Lifelines library) are not available. Please ensure it's installed (`pip install lifelines`).", "warning")
    else:
        duration_col_for_analysis = 'duration_minutes_numeric' # Standardized numeric duration column

        if duration_col_for_analysis in filtered_df.columns and \
           pd.api.types.is_numeric_dtype(filtered_df[duration_col_for_analysis]):
            
            durations = filtered_df[duration_col_for_analysis].dropna()
            
            if not durations.empty and len(durations) >= 5: # Need a few data points for meaningful analysis
                # For survival analysis, 'event_observed' is typically True if the event (e.g., trade closing) occurred.
                # In this context, all trades in the journal have closed, so event_observed is always True.
                event_observed = pd.Series([True] * len(durations), index=durations.index)

                with st.spinner("Performing Kaplan-Meier survival analysis for trade duration..."):
                    km_service_results = analysis_service.perform_kaplan_meier_analysis(durations, event_observed)

                if km_service_results and 'error' not in km_service_results and 'survival_function_df' in km_service_results:
                    survival_df = km_service_results['survival_function_df']
                    km_plot_fig = go.Figure()
                    
                    # Plot Kaplan-Meier estimate
                    km_plot_fig.add_trace(go.Scatter(
                        x=survival_df.index, y=survival_df['KM_estimate'],
                        mode='lines', name='Survival Probability (KM Estimate)', line_shape='hv', # hv for step plot
                        line=dict(color=COLORS.get('royal_blue', 'blue'))
                    ))
                    
                    # Plot confidence interval if available
                    if 'confidence_interval_df' in km_service_results and not km_service_results['confidence_interval_df'].empty:
                        ci_df = km_service_results['confidence_interval_df']
                        # Ensure column names match what lifelines provides (e.g., 'KM_estimate_lower_0.95')
                        lower_ci_col = f'KM_estimate_lower_{km_service_results.get("confidence_level", 0.95)}'
                        upper_ci_col = f'KM_estimate_upper_{km_service_results.get("confidence_level", 0.95)}'

                        if lower_ci_col in ci_df.columns and upper_ci_col in ci_df.columns:
                            km_plot_fig.add_trace(go.Scatter(
                                x=ci_df.index, y=ci_df[lower_ci_col], mode='lines',
                                line=dict(width=0), showlegend=False, line_shape='hv'
                            ))
                            km_plot_fig.add_trace(go.Scatter(
                                x=ci_df.index, y=ci_df[upper_ci_col], mode='lines',
                                line=dict(width=0), fill='tonexty', fillcolor='rgba(65,105,225,0.2)', # Light blue fill
                                name=f'{int(km_service_results.get("confidence_level", 0.95)*100)}% Confidence Interval', 
                                showlegend=True, line_shape='hv'
                            ))
                        else:
                            logger.warning(f"RiskDurationPage: Confidence interval columns not found in survival analysis results: {lower_ci_col}, {upper_ci_col}")
                    
                    duration_display_name = EXPECTED_COLUMNS.get('duration_minutes', 'duration_minutes').replace('_', ' ').title()
                    km_plot_fig.update_layout(
                        title_text=f"Trade Survival Curve for {duration_display_name}",
                        xaxis_title=f"Duration ({duration_display_name})",
                        yaxis_title="Probability of Trade Still Being Open", # More intuitive y-axis label
                        yaxis_range=[0, 1.05] # Ensure 0 to 1 is visible
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
            # Provide more specific feedback if the duration column is missing
            duration_config_key = 'duration_minutes'
            expected_duration_col_name = EXPECTED_COLUMNS.get(duration_config_key)
            available_numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
            
            error_msg = (
                f"The standardized numeric duration column ('{duration_col_for_analysis}') was not found or is not numeric. "
                f"This column is typically engineered in `data_processing.py` based on the CSV column mapped by `EXPECTED_COLUMNS['{duration_config_key}']` "
                f"(currently configured as '{expected_duration_col_name}').\n\n"
                f"Please check:\n"
                f"1. Your uploaded CSV contains a column for trade duration that can be converted to minutes.\n"
                f"2. `config.py` correctly maps `EXPECTED_COLUMNS['{duration_config_key}']` to your CSV's duration column name (after cleaning by `data_processing.py`).\n"
                f"3. The duration data is numeric or can be converted to numeric minutes.\n\n"
                f"Available numeric columns in the processed data: {available_numeric_cols}"
            )
            display_custom_message(error_msg, "warning")
            logger.warning(f"RiskDurationPage: Duration column for survival analysis ('{duration_col_for_analysis}') not found or not numeric. Configured as '{expected_duration_col_name}'. Available df columns: {filtered_df.columns.tolist()}")

# --- Main execution for the page ---
if __name__ == "__main__":
    # This block is for individual page testing if needed.
    # Streamlit typically runs the selected page script from top to bottom.
    if 'app_initialized' not in st.session_state: # Basic check
        st.warning("This page is part of a multi-page app. Please run the main app.py script for full functionality.")
        # Initialize minimal state for testing if necessary
        # st.session_state.filtered_data = pd.DataFrame(...) # Mock data
        # st.session_state.kpi_results = {...}
        # st.session_state.current_theme = 'dark'
        # st.session_state.app_initialized = True
    
    show_risk_duration_page()
