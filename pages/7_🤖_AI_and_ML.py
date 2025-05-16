"""
pages/7_🤖_AI_and_ML.py
AI and ML insights. UI/UX enhanced.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, FORECAST_HORIZON, COLORS
    from utils.common_utils import display_custom_message
    from services.analysis_service import AnalysisService
    from plotting import _apply_custom_theme, plot_value_over_time # Added plot_value_over_time for forecast
    from ai_models import PROPHET_AVAILABLE, PMDARIMA_AVAILABLE, LIFELINES_AVAILABLE
except ImportError as e:
    st.error(f"AI & ML Page Error: Critical module import failed: {e}.")
    APP_TITLE = "TradingDashboard_Error"; logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in 7_🤖_AI_and_ML.py: {e}", exc_info=True)
    PROPHET_AVAILABLE, PMDARIMA_AVAILABLE, LIFELINES_AVAILABLE = False, False, False; COLORS = {}
    def display_custom_message(msg, type="error"): st.error(msg)
    class AnalysisService:
        def get_arima_forecast(self, *args, **kwargs): return {"error": "Service not loaded"}
        def get_prophet_forecast(self, *args, **kwargs): return {"error": "Service not loaded"}
    def _apply_custom_theme(fig, theme): return fig
    def plot_value_over_time(*args, **kwargs): return None
    st.stop()

logger = logging.getLogger(APP_TITLE)
analysis_service = AnalysisService()

def show_ai_ml_page():
    st.title("🤖 AI & Machine Learning Insights")
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Upload data to access AI/ML tools.", "info"); return
    
    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    date_col = EXPECTED_COLUMNS.get('date')

    if filtered_df.empty:
        display_custom_message("No data matches filters.", "info"); return

    # --- Time Series Forecasting ---
    st.subheader("Time Series Forecasting")
    with st.expander("Configure & Run Forecast", expanded=True):
        forecast_series_options = {}
        if pnl_col and date_col and pnl_col in filtered_df.columns and date_col in filtered_df.columns:
            # Daily PnL: Already grouped by day, ensure index is DatetimeIndex
            daily_pnl_series = filtered_df.groupby(filtered_df[date_col].dt.normalize())[pnl_col].sum().dropna()
            if not daily_pnl_series.empty:
                if not isinstance(daily_pnl_series.index, pd.DatetimeIndex):
                    daily_pnl_series.index = pd.to_datetime(daily_pnl_series.index)
                daily_pnl_series = daily_pnl_series.asfreq('D') # Ensure daily frequency
                forecast_series_options["Daily PnL"] = daily_pnl_series

        if 'cumulative_pnl' in filtered_df.columns and date_col and date_col in filtered_df.columns:
            # Equity Curve: May be irregular, resample to daily
            equity_curve_series_raw = filtered_df.set_index(date_col)['cumulative_pnl'].dropna()
            if not equity_curve_series_raw.empty:
                if not equity_curve_series_raw.index.is_monotonic_increasing:
                     equity_curve_series_raw = equity_curve_series_raw.sort_index()
                # Resample to daily, taking the last observation of the day
                equity_curve_daily = equity_curve_series_raw.resample('D').last().ffill() # Forward fill to handle weekends/holidays for Prophet
                if not equity_curve_daily.empty:
                    forecast_series_options["Equity Curve (Daily Resampled)"] = equity_curve_daily
        
        if not forecast_series_options:
            st.warning("No suitable time series data (Daily PnL or Equity Curve) found for forecasting after processing.")
        else:
            with st.form("forecasting_form_aiml_v4"): # Incremented key for new form elements
                sel_fc_series_name = st.selectbox(
                    "Series to forecast:",
                    list(forecast_series_options.keys()),
                    key="fc_series_v4"
                )
                
                model_opts = []
                # The ARIMA (Auto) option should only appear if pmdarima is available.
                # The manual option should always be available.
                if PMDARIMA_AVAILABLE:
                    model_opts.append("ARIMA (Auto - pmdarima)")
                model_opts.append("ARIMA (Manual Order)") # Standard statsmodels ARIMA
                
                if PROPHET_AVAILABLE:
                    model_opts.append("Prophet")
                
                if not model_opts:
                    st.error("No forecasting models available (ARIMA from statsmodels should always be an option if library is present). Check installations.")
                    st.stop() # Stop if no models can be listed
                
                sel_fc_model = st.selectbox("Forecast Model:", model_opts, key="fc_model_v4")
                n_fc_periods = st.number_input(
                    "Periods to Forecast:", 
                    min_value=1, max_value=365, 
                    value=min(FORECAST_HORIZON, 90), 
                    key="fc_periods_v4",
                    help="Number of future periods (typically days) to forecast."
                )
                
                arima_order_manual = None
                arima_seasonal_order_manual = None

                if "ARIMA (Manual Order)" in sel_fc_model:
                    st.write("Specify ARIMA(p,d,q) order:")
                    c1, c2, c3 = st.columns(3)
                    p_manual = c1.number_input("p (AR order)", 0, 5, 1, key="arima_p_v4")
                    d_manual = c2.number_input("d (Differencing)", 0, 2, 1, key="arima_d_v4")
                    q_manual = c3.number_input("q (MA order)", 0, 5, 1, key="arima_q_v4")
                    arima_order_manual = (p_manual, d_manual, q_manual)

                    use_seasonal_arima = st.checkbox("Use Seasonal ARIMA (SARIMA)?", key="sarima_check_v4")
                    if use_seasonal_arima:
                        st.write("Specify SARIMA(P,D,Q,s) seasonal order:")
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        P_manual = sc1.number_input("P (Seasonal AR)", 0, 2, 1, key="sarima_P_v4")
                        D_manual = sc2.number_input("D (Seasonal Diff)", 0, 1, 0, key="sarima_D_v4")
                        Q_manual = sc3.number_input("Q (Seasonal MA)", 0, 2, 1, key="sarima_Q_v4")
                        s_manual = sc4.number_input("s (Seasonality Period)", 1, 365, 7, key="sarima_s_v4", help="e.g., 7 for daily data with weekly seasonality, 12 for monthly with annual.")
                        arima_seasonal_order_manual = (P_manual, D_manual, Q_manual, s_manual)
                
                submit_fc_btn = st.form_submit_button(f"Generate {sel_fc_model} Forecast")

    if 'submit_fc_btn' in locals() and submit_fc_btn and forecast_series_options and sel_fc_series_name:
        ts_to_forecast = forecast_series_options[sel_fc_series_name]
        
        if len(ts_to_forecast.dropna()) < 20: # Check after potential resampling
            display_custom_message("Need at least 20 data points in the selected series for forecasting.", "warning")
        else:
            with st.spinner(f"Training {sel_fc_model} and forecasting {n_fc_periods} periods..."):
                forecast_output = None
                if "ARIMA" in sel_fc_model:
                    # If "ARIMA (Auto - pmdarima)" is selected, arima_order_manual will be None, triggering auto_arima in service.
                    # If "ARIMA (Manual Order)" is selected, arima_order_manual will have the tuple.
                    forecast_output = analysis_service.get_arima_forecast(
                        ts_to_forecast, 
                        order=arima_order_manual, # This will be None for Auto, or a tuple for Manual
                        seasonal_order=arima_seasonal_order_manual, # None if not SARIMA
                        n_periods=n_fc_periods
                    )
                elif sel_fc_model == "Prophet":
                    # Prepare DataFrame for Prophet: 'ds' and 'y' columns
                    # The series from forecast_series_options should already have a DatetimeIndex
                    prophet_df_in = ts_to_forecast.reset_index()
                    prophet_df_in.columns = ['ds', 'y'] # Rename columns
                    
                    # Ensure 'ds' is datetime (should be, but double-check)
                    prophet_df_in['ds'] = pd.to_datetime(prophet_df_in['ds'])
                    
                    logger.debug(f"Prophet input df head for '{sel_fc_series_name}':\n{prophet_df_in.head()}")
                    logger.debug(f"Prophet input df 'ds' inferred freq: {pd.infer_freq(prophet_df_in['ds'])}")

                    forecast_output = analysis_service.get_prophet_forecast(
                        prophet_df_in, 
                        n_periods=n_fc_periods
                    )
            
            if forecast_output and 'error' not in forecast_output:
                st.success(f"{sel_fc_model} forecast generated successfully!")
                
                # Plotting the forecast
                original_series_name = sel_fc_series_name
                forecast_values = None
                conf_int_lower = None
                conf_int_upper = None

                if "ARIMA" in sel_fc_model:
                    forecast_values = forecast_output.get('forecast')
                    conf_int_lower = forecast_output.get('conf_int_lower')
                    conf_int_upper = forecast_output.get('conf_int_upper')
                elif sel_fc_model == "Prophet":
                    forecast_df = forecast_output.get('forecast_df')
                    if forecast_df is not None and not forecast_df.empty:
                        # Prophet forecast_df contains history + future.
                        # We need to align its 'ds' with the original series for plotting continuity if possible,
                        # or just plot the forecast part.
                        # For simplicity, we'll plot the 'yhat' for the forecast period.
                        # The 'ds' column in forecast_df is the date.
                        
                        # Get the forecast part
                        forecast_segment = forecast_df[forecast_df['ds'] > ts_to_forecast.index.max()]
                        
                        forecast_values = forecast_segment.set_index('ds')['yhat']
                        conf_int_lower = forecast_segment.set_index('ds')['yhat_lower']
                        conf_int_upper = forecast_segment.set_index('ds')['yhat_upper']
                        original_series_name = f"{sel_fc_series_name} (Prophet Fit)" # Prophet also shows historical fit

                if forecast_values is not None and not forecast_values.empty:
                    fig = go.Figure()
                    # Plot original series
                    fig.add_trace(go.Scatter(
                        x=ts_to_forecast.index, y=ts_to_forecast,
                        mode='lines', name=original_series_name,
                        line=dict(color=COLORS.get("royal_blue"))
                    ))
                    # Plot forecast values
                    fig.add_trace(go.Scatter(
                        x=forecast_values.index, y=forecast_values,
                        mode='lines', name=f'{sel_fc_model} Forecast',
                        line=dict(color=COLORS.get("orange"))
                    ))
                    # Plot confidence intervals if available
                    if conf_int_lower is not None and conf_int_upper is not None:
                        fig.add_trace(go.Scatter(
                            x=conf_int_lower.index, y=conf_int_lower,
                            mode='lines', line=dict(width=0), showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=conf_int_upper.index, y=conf_int_upper,
                            mode='lines', line=dict(width=0), fill='tonexty',
                            fillcolor='rgba(255,165,0,0.2)', name='Confidence Interval'
                        ))
                    
                    fig.update_layout(
                        title=f"{sel_fc_model} Forecast for {sel_fc_series_name}",
                        xaxis_title="Date", yaxis_title="Value",
                        hovermode="x unified"
                    )
                    st.plotly_chart(_apply_custom_theme(fig, plot_theme), use_container_width=True)
                    
                    if "ARIMA" in sel_fc_model and forecast_output.get('model_summary'):
                        with st.expander("ARIMA Model Summary"):
                            st.text(forecast_output.get('model_summary'))
                else:
                    display_custom_message("Forecast generated, but no forecast values to plot.", "warning")

            elif forecast_output: # Error occurred
                display_custom_message(f"Forecast Error ({sel_fc_model}): {forecast_output.get('error', 'Unknown error')}", "error")
            else: # Should not happen if service always returns dict
                display_custom_message(f"Forecast analysis for {sel_fc_model} failed to return results.", "error")

    # --- Anomaly Detection & Survival Analysis ---
    # (Keep existing logic for these sections)

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_ai_ml_page()
