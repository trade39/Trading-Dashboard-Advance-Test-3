"""
ai_models.py

Implements or wraps various machine learning models for trading analytics,
including time series forecasting (ARIMA, Prophet, LSTMs),
causal inference, survival analysis, meta-labeling, and advanced anomaly detection.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

# Time Series Forecasting
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning as StatsmodelsValueWarning

# Logging setup (get the main app logger or a local one if run standalone)
try:
    from config import APP_TITLE, FORECAST_HORIZON, CONFIDENCE_LEVEL
except ImportError:
    print("Warning (ai_models.py): Could not import from config. Using fallback values.")
    APP_TITLE = "TradingDashboard_Default_AI"
    FORECAST_HORIZON = 30
    CONFIDENCE_LEVEL = 0.95

import logging
logger = logging.getLogger(APP_TITLE) # Main app logger

# pmdarima: Conditional import
PMDARIMA_AVAILABLE = False
auto_arima = None
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
    logger_pmdarima = logging.getLogger("pmdarima")
    logger_pmdarima.setLevel(logging.ERROR) 
except (ImportError, ValueError) as e:
    print(f"WARNING (ai_models.py): Could not import pmdarima.auto_arima. ARIMA (Auto) functionality will be disabled. Error: {e}")

# Prophet: Conditional import
PROPHET_AVAILABLE = False
Prophet = None 
fb_prophet_logger = None # Initialize to None to avoid NameError if import fails
try:
    from prophet import Prophet
    from prophet.forecaster import logger as prophet_forecaster_logger # Use a distinct alias
    fb_prophet_logger = prophet_forecaster_logger # Assign to the module-level variable
    fb_prophet_logger.setLevel(logging.ERROR) # Set level only if successfully imported
    PROPHET_AVAILABLE = True
except ImportError:
    print("WARNING (ai_models.py): Prophet library not found. Prophet forecasting will be disabled.")

# Survival Analysis
LIFELINES_AVAILABLE = False
KaplanMeierFitter, CoxPHFitter = None, None
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    print("WARNING (ai_models.py): Lifelines library not found. Survival analysis will be disabled.")

# Anomaly Detection
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


# --- Time Series Forecasting ---

@st.cache_data(show_spinner="Training ARIMA model and forecasting...", ttl=3600)
def forecast_arima(
    series: pd.Series,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    n_periods: int = FORECAST_HORIZON
) -> Optional[Dict[str, Any]]:
    series = series.dropna()
    if len(series) < 10: 
        logger.warning("ARIMA: Not enough data points for forecasting (need at least 10).")
        return {"error": "Insufficient data for ARIMA (need at least 10 points)."}

    results: Dict[str, Any] = {}
    model_fit = None 

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            warnings.simplefilter('ignore', UserWarning) 
            warnings.simplefilter('ignore', StatsmodelsValueWarning)

            if order is None: # AUTO ARIMA (pmdarima)
                if not PMDARIMA_AVAILABLE or auto_arima is None:
                    logger.error("ARIMA: auto_arima (pmdarima) not available. Cannot determine order automatically.")
                    return {"error": "auto_arima (pmdarima) is not available for automatic order selection. Please specify ARIMA order manually or check pmdarima installation."}
                
                logger.info(f"Using auto_arima to find best model order for series of length {len(series)}.")
                m_period = 1
                if isinstance(series.index, pd.DatetimeIndex):
                    inferred_freq = pd.infer_freq(series.index)
                    if inferred_freq:
                        freq_upper = inferred_freq.upper()
                        if 'D' in freq_upper: m_period = 7
                        elif 'W' in freq_upper: m_period = 52
                        elif 'M' in freq_upper or 'MS' in freq_upper : m_period = 12
                
                s_from_seasonal_order = seasonal_order[3] if seasonal_order and len(seasonal_order) == 4 else None
                
                model_fit = auto_arima(
                    series,
                    start_p=1, start_q=1, max_p=3, max_q=3,
                    start_P=0, seasonal=True if (s_from_seasonal_order and s_from_seasonal_order > 1) or (m_period > 1 and not s_from_seasonal_order) else False,
                    m=s_from_seasonal_order if s_from_seasonal_order and s_from_seasonal_order > 1 else m_period,
                    d=None, D=None,
                    trace=False, error_action='ignore',
                    suppress_warnings=True, stepwise=True,
                    seasonal_order=seasonal_order if seasonal_order else None
                )
                results['model_order'] = model_fit.order
                results['seasonal_order'] = model_fit.seasonal_order if hasattr(model_fit, 'seasonal_order') else None
                logger.info(f"Auto ARIMA selected order: {results['model_order']}, seasonal_order: {results['seasonal_order']}")

            else: # MANUAL ARIMA (statsmodels)
                logger.info(f"Using specified statsmodels ARIMA order: {order}, seasonal_order: {seasonal_order}")
                if seasonal_order and seasonal_order[3] > 0: 
                    if not isinstance(series.index, pd.DatetimeIndex):
                        try:
                            series.index = pd.to_datetime(series.index)
                        except:
                            return {"error": "Manual SARIMA requires a DatetimeIndex. Could not convert series index."}
                    if series.index.freq is None:
                        inferred_freq = pd.infer_freq(series.index)
                        if inferred_freq:
                            series = series.asfreq(inferred_freq)
                            logger.info(f"Inferred frequency '{inferred_freq}' for manual SARIMA.")
                        else:
                            series = series.asfreq('D') 
                            logger.warning("Could not infer frequency for manual SARIMA, attempting daily ('D'). This might be incorrect.")
                
                model = ARIMA(series, order=order, seasonal_order=seasonal_order)
                model_fit = model.fit()
                results['model_order'] = order
                results['seasonal_order'] = seasonal_order
        
        if model_fit is None:
             return {"error": "ARIMA model fitting failed before forecast step."}

        if PMDARIMA_AVAILABLE and isinstance(model_fit, auto_arima(series, stepwise=False, error_action='ignore').__class__): # type: ignore
            forecast_values, conf_int_values_array = model_fit.predict(n_periods=n_periods, return_conf_int=True, alpha=1-CONFIDENCE_LEVEL)
            conf_int_df = pd.DataFrame(conf_int_values_array, columns=['lower', 'upper'])
        elif isinstance(model_fit, ARIMA(series, order=(1,0,0)).fit().__class__): 
            fc_results = model_fit.get_forecast(steps=n_periods)
            forecast_values = fc_results.predicted_mean
            conf_int_df = fc_results.conf_int(alpha=1-CONFIDENCE_LEVEL) 
        else:
            logger.error(f"ARIMA: Unknown model fit type for prediction: {type(model_fit)}")
            return {"error": "Unknown ARIMA model fit type encountered during prediction."}

        if isinstance(series.index, pd.DatetimeIndex):
            last_date = series.index[-1]
            freq_to_use = series.index.freq or pd.infer_freq(series.index)
            if not freq_to_use: 
                logger.warning("ARIMA forecast: Could not determine frequency for forecast index. Using period numbers.")
                forecast_index = pd.RangeIndex(start=len(series), stop=len(series) + n_periods)
            else:
                 forecast_index = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq_to_use)[1:]
        else: 
            forecast_index = pd.RangeIndex(start=len(series), stop=len(series) + n_periods)

        results['forecast'] = pd.Series(forecast_values, index=forecast_index, name='forecast')
        results['conf_int_lower'] = pd.Series(conf_int_df.iloc[:, 0].values, index=forecast_index, name='conf_int_lower')
        results['conf_int_upper'] = pd.Series(conf_int_df.iloc[:, 1].values, index=forecast_index, name='conf_int_upper')
        
        results['aic'] = model_fit.aic() if hasattr(model_fit, 'aic') and callable(model_fit.aic) else (model_fit.aic if hasattr(model_fit, 'aic') else np.nan)
        results['model_summary'] = str(model_fit.summary()) if hasattr(model_fit, 'summary') else "Summary not available."

    except Exception as e:
        logger.error(f"Error during ARIMA forecasting: {e}", exc_info=True)
        results['error'] = str(e)
        return results

    return results


@st.cache_data(show_spinner="Training Prophet model and forecasting...", ttl=3600)
def forecast_prophet(
    series_df: pd.DataFrame, 
    n_periods: int = FORECAST_HORIZON,
    **prophet_kwargs
) -> Optional[Dict[str, Any]]:
    if not PROPHET_AVAILABLE or Prophet is None:
        logger.warning("Prophet library is not installed/loaded. Skipping Prophet forecasting.")
        return {"error": "Prophet library not installed or failed to load."}

    if not isinstance(series_df, pd.DataFrame) or 'ds' not in series_df.columns or 'y' not in series_df.columns:
        logger.error("Prophet: Input must be a DataFrame with 'ds' and 'y' columns.")
        return {"error": "Input DataFrame must contain 'ds' (datetime) and 'y' (numeric) columns."}
    
    temp_df = series_df.copy()
    try:
        temp_df['ds'] = pd.to_datetime(temp_df['ds'])
    except Exception as e:
        logger.error(f"Prophet: Could not convert 'ds' column to datetime: {e}")
        return {"error": f"Could not convert 'ds' column to datetime: {e}"}
    try:
        temp_df['y'] = pd.to_numeric(temp_df['y'])
    except Exception as e:
        logger.error(f"Prophet: Could not convert 'y' column to numeric: {e}")
        return {"error": f"Could not convert 'y' column to numeric: {e}"}

    temp_df = temp_df[['ds', 'y']].dropna()
    if len(temp_df) < 5: 
        return {"error": "Insufficient data for Prophet forecasting (need at least 5 non-NaN points)."}

    inferred_freq = pd.infer_freq(temp_df['ds'])
    logger.info(f"Prophet: Inferred frequency for 'ds' column: {inferred_freq}")
    
    results = {}
    try:
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            model = Prophet(**prophet_kwargs)
            model.fit(temp_df)
        
        future_df = model.make_future_dataframe(periods=n_periods, freq=inferred_freq if inferred_freq else 'D')
        forecast_df = model.predict(future_df)
        
        results['forecast_df'] = forecast_df 
        results['model'] = model 
        results['model_params'] = model.params if hasattr(model, 'params') else None
    except Exception as e:
        logger.error(f"Error during Prophet forecasting: {e}", exc_info=True)
        results['error'] = str(e)
    return results

@st.cache_data(show_spinner="Performing Kaplan-Meier survival analysis...", ttl=3600)
def survival_analysis_kaplan_meier(
    durations: Union[List[float], pd.Series],
    event_observed: Union[List[bool], pd.Series],
    confidence_level: float = CONFIDENCE_LEVEL
) -> Optional[Dict[str, Any]]:
    if not LIFELINES_AVAILABLE or KaplanMeierFitter is None:
        return {"error": "Lifelines library not available for Kaplan-Meier."}
    
    durations_series = pd.Series(durations).dropna()
    if durations_series.empty:
        return {"error": "Durations data is empty after NaN removal for Kaplan-Meier."}
    
    event_observed_series = pd.Series(event_observed).loc[durations_series.index].fillna(True).astype(bool)

    if len(durations_series) != len(event_observed_series) or len(durations_series) < 2:
        return {"error": "Data mismatch or insufficient data for Kaplan-Meier (need at least 2 observations)."}
    results = {}
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(durations_series, event_observed=event_observed_series, alpha=(1-confidence_level))
        results['kmf_model'] = kmf
        results['survival_function_df'] = kmf.survival_function_
        results['median_survival_time'] = kmf.median_survival_time_
        results['confidence_interval_df'] = kmf.confidence_interval_survival_function_
        results['confidence_level'] = confidence_level
    except Exception as e:
        logger.error(f"Error during Kaplan-Meier analysis: {e}", exc_info=True)
        results['error'] = str(e)
    return results


@st.cache_data(show_spinner="Performing Cox Proportional Hazards analysis...", ttl=3600)
def survival_analysis_cox_ph(
    df_cox: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    if not LIFELINES_AVAILABLE or CoxPHFitter is None:
        return {"error": "Lifelines library not available for Cox PH."}
    
    if not all(col in df_cox.columns for col in [duration_col, event_col]):
        return {"error": f"Duration column '{duration_col}' or event column '{event_col}' missing in DataFrame for Cox PH."}
    
    analysis_cols = [duration_col, event_col] + (covariate_cols if covariate_cols else [])
    df_analysis = df_cox[analysis_cols].copy().dropna()

    if len(df_analysis) < 10: 
        return {"error": "Insufficient data for Cox PH model after NaN removal (need at least 10 rows)."}
    results = {}
    try:
        cph = CoxPHFitter()
        cph.fit(df_analysis, duration_col=duration_col, event_col=event_col)
        results['cph_model'] = cph
        results['summary_df'] = cph.summary
    except Exception as e:
        logger.error(f"Error during Cox PH analysis: {e}", exc_info=True)
        results['error'] = str(e)
    return results


@st.cache_data(show_spinner="Performing anomaly detection...", ttl=3600)
def detect_anomalies(
    data: Union[pd.DataFrame, pd.Series],
    method: str = 'isolation_forest',
    contamination: Union[str, float] = 'auto',
    **model_kwargs
) -> Optional[Dict[str, Any]]:
    if isinstance(data, pd.Series):
        data_df = data.to_frame(name=data.name or 'value')
    elif isinstance(data, pd.DataFrame):
        data_df = data.copy()
    else:
        return {"error": "Invalid input data type for anomaly detection. Must be Series or DataFrame."}

    numeric_cols = data_df.select_dtypes(include=np.number).columns
    if not numeric_cols.any():
        return {"error": "No numeric columns found in the data for anomaly detection."}
    data_df_numeric = data_df[numeric_cols].dropna()

    if data_df_numeric.empty or len(data_df_numeric) < 5:
        return {"error": "Insufficient numeric data for anomaly detection after NaN removal (need at least 5 rows)."}
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_df_numeric)
    results = {}
    try:
        if method == 'isolation_forest':
            contam_val = 'auto' if contamination == 'auto' else float(contamination)
            if not (isinstance(contam_val, str) and contam_val == 'auto') and not (0 < contam_val <= 0.5):
                logger.warning(f"Isolation Forest: Contamination '{contam_val}' invalid. Using 'auto'.")
                contam_val = 'auto'
            model = IsolationForest(contamination=contam_val, random_state=42, **model_kwargs)
        elif method == 'one_class_svm':
            nu_val = 0.05 
            if contamination != 'auto':
                try:
                    nu_val = float(contamination)
                    if not (0 < nu_val <= 1.0): 
                        logger.warning(f"OneClassSVM: nu value (from contamination '{contamination}') '{nu_val}' invalid. Using default 0.05.")
                        nu_val = 0.05
                except ValueError:
                    logger.warning(f"OneClassSVM: Could not convert contamination '{contamination}' to float for nu. Using default 0.05.")
                    nu_val = 0.05
            model = OneClassSVM(nu=nu_val, kernel="rbf", gamma='auto', **model_kwargs)
        else:
            return {"error": f"Unsupported anomaly detection method: {method}."}
        
        model.fit(data_scaled)
        predictions = model.predict(data_scaled)
        
        results['anomalies_flags'] = pd.Series(predictions == -1, index=data_df_numeric.index)
        results['anomalies_indices'] = data_df_numeric.index[predictions == -1].tolist()
        
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(data_scaled)
        elif hasattr(model, 'score_samples'):
            scores = -model.score_samples(data_scaled) 
        else:
            scores = None
        
        if scores is not None:
            results['anomaly_scores'] = pd.Series(scores, index=data_df_numeric.index)
            
        results['model'] = model
        results['method'] = method
        results['processed_numeric_cols'] = numeric_cols.tolist()

    except Exception as e:
        logger.error(f"Error during anomaly detection ({method}): {e}", exc_info=True)
        results['error'] = str(e)
    return results
