"""
services/analysis_service.py

Orchestrates analytical calculations and model executions.
Includes a standalone function for fetching benchmark data.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import yfinance as yf 
import datetime

try:
    from config import APP_TITLE, RISK_FREE_RATE, EXPECTED_COLUMNS, FORECAST_HORIZON, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS
    from calculations import calculate_all_kpis
    from statistical_methods import (
        bootstrap_confidence_interval, fit_distributions_to_pnl,
        decompose_time_series, detect_change_points
    )
    from stochastic_models import (
        simulate_gbm, fit_ornstein_uhlenbeck,
        simulate_merton_jump_diffusion, fit_markov_chain_trade_sequence
    )
    from ai_models import (
        forecast_arima, forecast_prophet, PROPHET_AVAILABLE, PMDARIMA_AVAILABLE,
        survival_analysis_kaplan_meier, survival_analysis_cox_ph, LIFELINES_AVAILABLE,
        detect_anomalies
    )
    from plotting import (
        plot_pnl_distribution
    )
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in AnalysisService module: {e}. Some functionalities may fail.")
    APP_TITLE = "TradingDashboard_ErrorState"; RISK_FREE_RATE = 0.02; EXPECTED_COLUMNS = {"pnl": "pnl", "date": "date"}; FORECAST_HORIZON = 30
    PROPHET_AVAILABLE = False; PMDARIMA_AVAILABLE = False; LIFELINES_AVAILABLE = False; CONFIDENCE_LEVEL = 0.95; BOOTSTRAP_ITERATIONS = 1000
    def calculate_all_kpis(df, rfr, benchmark_daily_returns=None, initial_capital=None): return {"error": "calc_kpis not loaded"}
    def bootstrap_confidence_interval(d, _sf, **kw): return {"error": "bootstrap_ci not loaded", "lb":np.nan, "ub":np.nan, "bootstrap_statistics": []}
    # Add dummy for survival_analysis_kaplan_meier if it's called by a method in this class during init or similar
    def survival_analysis_kaplan_meier(*args, **kwargs): return {"error": "survival_analysis_kaplan_meier not loaded"}


import logging
logger = logging.getLogger(APP_TITLE)

@st.cache_data(ttl=3600, show_spinner="Fetching benchmark data...") 
def get_benchmark_data_static(
    ticker: str, 
    start_date_str: str, 
    end_date_str: str
) -> Optional[pd.Series]:
    logger_static_func = logging.getLogger(f"{APP_TITLE}.get_benchmark_data_static")
    if not ticker:
        logger_static_func.info("No benchmark ticker provided. Skipping data fetch.")
        return None
    try:
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str)
        if start_dt >= end_dt:
            logger_static_func.warning(f"Benchmark start date {start_date_str} >= end date {end_date_str}. Cannot fetch.")
            return None
        fetch_end_dt = end_dt + pd.Timedelta(days=1)
        logger_static_func.info(f"Fetching benchmark: {ticker} from {start_dt.date()} to {end_dt.date()}")
        data = yf.download(ticker, start=start_dt, end=fetch_end_dt, progress=False, auto_adjust=True, actions=False)
        if data.empty or 'Close' not in data.columns:
            logger_static_func.warning(f"No data or 'Close' not found for benchmark {ticker}.")
            return None
        daily_adj_close = data['Close'].dropna()
        if len(daily_adj_close) < 2:
            logger_static_func.warning(f"Not enough benchmark data for {ticker} (<2 points).")
            return None
        daily_returns = daily_adj_close.pct_change().dropna()
        daily_returns.name = f"{ticker}_returns"
        logger_static_func.info(f"Fetched benchmark returns for {ticker}. Shape: {daily_returns.shape}")
        return daily_returns
    except Exception as e:
        logger_static_func.error(f"Error fetching benchmark for {ticker}: {e}", exc_info=True)
        return None

class AnalysisService:
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE) 
        self.logger.info(f"AnalysisService initialized. Has perform_kaplan_meier_analysis: {hasattr(self, 'perform_kaplan_meier_analysis')}")
        if not PMDARIMA_AVAILABLE: self.logger.warning("PMDARIMA (auto_arima) unavailable in AnalysisService.")
        if not PROPHET_AVAILABLE: self.logger.warning("Prophet unavailable in AnalysisService.")
        if not LIFELINES_AVAILABLE: self.logger.warning("Lifelines (survival analysis) unavailable in AnalysisService.")

    def get_core_kpis(
        self, 
        trades_df: pd.DataFrame, 
        risk_free_rate: Optional[float] = None,
        benchmark_daily_returns: Optional[pd.Series] = None,
        initial_capital: Optional[float] = None
    ) -> Dict[str, Any]:
        if trades_df is None or trades_df.empty: return {"error": "Input data for KPI calculation is empty."}
        rfr = risk_free_rate if risk_free_rate is not None else RISK_FREE_RATE
        try:
            pnl_col_name = EXPECTED_COLUMNS.get('pnl')
            date_col_name = EXPECTED_COLUMNS.get('date') # Ensure date column is also checked
            if not pnl_col_name or pnl_col_name not in trades_df.columns:
                return {"error": f"Required PnL column ('{pnl_col_name}') not found."}
            if not date_col_name or date_col_name not in trades_df.columns:
                return {"error": f"Required Date column ('{date_col_name}') not found."}
            if trades_df[pnl_col_name].isnull().all():
                 return {"error": f"PnL column ('{pnl_col_name}') contains only NaN values."}

            kpi_results = calculate_all_kpis(
                trades_df, 
                risk_free_rate=rfr,
                benchmark_daily_returns=benchmark_daily_returns,
                initial_capital=initial_capital
            )
            if pd.isna(kpi_results.get('total_pnl')) and pd.isna(kpi_results.get('sharpe_ratio')):
                 self.logger.warning("Several critical KPIs are NaN. This might indicate issues with input PnL data.")
            return kpi_results
        except Exception as e: self.logger.error(f"Error calculating core KPIs: {e}", exc_info=True); return {"error": str(e)}

    def perform_kaplan_meier_analysis(self, durations: pd.Series, event_observed: pd.Series) -> Dict[str, Any]:
        self.logger.debug(f"Attempting Kaplan-Meier analysis. Lifelines available: {LIFELINES_AVAILABLE}")
        if not LIFELINES_AVAILABLE: 
            return {"error": "Lifelines library not available for Kaplan-Meier analysis."}
        if durations is None or durations.dropna().empty:
            return {"error": "Durations data is empty or all NaN for Kaplan-Meier."}
        if len(durations.dropna()) < 5 : # Lifelines might need a few points
             return {"error": "Insufficient duration data points (need at least 5) for Kaplan-Meier."}
        
        # Align event_observed with non-NaN durations
        valid_durations = durations.dropna()
        aligned_event_observed = event_observed.loc[valid_durations.index].fillna(True).astype(bool)

        if len(valid_durations) != len(aligned_event_observed):
            return {"error": "Mismatch between valid durations and event observations after alignment."}

        try: 
            # survival_analysis_kaplan_meier is imported from ai_models.py
            result = survival_analysis_kaplan_meier(valid_durations, aligned_event_observed)
            if result is None: # Function itself might return None on internal error
                return {"error": "Kaplan-Meier analysis function returned None unexpectedly."}
            self.logger.info("Kaplan-Meier analysis successful in service.")
            return result
        except Exception as e: 
            self.logger.error(f"Error during Kaplan-Meier analysis in service: {e}", exc_info=True)
            return {"error": str(e)}

    # ... (rest of AnalysisService methods: get_bootstrapped_kpi_cis, etc.) ...
    def get_bootstrapped_kpi_cis(self, trades_df: pd.DataFrame, kpis_to_bootstrap: Optional[List[str]] = None) -> Dict[str, Any]:
        if trades_df is None or trades_df.empty: return {"error": "Input data for CI calculation is empty."}
        if kpis_to_bootstrap is None: kpis_to_bootstrap = ['avg_trade_pnl', 'win_rate', 'sharpe_ratio']
        
        pnl_col_name = EXPECTED_COLUMNS.get('pnl')
        if not pnl_col_name or pnl_col_name not in trades_df.columns:
            return {"error": f"PnL column ('{pnl_col_name}') not found for CI calculation."}
            
        pnl_series = trades_df[pnl_col_name].dropna()
        if pnl_series.empty or len(pnl_series) < 2:
             return {"error": "PnL data insufficient (empty or < 2 values) for CI calculation."}

        confidence_intervals: Dict[str, Any] = {}
        for kpi_key in kpis_to_bootstrap:
            stat_fn: Optional[Callable[[pd.Series], float]] = None
            
            if kpi_key == 'avg_trade_pnl': stat_fn = np.mean
            elif kpi_key == 'win_rate': stat_fn = lambda x: (np.sum(x > 0) / len(x)) * 100 if len(x) > 0 else 0.0
            elif kpi_key == 'sharpe_ratio':
                def simplified_sharpe_stat_fn(returns_sample: pd.Series) -> float:
                    if len(returns_sample) < 2: return 0.0
                    std_dev = returns_sample.std()
                    if std_dev == 0 or np.isnan(std_dev): return 0.0 if returns_sample.mean() <= 0 else np.inf
                    return returns_sample.mean() / std_dev
                stat_fn = simplified_sharpe_stat_fn
            
            if stat_fn:
                try:
                    res = bootstrap_confidence_interval(pnl_series, _statistic_func=stat_fn)
                    if 'error' not in res: confidence_intervals[kpi_key] = (res['lower_bound'], res['upper_bound'])
                    else: confidence_intervals[kpi_key] = (np.nan, np.nan); self.logger.warning(f"CI calc error for {kpi_key}: {res['error']}")
                except Exception as e: self.logger.error(f"Exception during bootstrap for {kpi_key}: {e}", exc_info=True); confidence_intervals[kpi_key] = (np.nan, np.nan)
            else: confidence_intervals[kpi_key] = (np.nan, np.nan); self.logger.warning(f"No CI stat_fn for {kpi_key}")
        return confidence_intervals

    def get_single_bootstrap_ci_visual_data(
        self,
        data_series: pd.Series,
        statistic_func: Callable[[pd.Series], float],
        n_iterations: int = BOOTSTRAP_ITERATIONS,
        confidence_level: float = CONFIDENCE_LEVEL
    ) -> Dict[str, Any]:
        if data_series is None or data_series.dropna().empty:
            return {"error": "Input data series for bootstrapping is empty or all NaN."}
        if len(data_series.dropna()) < 2:
            return {"error": "Insufficient data points (need at least 2) for bootstrapping."}
        
        try:
            results = bootstrap_confidence_interval(
                data=data_series.dropna(),
                _statistic_func=statistic_func,
                n_iterations=n_iterations,
                confidence_level=confidence_level
            )
            return results
        except Exception as e:
            self.logger.error(f"Error in get_single_bootstrap_ci_visual_data: {e}", exc_info=True)
            return {"error": str(e)}

    def get_time_series_decomposition(self, series: pd.Series, model: str = 'additive', period: Optional[int] = None) -> Dict[str, Any]:
        if series is None or series.empty: return {"error": "Series is empty for decomposition."}
        min_len = (2 * (period or 2)) + 1
        if len(series.dropna()) < min_len : return {"error": f"Series too short (need {min_len} non-NaN points) for period {period}."}
        try: 
            result = decompose_time_series(series.dropna(), model=model, period=period)
            return {"decomposition_result": result} if result is not None else {"error": "Decomposition returned None."}
        except Exception as e: self.logger.error(f"Error in TS decomp: {e}", exc_info=True); return {"error": str(e)}
    
    def analyze_pnl_distribution_fit(self, pnl_series: pd.Series, distributions_to_try: Optional[List[str]] = None) -> Dict[str, Any]:
        if pnl_series is None or pnl_series.dropna().empty: return {"error": "PnL series is empty."}
        try: return fit_distributions_to_pnl(pnl_series.dropna(), distributions_to_try=distributions_to_try)
        except Exception as e: self.logger.error(f"Error in PnL dist fit: {e}", exc_info=True); return {"error": str(e)}

    def find_change_points(self, series: pd.Series, model: str = "l2", penalty: str = "bic") -> Dict[str, Any]:
        if series is None or series.dropna().empty or len(series.dropna()) < 10: return {"error": "Series too short for change point detection."}
        try: return detect_change_points(series.dropna(), model=model, penalty=penalty)
        except Exception as e: self.logger.error(f"Error in change point detect: {e}", exc_info=True); return {"error": str(e)}

    def run_gbm_simulation(self, s0: float, mu: float, sigma: float, dt: float, n_steps: int, n_sims: int = 1) -> Dict[str, Any]:
        try:
            paths = simulate_gbm(s0, mu, sigma, dt, n_steps, n_sims)
            return {"paths": paths} if (paths is not None and paths.size > 0) else {"error": "GBM simulation returned empty or invalid paths."}
        except Exception as e: self.logger.error(f"Error in GBM sim: {e}", exc_info=True); return {"error": str(e)}

    def estimate_ornstein_uhlenbeck(self, series: pd.Series) -> Dict[str, Any]:
        if series is None or series.dropna().empty or len(series.dropna()) < 20: return {"error": "Series too short for OU fitting."}
        try: 
            result = fit_ornstein_uhlenbeck(series.dropna())
            return result if result is not None else {"error": "OU fitting returned None."}
        except Exception as e: self.logger.error(f"Error in OU fit: {e}", exc_info=True); return {"error": str(e)}

    def analyze_markov_chain_trades(self, pnl_series: pd.Series, n_states: int = 2) -> Dict[str, Any]:
        if pnl_series is None or pnl_series.dropna().empty or len(pnl_series.dropna()) < 10: return {"error": "PnL series too short for Markov chain."}
        try: 
            result = fit_markov_chain_trade_sequence(pnl_series.dropna(), n_states=n_states)
            return result if result is not None else {"error": "Markov chain analysis returned None."}
        except Exception as e: self.logger.error(f"Error in Markov chain: {e}", exc_info=True); return {"error": str(e)}

    def get_arima_forecast(self, series: pd.Series, order: Optional[Tuple[int,int,int]]=None, seasonal_order: Optional[Tuple[int, int, int, int]] = None, n_periods: int = FORECAST_HORIZON) -> Dict[str,Any]:
        if not PMDARIMA_AVAILABLE and order is None: return {"error": "pmdarima (for auto_arima) is not available. Please specify ARIMA order or check installation."}
        if series is None or series.dropna().empty or len(series.dropna()) < 20: return {"error": "Series too short for ARIMA."}
        try: 
            result = forecast_arima(series.dropna(), order=order, seasonal_order=seasonal_order, n_periods=n_periods)
            return result if result is not None else {"error": "ARIMA forecast returned None."}
        except Exception as e: self.logger.error(f"Error in ARIMA forecast: {e}", exc_info=True); return {"error": str(e)}

    def get_prophet_forecast(self, series_df: pd.DataFrame, n_periods: int = FORECAST_HORIZON) -> Dict[str,Any]:
        if not PROPHET_AVAILABLE: return {"error": "Prophet library not installed/loaded."}
        if series_df is None or series_df.empty or len(series_df) < 10: return {"error": "DataFrame too short for Prophet."}
        try: 
            result = forecast_prophet(series_df, n_periods=n_periods)
            return result if result is not None else {"error": "Prophet forecast returned None."}
        except Exception as e: self.logger.error(f"Error in Prophet forecast: {e}", exc_info=True); return {"error": str(e)}

    def find_anomalies(self, data: Union[pd.DataFrame, pd.Series], method: str = 'isolation_forest', contamination: Union[str, float] = 'auto') -> Dict[str, Any]:
        if data is None or data.empty or len(data) < 10: return {"error": "Data too short for anomaly detection."}
        try: 
            result = detect_anomalies(data, method=method, contamination=contamination)
            return result if result is not None else {"error": "Anomaly detection returned None."}
        except Exception as e: self.logger.error(f"Error in anomaly detection: {e}", exc_info=True); return {"error": str(e)}

    # perform_cox_ph_analysis was missing, re-added it.
    def perform_cox_ph_analysis(self, df_cox: pd.DataFrame, duration_col: str, event_col: str, covariate_cols: Optional[List[str]]=None) -> Dict[str,Any]:
        self.logger.debug(f"Attempting Cox PH analysis. Lifelines available: {LIFELINES_AVAILABLE}")
        if not LIFELINES_AVAILABLE: 
            return {"error": "Lifelines library not available for Cox PH analysis."}
        if df_cox is None or df_cox.empty or len(df_cox) < 10: # Cox might need more depending on covariates
            return {"error": "DataFrame too short or empty for Cox PH model (need at least 10 rows)."}
        
        required_cols = [duration_col, event_col]
        if covariate_cols:
            required_cols.extend(covariate_cols)
        
        missing_cols = [col for col in required_cols if col not in df_cox.columns]
        if missing_cols:
            return {"error": f"Missing columns for Cox PH analysis: {', '.join(missing_cols)}"}
            
        try: 
            # survival_analysis_cox_ph is imported from ai_models.py
            result = survival_analysis_cox_ph(df_cox, duration_col, event_col, covariate_cols)
            if result is None:
                 return {"error": "Cox PH analysis function returned None unexpectedly."}
            self.logger.info("Cox PH analysis successful in service.")
            return result
        except Exception as e: 
            self.logger.error(f"Error during Cox PH analysis in service: {e}", exc_info=True)
            return {"error": str(e)}

    def generate_pnl_distribution_plot(self, trades_df: pd.DataFrame, theme: str = 'dark') -> Optional[Any]:
        if trades_df is None or trades_df.empty: return None
        pnl_col = EXPECTED_COLUMNS.get('pnl')
        if not pnl_col or pnl_col not in trades_df.columns: return None
        try: return plot_pnl_distribution(trades_df, pnl_col=pnl_col, title="PnL Distribution (per Trade)", theme=theme)
        except Exception as e: self.logger.error(f"Error generating PnL dist plot: {e}", exc_info=True); return None
