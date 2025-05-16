"""
statistical_methods.py

Implements advanced statistical methods for trading performance analysis,
including hypothesis testing, bootstrapping, distribution fitting,
time series decomposition, and change point detection.
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
import ruptures as rpt
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

try:
    from config import BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL, DISTRIBUTIONS_TO_FIT, APP_TITLE
except ImportError:
    APP_TITLE = "TradingDashboard_Default_Stats"
    BOOTSTRAP_ITERATIONS = 1000
    CONFIDENCE_LEVEL = 0.95
    DISTRIBUTIONS_TO_FIT = ['norm', 't']

import logging
logger = logging.getLogger(APP_TITLE)


@st.cache_data(show_spinner="Performing hypothesis test...", ttl=3600)
def perform_hypothesis_test(
    data1: Union[List[float], pd.Series, np.ndarray, pd.DataFrame],
    data2: Optional[Union[List[float], pd.Series]] = None,
    test_type: str = 't-test_ind', alpha: float = 0.05, **kwargs
) -> Dict[str, Any]:
    # ... (implementation as before) ...
    results: Dict[str, Any] = {"test_type": test_type, "alpha": alpha}
    if test_type != 'chi-squared':
        if isinstance(data1, list) and test_type == 'anova': pass
        else: data1 = pd.Series(data1).dropna()
    if data2 is not None: data2 = pd.Series(data2).dropna()
    try:
        if test_type == 't-test_ind':
            if data2 is None or len(data1) < 2 or len(data2) < 2: return {"error": "Insufficient data for independent t-test."}
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=kwargs.get('equal_var', False), nan_policy='omit')
        elif test_type == 't-test_rel':
            if data2 is None or len(data1) != len(data2) or len(data1) < 2: return {"error": "Data for paired t-test must be of equal length and sufficient size."}
            stat, p_value = stats.ttest_rel(data1, data2, nan_policy='omit')
        elif test_type == 'anova':
            if not isinstance(data1, list) or len(data1) < 2 or any(len(pd.Series(g).dropna()) < 2 for g in data1): return {"error": "ANOVA requires at least two groups with at least two observations each."}
            valid_groups = [pd.Series(g).dropna() for g in data1 if len(pd.Series(g).dropna()) >=2]
            if len(valid_groups) < 2: return {"error": "ANOVA requires at least two valid groups after NaN removal."}
            stat, p_value = stats.f_oneway(*valid_groups)
        elif test_type == 'chi-squared':
            if not isinstance(data1, (np.ndarray, pd.DataFrame)) or pd.DataFrame(data1).ndim != 2: return {"error": "Chi-squared test requires a 2D contingency table."}
            chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(pd.DataFrame(data1))
            stat = chi2_stat; results['df'] = dof; results['expected_frequencies'] = expected_freq.tolist()
        else: return {"error": f"Unsupported test type: {test_type}"}
        results['statistic'] = stat; results['p_value'] = p_value
        results['significant'] = p_value < alpha
        results['interpretation'] = f"Result is {'statistically significant' if results['significant'] else 'not statistically significant'} at alpha = {alpha} (p-value: {p_value:.4f})."
        results['conclusion'] = "Reject null hypothesis." if results['significant'] else "Fail to reject null hypothesis."
    except Exception as e: logger.error(f"Error in hypothesis test '{test_type}': {e}", exc_info=True); results['error'] = str(e)
    return results

@st.cache_data(show_spinner="Performing bootstrapping for CIs...", ttl=3600)
def bootstrap_confidence_interval(
    data: Union[List[float], pd.Series],
    _statistic_func: Callable[[pd.Series], float], # Parameter name starts with underscore
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    confidence_level: float = CONFIDENCE_LEVEL
) -> Dict[str, Any]:
    data_series = pd.Series(data).dropna()
    if len(data_series) < 2:
        logger.warning("Bootstrapping CI: Not enough data points (need at least 2).")
        observed_stat_val = np.nan
        if not data_series.empty:
            try: observed_stat_val = _statistic_func(data_series)
            except: pass
        return {"lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": observed_stat_val, "bootstrap_statistics": [], "error": "Insufficient data for bootstrapping."}

    bootstrap_statistics = np.empty(n_iterations)
    n_size = len(data_series)
    data_values = data_series.values

    for i in range(n_iterations):
        resample_values = np.random.choice(data_values, size=n_size, replace=True)
        bootstrap_statistics[i] = _statistic_func(pd.Series(resample_values))

    observed_statistic = _statistic_func(data_series)
    alpha = (1 - confidence_level) / 2
    valid_bootstrap_stats = bootstrap_statistics[~np.isnan(bootstrap_statistics)]
    if len(valid_bootstrap_stats) < n_iterations * 0.1:
        logger.warning(f"Bootstrapping for {_statistic_func.__name__ if hasattr(_statistic_func, '__name__') else 'custom_stat'} resulted in many NaNs.")
        return {"lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": observed_statistic, "bootstrap_statistics": bootstrap_statistics.tolist(), "error": "Many NaNs in bootstrap samples."}

    lower_bound = np.percentile(valid_bootstrap_stats, alpha * 100)
    upper_bound = np.percentile(valid_bootstrap_stats, (1 - alpha) * 100)

    return {"lower_bound": lower_bound, "upper_bound": upper_bound, "observed_statistic": observed_statistic, "bootstrap_statistics": bootstrap_statistics.tolist()}

# ... (rest of statistical_methods.py as previously provided) ...
@st.cache_data(show_spinner="Fitting distributions to PnL data...", ttl=3600)
def fit_distributions_to_pnl(pnl_series: pd.Series, distributions_to_try: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    if distributions_to_try is None: distributions_to_try = DISTRIBUTIONS_TO_FIT
    pnl_clean = pnl_series.dropna()
    if pnl_clean.empty: return {"error": "PnL series is empty after NaN removal."}
    results = {}
    for dist_name in distributions_to_try:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(pnl_clean)
            D, p_value = stats.kstest(pnl_clean, dist_name, args=params, N=len(pnl_clean))
            results[dist_name] = {
                "params": params, "param_names": [p for p in dist.shapes.split(',') + ['loc', 'scale'] if p] if dist.shapes else ['loc', 'scale'],
                "ks_statistic": D, "ks_p_value": p_value,
                "interpretation": f"KS p-value ({p_value:.4f}) suggests data {'may come' if p_value > 0.05 else 'likely does not come'} from {dist_name}."
            }
        except Exception as e: logger.error(f"Error fitting {dist_name}: {e}"); results[dist_name] = {"error": str(e)}
    return results

@st.cache_data(show_spinner="Decomposing time series...", ttl=3600)
def decompose_time_series(series: pd.Series, model: str = 'additive', period: Optional[int] = None, extrapolate_trend: str = 'freq') -> Optional[DecomposeResult]:
    series_clean = series.dropna()
    if not isinstance(series_clean.index, pd.DatetimeIndex):
        try: series_clean.index = pd.to_datetime(series_clean.index)
        except: logger.error("TS Decomp: Failed to convert index to DatetimeIndex."); return None
    min_len = (2 * (period or 2)) +1
    if series_clean.empty or len(series_clean) < min_len:
        logger.warning(f"TS Decomp: Not enough data (need {min_len}, got {len(series_clean)})."); return None
    if series_clean.index.freq is None and period is None:
        inferred_freq = pd.infer_freq(series_clean.index)
        if inferred_freq: series_clean = series_clean.asfreq(inferred_freq)
        else:
            try:
                series_daily = series_clean.resample('D').mean()
                if not series_daily.isnull().all(): series_clean = series_daily.interpolate(method='linear')
                else: logger.warning("TS Decomp: Daily resampling yielded all NaNs."); return None
            except Exception as e: logger.error(f"TS Decomp: Error resampling to daily: {e}"); return None
    try: return seasonal_decompose(series_clean, model=model, period=period, extrapolate_trend=extrapolate_trend)
    except Exception as e: logger.error(f"TS Decomp error: {e}", exc_info=True); return None

@st.cache_data(show_spinner="Detecting change points...", ttl=3600)
def detect_change_points(series: pd.Series, model: str = "l2", penalty: str = "bic", n_bkps: Optional[int] = None, min_size: int = 2) -> Dict[str, Any]:
    series_values = series.dropna().values
    if len(series_values) < min_size * 2 + 1: return {"error": f"Insufficient data for change point detection (need at least {min_size*2+1} points)."}
    results: Dict[str, Any] = {}
    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(series_values)
        if n_bkps is not None: breakpoints_idx = algo.predict(n_bkps=n_bkps)
        else:
            pen_val_map = {"aic": rpt.penalty.aic, "bic": rpt.penalty.bic, "mbic": rpt.penalty.mbic}
            pen_func = pen_val_map.get(penalty.lower(), lambda n_s: float(penalty) if isinstance(penalty, (float,int)) else rpt.penalty.bic(n_s))
            pen_value = pen_func(len(series_values))
            breakpoints_idx = algo.predict(pen=pen_value)
        actual_change_points_indices = [series.index[i-1] for i in breakpoints_idx if 0 < i < len(series_values)] if isinstance(series.index, pd.DatetimeIndex) else [i for i in breakpoints_idx if 0 < i < len(series_values)]
        results = {'breakpoints_algo_indices': breakpoints_idx, 'change_points_original_indices': actual_change_points_indices, 'series_to_plot': series}
    except Exception as e: logger.error(f"Change point detection error: {e}", exc_info=True); results['error'] = str(e)
    return results
