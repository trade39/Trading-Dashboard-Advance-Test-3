"""
calculations.py

Implements calculations for all specified Key Performance Indicators (KPIs)
and provides functions for their qualitative interpretation and color-coding
based on thresholds defined in config.py.
Includes benchmark-relative metrics like Alpha and Beta.
"""
import pandas as pd
import numpy as np
from scipy import stats # Retained for potential future use, though not directly in KPIs now
from typing import Dict, Any, Tuple, List, Optional, Union # Added Union

import logging

# Assuming config.py is in the root directory
from config import RISK_FREE_RATE, KPI_CONFIG, COLORS, EXPECTED_COLUMNS, APP_TITLE

logger = logging.getLogger(APP_TITLE) # Use APP_TITLE from config
# Basic handler if not configured by main app (e.g. if module is tested standalone)
if not logger.handlers and not logging.getLogger().handlers: # Check root logger too
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False # Avoid duplicate logs if root logger also gets configured


def _calculate_returns(pnl_series: pd.Series, initial_capital: Optional[float] = None) -> pd.Series:
    if pnl_series.empty:
        return pd.Series(dtype=float)
    if initial_capital and initial_capital != 0:
        return pnl_series / initial_capital
    return pnl_series

def _calculate_drawdowns(cumulative_pnl: pd.Series) -> Tuple[pd.Series, float, float, pd.Series]:
    if cumulative_pnl.empty:
        return pd.Series(dtype=float), 0.0, 0.0, pd.Series(dtype=float)

    high_water_mark = cumulative_pnl.cummax()
    drawdown_series = high_water_mark - cumulative_pnl
    max_drawdown_abs = drawdown_series.max() if not drawdown_series.empty else 0.0
    
    # Replace 0s in HWM with NaN before division to avoid 0/0 or X/0 issues if HWM is 0
    hwm_for_pct = high_water_mark.replace(0, np.nan)
    drawdown_pct_series = (drawdown_series / hwm_for_pct).fillna(0) * 100
    max_drawdown_pct = drawdown_pct_series.max() if not drawdown_pct_series.empty else 0.0
    
    if max_drawdown_abs > 0 and max_drawdown_pct == 0 and high_water_mark.abs().sum() == 0:
         max_drawdown_pct = 100.0

    return drawdown_series, max_drawdown_abs, max_drawdown_pct, drawdown_pct_series


def _calculate_streaks(pnl_series: pd.Series) -> Tuple[int, int]:
    if pnl_series.empty:
        return 0, 0
    wins = pnl_series > 0
    losses = pnl_series < 0
    max_win_streak = current_win_streak = 0
    for w in wins:
        current_win_streak = current_win_streak + 1 if w else 0
        max_win_streak = max(max_win_streak, current_win_streak)
    max_loss_streak = current_loss_streak = 0
    for l_val in losses:
        current_loss_streak = current_loss_streak + 1 if l_val else 0
        max_loss_streak = max(max_loss_streak, current_loss_streak)
    return int(max_win_streak), int(max_loss_streak)

def calculate_benchmark_metrics(
    strategy_daily_returns: Union[pd.Series, pd.DataFrame], # Allow DataFrame for robustness
    benchmark_daily_returns: Union[pd.Series, pd.DataFrame], # Allow DataFrame for robustness
    risk_free_rate: float, 
    periods_per_year: int = 252
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "alpha": np.nan, "beta": np.nan, "benchmark_correlation": np.nan,
        "tracking_error": np.nan, "information_ratio": np.nan
    }

    sdr = strategy_daily_returns
    bdr = benchmark_daily_returns

    # Ensure inputs are 1D pandas Series
    if isinstance(sdr, pd.DataFrame):
        if sdr.shape[1] == 1: sdr = sdr.iloc[:, 0].copy() # Ensure it's a Series
        else: logger.error("strategy_daily_returns is a multi-column DataFrame for benchmark metrics."); return metrics
    if isinstance(bdr, pd.DataFrame):
        if bdr.shape[1] == 1: bdr = bdr.iloc[:, 0].copy() # Ensure it's a Series
        else: logger.error("benchmark_daily_returns is a multi-column DataFrame for benchmark metrics."); return metrics
    
    # After potential conversion, they must be Series
    if not isinstance(sdr, pd.Series):
        logger.error(f"strategy_daily_returns is not a Series after processing. Type: {type(sdr)}")
        return metrics
    if not isinstance(bdr, pd.Series):
        logger.error(f"benchmark_daily_returns is not a Series after processing. Type: {type(bdr)}")
        return metrics

    if sdr.empty or bdr.empty:
        logger.warning("Cannot calculate benchmark metrics: strategy or benchmark returns are empty after ensuring Series type.")
        return metrics

    # Align data by date index
    aligned_df = pd.DataFrame({'strategy': sdr, 'benchmark': bdr}).dropna()

    if len(aligned_df) < 2: 
        logger.warning("Not enough overlapping data points (<2) between strategy and benchmark to calculate metrics.")
        return metrics

    strat_returns_1d = aligned_df['strategy']
    bench_returns_1d = aligned_df['benchmark']
    
    # Beta
    covariance = strat_returns_1d.cov(bench_returns_1d)
    benchmark_variance = bench_returns_1d.var()
    if benchmark_variance != 0 and not np.isnan(benchmark_variance):
        metrics['beta'] = covariance / benchmark_variance
    else:
        metrics['beta'] = np.nan 

    # Alpha
    daily_rfr = (1 + risk_free_rate)**(1/periods_per_year) - 1
    avg_strat_return_period = strat_returns_1d.mean()
    avg_bench_return_period = bench_returns_1d.mean()

    if not np.isnan(metrics['beta']):
        alpha_period = (avg_strat_return_period - daily_rfr) - metrics['beta'] * (avg_bench_return_period - daily_rfr)
        metrics['alpha'] = alpha_period * periods_per_year * 100 
    else:
        metrics['alpha'] = np.nan

    metrics['benchmark_correlation'] = strat_returns_1d.corr(bench_returns_1d)

    difference_returns = strat_returns_1d - bench_returns_1d
    tracking_error_period = difference_returns.std()
    if not np.isnan(tracking_error_period):
        metrics['tracking_error'] = tracking_error_period * np.sqrt(periods_per_year) * 100
    else:
        metrics['tracking_error'] = np.nan
        
    if tracking_error_period != 0 and not np.isnan(tracking_error_period):
        avg_excess_return_period = difference_returns.mean()
        metrics['information_ratio'] = avg_excess_return_period / tracking_error_period
    else:
        metrics['information_ratio'] = np.nan
        
    return metrics


def calculate_all_kpis(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    benchmark_daily_returns: Optional[pd.Series] = None,
    initial_capital: Optional[float] = None
) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}
    pnl_col = EXPECTED_COLUMNS['pnl']
    date_col = EXPECTED_COLUMNS['date']

    if df is None or df.empty or pnl_col not in df.columns or date_col not in df.columns:
        logger.warning("KPI calculation skipped: DataFrame is None, empty, or essential columns missing.")
        for kpi_key_loop in KPI_CONFIG.keys(): kpis[kpi_key_loop] = np.nan # Changed kpi_key to kpi_key_loop
        return kpis

    pnl_series = df[pnl_col].dropna()
    if pnl_series.empty:
        logger.warning("KPI calculation skipped: PnL series is empty after dropping NaNs.")
        for kpi_key_loop in KPI_CONFIG.keys(): kpis[kpi_key_loop] = np.nan # Changed kpi_key to kpi_key_loop
        return kpis

    kpis['total_pnl'] = pnl_series.sum()
    kpis['total_trades'] = len(pnl_series)
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    num_wins = len(wins)
    num_losses = len(losses)
    kpis['win_rate'] = (num_wins / kpis['total_trades']) * 100 if kpis['total_trades'] > 0 else 0.0
    total_gross_profit = wins.sum()
    total_gross_loss = abs(losses.sum())
    kpis['profit_factor'] = total_gross_profit / total_gross_loss if total_gross_loss > 0 else np.inf if total_gross_profit > 0 else 0.0
    kpis['avg_trade_pnl'] = pnl_series.mean() if kpis['total_trades'] > 0 else 0.0
    kpis['avg_win'] = wins.mean() if num_wins > 0 else 0.0
    kpis['avg_loss'] = abs(losses.mean()) if num_losses > 0 else 0.0
    kpis['win_loss_ratio'] = kpis['avg_win'] / kpis['avg_loss'] if kpis['avg_loss'] > 0 else np.inf if kpis['avg_win'] > 0 else 0.0

    if 'cumulative_pnl' in df.columns:
        cum_pnl_for_dd = pd.to_numeric(df['cumulative_pnl'], errors='coerce').fillna(method='ffill').fillna(0)
        if not cum_pnl_for_dd.empty:
             _, kpis['max_drawdown_abs'], kpis['max_drawdown_pct'], _ = _calculate_drawdowns(cum_pnl_for_dd)
        else:
            kpis['max_drawdown_abs'], kpis['max_drawdown_pct'] = 0.0, 0.0
    else:
        temp_cum_pnl = pnl_series.cumsum()
        if not temp_cum_pnl.empty:
            _, kpis['max_drawdown_abs'], kpis['max_drawdown_pct'], _ = _calculate_drawdowns(temp_cum_pnl)
        else:
            kpis['max_drawdown_abs'], kpis['max_drawdown_pct'] = 0.0, 0.0
    if np.isinf(kpis['max_drawdown_pct']): kpis['max_drawdown_pct'] = 100.0

    df[date_col] = pd.to_datetime(df[date_col])
    # Ensure daily_pnl is a Series and has a DatetimeIndex
    daily_pnl = df.groupby(df[date_col].dt.normalize())[pnl_col].sum()
    if not isinstance(daily_pnl.index, pd.DatetimeIndex):
        daily_pnl.index = pd.to_datetime(daily_pnl.index)
    
    strategy_daily_returns = daily_pnl # Default to absolute PnL
    if initial_capital is not None and initial_capital > 0:
        strategy_daily_returns = daily_pnl / initial_capital 
    
    # Ensure strategy_daily_returns is a Series for subsequent calculations
    if not isinstance(strategy_daily_returns, pd.Series):
        strategy_daily_returns = pd.Series(strategy_daily_returns)


    if not strategy_daily_returns.empty and len(strategy_daily_returns) > 1:
        mean_daily_return = strategy_daily_returns.mean()
        std_daily_return = strategy_daily_returns.std()
        periods_per_year = 252 
        daily_rfr = (1 + risk_free_rate)**(1/periods_per_year) - 1

        if std_daily_return != 0 and not np.isnan(std_daily_return):
            kpis['sharpe_ratio'] = (mean_daily_return - daily_rfr) / std_daily_return * np.sqrt(periods_per_year)
        else:
            kpis['sharpe_ratio'] = 0.0 if mean_daily_return <= daily_rfr else np.inf

        negative_daily_returns = strategy_daily_returns[strategy_daily_returns < daily_rfr] 
        downside_std_daily = (negative_daily_returns - daily_rfr).std() 
        if downside_std_daily != 0 and not np.isnan(downside_std_daily):
            kpis['sortino_ratio'] = (mean_daily_return - daily_rfr) / downside_std_daily * np.sqrt(periods_per_year)
        else:
            kpis['sortino_ratio'] = 0.0 if mean_daily_return <= daily_rfr else np.inf
            
        annualized_return_from_daily = mean_daily_return * periods_per_year
        if kpis.get('max_drawdown_pct', 0) > 0: # Use .get for safety
            mdd_decimal = kpis['max_drawdown_pct'] / 100.0
            kpis['calmar_ratio'] = annualized_return_from_daily / mdd_decimal if mdd_decimal > 0 else \
                                   (np.inf if annualized_return_from_daily > 0 else 0.0)
        else: 
            kpis['calmar_ratio'] = np.inf if annualized_return_from_daily > 0 else 0.0
            
    else: 
        kpis['sharpe_ratio'] = 0.0
        kpis['sortino_ratio'] = 0.0
        kpis['calmar_ratio'] = 0.0

    if not daily_pnl.empty:
        losses_for_var_daily = -daily_pnl[daily_pnl < 0] 
        if not losses_for_var_daily.empty:
            kpis['var_95_loss'] = losses_for_var_daily.quantile(0.95)
            kpis['cvar_95_loss'] = losses_for_var_daily[losses_for_var_daily >= kpis.get('var_95_loss', 0)].mean() # Use .get
            kpis['var_99_loss'] = losses_for_var_daily.quantile(0.99)
            kpis['cvar_99_loss'] = losses_for_var_daily[losses_for_var_daily >= kpis.get('var_99_loss', 0)].mean() # Use .get
        else: 
            kpis['var_95_loss'] = kpis['cvar_95_loss'] = kpis['var_99_loss'] = kpis['cvar_99_loss'] = 0.0
    else: 
        kpis['var_95_loss'] = kpis['cvar_95_loss'] = kpis['var_99_loss'] = kpis['cvar_99_loss'] = 0.0

    kpis['pnl_skewness'] = pnl_series.skew() if kpis['total_trades'] > 2 else 0.0
    kpis['pnl_kurtosis'] = pnl_series.kurtosis() if kpis['total_trades'] > 3 else 0.0

    kpis['max_win_streak'], kpis['max_loss_streak'] = _calculate_streaks(pnl_series)

    kpis['trading_days'] = daily_pnl.count() 
    kpis['avg_daily_pnl'] = daily_pnl.mean() if not daily_pnl.empty else 0.0
    kpis['risk_free_rate_used'] = risk_free_rate * 100

    if benchmark_daily_returns is not None and not benchmark_daily_returns.empty:
        if initial_capital is None:
            logger.warning("Calculating benchmark metrics (Alpha, Beta) using absolute daily PnL for strategy as initial_capital was not provided. Results may be hard to interpret against benchmark's percentage returns.")
        
        # Ensure benchmark_daily_returns is a Series
        temp_benchmark_returns = benchmark_daily_returns
        if isinstance(temp_benchmark_returns, pd.DataFrame):
            temp_benchmark_returns = temp_benchmark_returns.squeeze()
        if not isinstance(temp_benchmark_returns, pd.Series):
            logger.error("Benchmark returns provided to calculate_all_kpis is not a Series. Skipping benchmark metrics.")
            temp_benchmark_returns = pd.Series(dtype=float) # Empty series to avoid error

        benchmark_kpis = calculate_benchmark_metrics(
            strategy_daily_returns, 
            temp_benchmark_returns,
            risk_free_rate
        )
        kpis.update(benchmark_kpis)
        
        if not strategy_daily_returns.empty and not temp_benchmark_returns.empty:
            # Align benchmark returns to strategy returns' index for total return calculation
            aligned_benchmark_returns_for_total = temp_benchmark_returns.reindex(strategy_daily_returns.index).dropna()
            if not aligned_benchmark_returns_for_total.empty:
                kpis['benchmark_total_return'] = ( (1 + aligned_benchmark_returns_for_total).cumprod().iloc[-1] - 1) * 100
            else:
                kpis['benchmark_total_return'] = np.nan
        else:
            kpis['benchmark_total_return'] = np.nan
    else: 
        for bkpi in ["benchmark_total_return", "alpha", "beta", "benchmark_correlation", "tracking_error", "information_ratio"]:
            kpis[bkpi] = np.nan

    for key, value in kpis.items():
        if pd.isna(value): kpis[key] = 0.0 
        elif np.isinf(value):
            if key in ["profit_factor", "win_loss_ratio", "sortino_ratio", "sharpe_ratio", "calmar_ratio"] and value > 0:
                pass 
            else:
                kpis[key] = 0.0 

    logger.info(f"Calculated KPIs (including benchmark if provided).")
    return kpis


def get_kpi_interpretation(kpi_key: str, value: float) -> Tuple[str, str]:
    if kpi_key not in KPI_CONFIG or pd.isna(value):
        return "N/A", "KPI not found or value is NaN"
    config = KPI_CONFIG[kpi_key]
    thresholds = config.get("thresholds", [])
    unit = config.get("unit", "")
    interpretation = "N/A"; threshold_desc = "N/A"
    for label, min_val, max_val_exclusive in thresholds:
        if min_val <= value < max_val_exclusive:
            interpretation = label
            if min_val == float('-inf'): threshold_desc = f"< {max_val_exclusive:,.1f}{unit}"
            elif max_val_exclusive == float('inf'): threshold_desc = f">= {min_val:,.1f}{unit}"
            else: threshold_desc = f"{min_val:,.1f} - {max_val_exclusive:,.1f}{unit}"
            break
    if interpretation == "N/A" and thresholds:
        last_label, last_min, last_max = thresholds[-1]
        if value >= last_min and last_max == float('inf'):
            interpretation = last_label; threshold_desc = f">= {last_min:,.1f}{unit}"
        elif value < thresholds[0][1] and thresholds[0][1] != float('-inf'): 
             interpretation = thresholds[0][0]; threshold_desc = f"< {thresholds[0][1]:,.1f}{unit}"
    return interpretation, f"Val: {value:,.2f}{unit} (Thr: {threshold_desc})" if interpretation != "N/A" else f"Val: {value:,.2f}{unit}"

def get_kpi_color(kpi_key: str, value: float) -> str:
    if kpi_key not in KPI_CONFIG or pd.isna(value) or np.isinf(value):
        return COLORS.get("gray", "#808080")
    config = KPI_CONFIG[kpi_key]
    color_logic = config.get("color_logic")
    if color_logic:
        return color_logic(value, 0) 
    return COLORS.get("gray", "#808080")

