"""
calculations.py

Implements calculations for all specified Key Performance Indicators (KPIs)
and provides functions for their qualitative interpretation and color-coding
based on thresholds defined in config.py.
Includes benchmark-relative metrics like Alpha and Beta.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple, List, Optional
import logging

from config import RISK_FREE_RATE, KPI_CONFIG, COLORS, EXPECTED_COLUMNS

logger = logging.getLogger(EXPECTED_COLUMNS.get("APP_TITLE", "TradingDashboard_Calc")) # Use APP_TITLE from config if available
if not logger.handlers: # Ensure basic handler if not configured by main app
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def _calculate_returns(pnl_series: pd.Series, initial_capital: Optional[float] = None) -> pd.Series:
    """
    Calculates returns. If initial_capital is provided, calculates percentage returns.
    Otherwise, PnL values are treated as absolute returns for risk-adjusted metrics.
    """
    if pnl_series.empty:
        return pd.Series(dtype=float)
    if initial_capital and initial_capital != 0:
        return pnl_series / initial_capital
    return pnl_series # Treat PnL as returns directly

def _calculate_drawdowns(cumulative_pnl: pd.Series) -> Tuple[pd.Series, float, float, pd.Series]:
    """
    Calculates drawdown series, max drawdown value, and max drawdown percentage.
    """
    if cumulative_pnl.empty:
        return pd.Series(dtype=float), 0.0, 0.0, pd.Series(dtype=float)

    high_water_mark = cumulative_pnl.cummax()
    drawdown_series = high_water_mark - cumulative_pnl
    max_drawdown_abs = drawdown_series.max() if not drawdown_series.empty else 0.0
    
    drawdown_pct_series = (drawdown_series / high_water_mark.replace(0, np.nan)).fillna(0) * 100
    max_drawdown_pct = drawdown_pct_series.max() if not drawdown_pct_series.empty else 0.0
    
    # Refined handling for max_drawdown_pct if HWM starts at 0 and PnL goes negative
    if high_water_mark.iloc[0] == 0 and cumulative_pnl.min() < 0:
        # If the first peak is 0, any loss means 100% drawdown from that peak.
        # However, the standard calculation (Peak - Trough) / Peak might yield inf or NaN.
        # A practical approach: if initial capital was 0, and PnL is negative,
        # the concept of percentage drawdown is less meaningful without a defined capital base.
        # The current fillna(0) for drawdown_pct_series handles 0/0.
        # If max_drawdown_abs > 0 and the peak at which it occurred was 0, it's problematic.
        # For simplicity, if max_drawdown_abs > 0 and initial HWM was 0, we might cap pct at 100 or report as N/A.
        # The current logic with replace(0, np.nan) in HWM for division should lead to NaNs that become 0.
        # This implies 0% drawdown if the peak was 0, which might be misleading if losses occurred.
        # Let's ensure if max_drawdown_abs is positive, max_drawdown_pct is also positive if applicable.
        if max_drawdown_abs > 0 and max_drawdown_pct == 0 and high_water_mark.abs().sum() == 0:
             max_drawdown_pct = 100.0 # If absolute drawdown exists but all peaks were zero (e.g. started at 0, lost money)

    return drawdown_series, max_drawdown_abs, max_drawdown_pct, drawdown_pct_series


def _calculate_streaks(pnl_series: pd.Series) -> Tuple[int, int]:
    """Calculates maximum win and loss streaks."""
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
    strategy_daily_returns: pd.Series,
    benchmark_daily_returns: pd.Series,
    risk_free_rate: float, # Annual RFR
    periods_per_year: int = 252 # Common for daily data
) -> Dict[str, Any]:
    """
    Calculates Alpha, Beta, Correlation, Tracking Error, and Information Ratio.
    Assumes daily returns are provided.
    """
    metrics: Dict[str, Any] = {
        "alpha": np.nan, "beta": np.nan, "benchmark_correlation": np.nan,
        "tracking_error": np.nan, "information_ratio": np.nan
    }
    if strategy_daily_returns.empty or benchmark_daily_returns.empty:
        logger.warning("Cannot calculate benchmark metrics: strategy or benchmark returns are empty.")
        return metrics

    # Align data by date index
    aligned_df = pd.DataFrame({
        'strategy': strategy_daily_returns,
        'benchmark': benchmark_daily_returns
    }).dropna()

    if len(aligned_df) < 2: # Need at least 2 data points for variance/covariance
        logger.warning("Not enough overlapping data points between strategy and benchmark to calculate metrics.")
        return metrics

    strat_returns = aligned_df['strategy']
    bench_returns = aligned_df['benchmark']
    
    # Beta
    # Beta = Cov(Strategy Returns, Benchmark Returns) / Var(Benchmark Returns)
    covariance = strat_returns.cov(bench_returns)
    benchmark_variance = bench_returns.var()
    if benchmark_variance != 0 and not np.isnan(benchmark_variance):
        metrics['beta'] = covariance / benchmark_variance
    else:
        metrics['beta'] = np.nan # or 0 if benchmark has no variance

    # Alpha
    # Alpha = (Avg Strategy Return - RFR_daily) - Beta * (Avg Benchmark Return - RFR_daily)
    # Annualize Alpha at the end
    daily_rfr = (1 + risk_free_rate)**(1/periods_per_year) - 1
    
    avg_strat_return_period = strat_returns.mean()
    avg_bench_return_period = bench_returns.mean()

    if not np.isnan(metrics['beta']):
        alpha_period = (avg_strat_return_period - daily_rfr) - metrics['beta'] * (avg_bench_return_period - daily_rfr)
        metrics['alpha'] = alpha_period * periods_per_year * 100 # Annualized and in percentage
    else:
        metrics['alpha'] = np.nan

    # Correlation
    metrics['benchmark_correlation'] = strat_returns.corr(bench_returns)

    # Tracking Error
    # Std Dev of (Strategy Returns - Benchmark Returns), annualized
    difference_returns = strat_returns - bench_returns
    tracking_error_period = difference_returns.std()
    if not np.isnan(tracking_error_period):
        metrics['tracking_error'] = tracking_error_period * np.sqrt(periods_per_year) * 100 # Annualized and in percentage
    else:
        metrics['tracking_error'] = np.nan
        
    # Information Ratio
    # (Avg Strategy Return - Avg Benchmark Return) / Tracking Error (using period returns for consistency)
    # Or, more commonly, (Annualized Alpha) / (Annualized Tracking Error)
    # Let's use (Avg (Strat - Bench) ) / StdDev(Strat - Bench) - not typically annualized itself but components are
    if tracking_error_period != 0 and not np.isnan(tracking_error_period):
        # Using excess return over benchmark per period
        avg_excess_return_period = difference_returns.mean()
        metrics['information_ratio'] = avg_excess_return_period / tracking_error_period
        # Some definitions annualize this: metrics['information_ratio'] *= np.sqrt(periods_per_year)
        # For now, using the non-annualized version based on period returns.
    else:
        metrics['information_ratio'] = np.nan
        
    return metrics


def calculate_all_kpis(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    benchmark_daily_returns: Optional[pd.Series] = None, # Daily returns of the benchmark
    initial_capital: Optional[float] = None # For calculating strategy returns if PnL is absolute
) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}
    pnl_col = EXPECTED_COLUMNS['pnl']
    date_col = EXPECTED_COLUMNS['date'] # Assuming this is the trade execution timestamp

    if df is None or df.empty or pnl_col not in df.columns or date_col not in df.columns:
        logger.warning("KPI calculation skipped: DataFrame is None, empty, or essential columns missing.")
        for kpi_key in KPI_CONFIG.keys(): kpis[kpi_key] = np.nan
        return kpis

    pnl_series = df[pnl_col].dropna()
    if pnl_series.empty:
        logger.warning("KPI calculation skipped: PnL series is empty after dropping NaNs.")
        for kpi_key in KPI_CONFIG.keys(): kpis[kpi_key] = np.nan
        return kpis

    # --- Basic Metrics ---
    kpis['total_pnl'] = pnl_series.sum()
    kpis['total_trades'] = len(pnl_series)
    # ... (rest of basic metric calculations as before) ...
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


    # --- Drawdown ---
    if 'cumulative_pnl' in df.columns:
        # Ensure cumulative_pnl is numeric and has no NaNs for drawdown calculation
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

    # --- Risk-Adjusted Ratios (Sharpe, Sortino, Calmar) ---
    # These require periodic returns. We'll use daily returns for consistency.
    # Group PnL by date to get daily PnL
    df[date_col] = pd.to_datetime(df[date_col])
    daily_pnl = df.groupby(df[date_col].dt.date)[pnl_col].sum()
    
    if initial_capital is not None and initial_capital > 0:
        # If initial capital is provided, calculate daily % returns
        # This is a simplification; true portfolio returns evolve with capital.
        # For a more accurate daily return series based on a fixed initial capital:
        # One approach: daily_returns = daily_pnl / initial_capital
        # However, for ratios, it's common to use daily PnL directly if capital base is not dynamic.
        # Let's assume daily_pnl itself represents the 'return' for ratio calculations if not using %
        strategy_daily_returns = daily_pnl / initial_capital # Percentage returns
    else:
        # If no initial capital, use absolute daily PnL as returns for ratio calculations.
        # This is common for strategies where capital base is not fixed or easily defined per period.
        strategy_daily_returns = daily_pnl # Absolute returns

    if not strategy_daily_returns.empty and len(strategy_daily_returns) > 1:
        mean_daily_return = strategy_daily_returns.mean()
        std_daily_return = strategy_daily_returns.std()
        periods_per_year = 252 # Assuming daily data

        # Daily risk-free rate
        daily_rfr = (1 + risk_free_rate)**(1/periods_per_year) - 1

        # Sharpe Ratio
        if std_daily_return != 0 and not np.isnan(std_daily_return):
            kpis['sharpe_ratio'] = (mean_daily_return - daily_rfr) / std_daily_return * np.sqrt(periods_per_year)
        else:
            kpis['sharpe_ratio'] = 0.0 if mean_daily_return <= daily_rfr else np.inf

        # Sortino Ratio
        negative_daily_returns = strategy_daily_returns[strategy_daily_returns < daily_rfr] # Returns below RFR
        downside_std_daily = (negative_daily_returns - daily_rfr).std() # Std dev of returns below target
        if downside_std_daily != 0 and not np.isnan(downside_std_daily):
            kpis['sortino_ratio'] = (mean_daily_return - daily_rfr) / downside_std_daily * np.sqrt(periods_per_year)
        else:
            kpis['sortino_ratio'] = 0.0 if mean_daily_return <= daily_rfr else np.inf
            
        # Calmar Ratio: Annualized Return / Max Drawdown %
        # Annualized return:
        annualized_return_from_daily = mean_daily_return * periods_per_year
        if kpis['max_drawdown_pct'] > 0:
            # Ensure max_drawdown_pct is in decimal form (e.g., 20% -> 0.20)
            mdd_decimal = kpis['max_drawdown_pct'] / 100.0
            kpis['calmar_ratio'] = annualized_return_from_daily / mdd_decimal if mdd_decimal > 0 else \
                                   (np.inf if annualized_return_from_daily > 0 else 0.0)
        else: # No drawdown
            kpis['calmar_ratio'] = np.inf if annualized_return_from_daily > 0 else 0.0
            
    else: # Not enough data for std dev based ratios
        kpis['sharpe_ratio'] = 0.0
        kpis['sortino_ratio'] = 0.0
        kpis['calmar_ratio'] = 0.0

    # --- VaR and CVaR (using daily PnL) ---
    if not daily_pnl.empty:
        losses_for_var_daily = -daily_pnl[daily_pnl < 0] # Positive values of daily losses
        if not losses_for_var_daily.empty:
            kpis['var_95_loss'] = losses_for_var_daily.quantile(0.95)
            kpis['cvar_95_loss'] = losses_for_var_daily[losses_for_var_daily >= kpis['var_95_loss']].mean()
            kpis['var_99_loss'] = losses_for_var_daily.quantile(0.99)
            kpis['cvar_99_loss'] = losses_for_var_daily[losses_for_var_daily >= kpis['var_99_loss']].mean()
        else: # No daily losses
            kpis['var_95_loss'] = kpis['cvar_95_loss'] = kpis['var_99_loss'] = kpis['cvar_99_loss'] = 0.0
    else: # No daily PnL data
        kpis['var_95_loss'] = kpis['cvar_95_loss'] = kpis['var_99_loss'] = kpis['cvar_99_loss'] = 0.0

    # --- Distributional Properties (using per-trade PnL) ---
    kpis['pnl_skewness'] = pnl_series.skew() if kpis['total_trades'] > 2 else 0.0
    kpis['pnl_kurtosis'] = pnl_series.kurtosis() if kpis['total_trades'] > 3 else 0.0

    # --- Streaks (using per-trade PnL) ---
    kpis['max_win_streak'], kpis['max_loss_streak'] = _calculate_streaks(pnl_series)

    # --- Other Metrics ---
    kpis['trading_days'] = daily_pnl.count() # Number of days with trades
    kpis['avg_daily_pnl'] = daily_pnl.mean() if not daily_pnl.empty else 0.0
    kpis['risk_free_rate_used'] = risk_free_rate * 100

    # --- Benchmark Metrics ---
    if benchmark_daily_returns is not None and not benchmark_daily_returns.empty:
        # Ensure strategy_daily_returns is percentage if benchmark is percentage
        # If initial_capital was not given, strategy_daily_returns are absolute.
        # For Alpha/Beta, consistent return types (percentage) are preferred.
        # This section assumes `strategy_daily_returns` are appropriate for benchmark comparison.
        # If `initial_capital` was not provided, `strategy_daily_returns` are absolute daily PnLs.
        # This might not be ideal for direct Alpha/Beta calculation against a benchmark's % returns.
        # For now, we proceed, but highlight this dependency.
        if initial_capital is None:
            logger.warning("Calculating benchmark metrics (Alpha, Beta) using absolute daily PnL for strategy as initial_capital was not provided. Results may be hard to interpret against benchmark's percentage returns.")

        benchmark_kpis = calculate_benchmark_metrics(
            strategy_daily_returns, # This should ideally be % returns
            benchmark_daily_returns,
            risk_free_rate
        )
        kpis.update(benchmark_kpis)
        
        # Calculate benchmark total return over the period of strategy_daily_returns
        if not strategy_daily_returns.empty and not benchmark_daily_returns.empty:
            aligned_benchmark_returns = benchmark_daily_returns.reindex(strategy_daily_returns.index).dropna()
            if not aligned_benchmark_returns.empty:
                kpis['benchmark_total_return'] = ( (1 + aligned_benchmark_returns).cumprod().iloc[-1] - 1) * 100 # As percentage
            else:
                kpis['benchmark_total_return'] = np.nan
    else: # No benchmark data
        kpis['benchmark_total_return'] = np.nan
        kpis['alpha'] = np.nan
        kpis['beta'] = np.nan
        kpis['benchmark_correlation'] = np.nan
        kpis['tracking_error'] = np.nan
        kpis['information_ratio'] = np.nan


    # Final cleanup of NaNs and Infs
    for key, value in kpis.items():
        if pd.isna(value): kpis[key] = 0.0 # Default NaN to 0, review if specific KPIs need other defaults
        elif np.isinf(value):
            # For ratios like Profit Factor, Inf can be valid if losses are zero and profits positive.
            # For others, it might indicate an issue or should be capped.
            if key in ["profit_factor", "win_loss_ratio", "sortino_ratio", "sharpe_ratio", "calmar_ratio"] and value > 0:
                pass # Positive infinity is okay for these if numerator positive and denominator zero
            else:
                kpis[key] = 0.0 # Default for problematic infinities or negative infinities

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
        elif value < thresholds[0][1] and thresholds[0][1] != float('-inf'): # Check if below first defined range start
             interpretation = thresholds[0][0]; threshold_desc = f"< {thresholds[0][1]:,.1f}{unit}" # Assuming first label is for values below its min
    return interpretation, f"Val: {value:,.2f}{unit} (Thr: {threshold_desc})" if interpretation != "N/A" else f"Val: {value:,.2f}{unit}"

def get_kpi_color(kpi_key: str, value: float) -> str:
    if kpi_key not in KPI_CONFIG or pd.isna(value) or np.isinf(value):
        return COLORS.get("gray", "#808080")
    config = KPI_CONFIG[kpi_key]
    color_logic = config.get("color_logic")
    if color_logic:
        # The lambda in config should handle its own logic.
        # The 't' (threshold) parameter in lambda is not strictly used by current color_logics,
        # but kept for potential future use if color depends on specific thresholds.
        return color_logic(value, 0) # Pass dummy threshold
    return COLORS.get("gray", "#808080")

if __name__ == '__main__':
    logger.info("--- Testing KPI Calculations (with benchmark stubs) ---")
    sample_dates = pd.to_datetime([
        '2023-01-01 10:00:00', '2023-01-01 12:00:00', '2023-01-02 09:30:00',
        '2023-01-02 15:00:00', '2023-01-03 11:00:00', '2023-01-04 10:00:00',
        '2023-01-04 14:00:00', '2023-01-05 09:00:00', '2023-01-05 16:00:00',
        '2023-01-06 10:00:00'
    ])
    sample_data = {
        EXPECTED_COLUMNS['date']: sample_dates,
        EXPECTED_COLUMNS['pnl']: [10, -5, 20, 5, -15, 8, -3, 12, 6, -7],
    }
    test_df = pd.DataFrame(sample_data)
    test_df['cumulative_pnl'] = test_df[EXPECTED_COLUMNS['pnl']].cumsum()

    # Mock benchmark daily returns (ensure index aligns with test_df daily dates)
    unique_trade_dates = pd.to_datetime(test_df[EXPECTED_COLUMNS['date']].dt.date.unique()).sort_values()
    mock_bench_returns = pd.Series(np.random.randn(len(unique_trade_dates)) * 0.005 + 0.0001, index=unique_trade_dates) # Small positive drift
    
    kpis_with_bench = calculate_all_kpis(test_df.copy(), risk_free_rate=0.02, benchmark_daily_returns=mock_bench_returns, initial_capital=100000)
    logger.info("\n--- KPIs with Benchmark ---")
    for k, v in kpis_with_bench.items():
        if k in KPI_CONFIG:
            interp, desc = get_kpi_interpretation(k,v)
            logger.info(f"{KPI_CONFIG[k]['name']}: {v:.4f} ({KPI_CONFIG[k]['unit']}) - {interp} ({desc})")
        else:
            logger.info(f"{k.replace('_',' ').title()}: {v:.4f}")

    kpis_no_bench = calculate_all_kpis(test_df.copy(), risk_free_rate=0.02, initial_capital=100000)
    logger.info("\n--- KPIs without Benchmark ---")
    for k, v in kpis_no_bench.items():
        if k in KPI_CONFIG and k not in ['alpha', 'beta', 'benchmark_correlation', 'benchmark_total_return', 'tracking_error', 'information_ratio']: # Exclude benchmark specific
            interp, desc = get_kpi_interpretation(k,v)
            logger.info(f"{KPI_CONFIG[k]['name']}: {v:.4f} ({KPI_CONFIG[k]['unit']}) - {interp} ({desc})")
        elif k not in ['alpha', 'beta', 'benchmark_correlation', 'benchmark_total_return', 'tracking_error', 'information_ratio']:
             logger.info(f"{k.replace('_',' ').title()}: {v:.4f}")

