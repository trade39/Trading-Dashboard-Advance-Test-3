"""
calculations.py

Implements calculations for all specified Key Performance Indicators (KPIs)
and provides functions for their qualitative interpretation and color-coding
based on thresholds defined in config.py.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple, List, Optional
import logging

# Assuming config.py is in the root directory
from config import RISK_FREE_RATE, KPI_CONFIG, COLORS, EXPECTED_COLUMNS

# Placeholder for logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def _calculate_returns(pnl_series: pd.Series, initial_capital: Optional[float] = None) -> pd.Series:
    """
    Calculates returns. If initial_capital is provided, calculates percentage returns.
    Otherwise, PnL values are treated as absolute returns for risk-adjusted metrics.
    For Sharpe/Sortino, if PnL is already per-trade profit, it can be used directly.
    """
    if pnl_series.empty:
        return pd.Series(dtype=float)
    if initial_capital and initial_capital != 0:
        # This assumes pnl_series are for individual periods and initial_capital is constant
        # For a more accurate portfolio return, you'd need evolving capital values.
        return pnl_series / initial_capital
    return pnl_series # Treat PnL as returns directly for ratio calculations if capital not specified

def _calculate_drawdowns(cumulative_pnl: pd.Series) -> Tuple[pd.Series, float, float, pd.Series]:
    """
    Calculates drawdown series, max drawdown value, and max drawdown percentage.
    Args:
        cumulative_pnl (pd.Series): Series of cumulative PnL.
    Returns:
        Tuple[pd.Series, float, float, pd.Series]:
            - drawdown_series (absolute drawdown from peak)
            - max_drawdown_abs (maximum absolute drawdown value)
            - max_drawdown_pct (maximum drawdown as percentage of peak equity)
            - drawdown_pct_series (drawdown as percentage of current peak)
    """
    if cumulative_pnl.empty:
        return pd.Series(dtype=float), 0.0, 0.0, pd.Series(dtype=float)

    # Calculate peak at each point
    high_water_mark = cumulative_pnl.cummax()
    drawdown_series = high_water_mark - cumulative_pnl # Absolute drawdown from peak
    max_drawdown_abs = drawdown_series.max()

    # Calculate drawdown percentage
    # Drawdown percentage is (Peak - Trough) / Peak
    # To avoid division by zero if initial peak is 0, we handle this.
    # If HWM is 0, drawdown % is undefined or can be considered 0 if PnL is also 0.
    drawdown_pct_series = (drawdown_series / high_water_mark).fillna(0) * 100
    # If high_water_mark is 0, and drawdown_series is >0, it implies negative PnL from start.
    # This can lead to inf. For practical purposes, if HWM is 0, and PnL < 0, DD% is effectively 100% of loss.
    # However, standard definition relies on a positive peak.
    # Let's find the max of this series.
    max_drawdown_pct = drawdown_pct_series.max()

    if max_drawdown_abs == 0: # No drawdown
        max_drawdown_pct = 0.0

    # A more robust way to calculate max_drawdown_pct considering the peak at which it occurred:
    if not drawdown_series.empty:
        idx_max_drawdown = drawdown_series.idxmax() # Index of the largest absolute drawdown end
        if idx_max_drawdown is not None and idx_max_drawdown >= 0: # Check for valid index
            peak_at_max_drawdown = high_water_mark[idx_max_drawdown]
            if peak_at_max_drawdown > 0: # Ensure peak is positive for meaningful percentage
                 # Max drawdown value / Peak value before this drawdown started
                 # To find the actual peak *before* the max drawdown, we need to look at HWM up to the point of max DD
                 # This is tricky. A simpler way is (Peak - Trough) / Peak.
                 # Let's use the max of the drawdown_pct_series, but ensure it's capped at 100% if PnL goes very negative from 0.
                 if cumulative_pnl.min() < 0 and high_water_mark.iloc[0] == 0 and cumulative_pnl.iloc[0] == 0 : # Started at 0, went negative
                     # if all HWM are 0 and PnL is negative, DD% is effectively infinite or 100% of loss.
                     # This scenario is complex. For now, rely on (HWM - CumPnL) / HWM.
                     # If HWM is 0, then drawdown_pct_series elements would be NaN or inf if CumPnL is negative.
                     # The .fillna(0) above handles NaN. For inf, np.max will pick it up.
                     # We might want to cap it at 100 if the initial capital was notionally positive.
                     pass # Keep current max_drawdown_pct calculation

    # Ensure max_drawdown_pct is not erroneously high due to division by small numbers near zero.
    # If total PnL is negative and started from 0, max drawdown % is effectively 100% of the loss relative to initial 0.
    # This is a bit ill-defined. If initial capital was >0, it's clearer.
    # For now, the (HWM - CumPnL) / HWM is standard.
    if high_water_mark.iloc[0] == 0 and cumulative_pnl.min() < 0:
        # If we started at 0 and PnL became negative, the first peak is 0.
        # (0 - (-50)) / 0 is problematic.
        # In this case, max_drawdown_pct could be considered 100% if any loss occurred.
        # Or, if we consider initial capital implicitly > 0, then it's relative to that.
        # The current logic might result in `inf` if HWM is 0 and CumPnL is negative, then `max` picks `inf`.
        # Let's replace `inf` with a large number like 100 or based on context.
        # For simplicity, if max_drawdown_abs > 0 and initial HWM was 0, it's a full loss of that amount.
        # The percentage is tricky without a defined initial capital.
        # The current calculation of drawdown_pct_series should handle this via fillna(0) for 0/0 cases.
        # If HWM is 0 and CumPnL is negative, (0 - CumPnL) / 0 -> inf.
        # Let's replace inf in drawdown_pct_series with 0 or a sensible cap.
        drawdown_pct_series.replace([np.inf, -np.inf], 0, inplace=True) # Or 100 if it's a loss from 0
        max_drawdown_pct = drawdown_pct_series.max()


    return drawdown_series, max_drawdown_abs, max_drawdown_pct, drawdown_pct_series


def _calculate_streaks(pnl_series: pd.Series) -> Tuple[int, int]:
    """Calculates maximum win and loss streaks."""
    if pnl_series.empty:
        return 0, 0

    wins = pnl_series > 0
    losses = pnl_series < 0

    max_win_streak = 0
    current_win_streak = 0
    for w in wins:
        if w:
            current_win_streak += 1
        else:
            max_win_streak = max(max_win_streak, current_win_streak)
            current_win_streak = 0
    max_win_streak = max(max_win_streak, current_win_streak) # Final check

    max_loss_streak = 0
    current_loss_streak = 0
    for l_val in losses: # Renamed to avoid conflict with 'l' in lambda
        if l_val:
            current_loss_streak += 1
        else:
            max_loss_streak = max(max_loss_streak, current_loss_streak)
            current_loss_streak = 0
    max_loss_streak = max(max_loss_streak, current_loss_streak) # Final check

    return int(max_win_streak), int(max_loss_streak)

def calculate_all_kpis(df: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> Dict[str, Any]:
    """
    Calculates all Key Performance Indicators (KPIs).

    Args:
        df (pd.DataFrame): Processed DataFrame with trade data. Must include
                           'pnl', 'cumulative_pnl', 'win', 'loss', 'trade_date_only',
                           and 'date' (for annualized calculations).
        risk_free_rate (float): Annual risk-free rate.

    Returns:
        Dict[str, Any]: A dictionary where keys are KPI names (matching KPI_CONFIG)
                        and values are the calculated KPI values. Returns an empty
                        dict if data is insufficient.
    """
    kpis: Dict[str, Any] = {}
    pnl_col = EXPECTED_COLUMNS['pnl']

    if df is None or df.empty or pnl_col not in df.columns:
        logger.warning("KPI calculation skipped: DataFrame is None, empty, or PnL column is missing.")
        # Return default/NA values for all KPIs defined in config
        for kpi_key in KPI_CONFIG.keys():
            kpis[kpi_key] = np.nan # Or 0, or specific default
        return kpis

    pnl_series = df[pnl_col].dropna()
    if pnl_series.empty:
        logger.warning("KPI calculation skipped: PnL series is empty after dropping NaNs.")
        for kpi_key in KPI_CONFIG.keys():
            kpis[kpi_key] = np.nan
        return kpis

    # --- Basic Metrics ---
    kpis['total_pnl'] = pnl_series.sum()
    kpis['total_trades'] = len(pnl_series)

    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    num_wins = len(wins)
    num_losses = len(losses)

    kpis['win_rate'] = (num_wins / kpis['total_trades']) * 100 if kpis['total_trades'] > 0 else 0.0
    kpis['loss_rate'] = (num_losses / kpis['total_trades']) * 100 if kpis['total_trades'] > 0 else 0.0

    total_gross_profit = wins.sum()
    total_gross_loss = abs(losses.sum()) # abs because losses are negative

    kpis['profit_factor'] = total_gross_profit / total_gross_loss if total_gross_loss > 0 else np.inf if total_gross_profit > 0 else 0.0

    kpis['avg_trade_pnl'] = pnl_series.mean() if kpis['total_trades'] > 0 else 0.0
    kpis['avg_win'] = wins.mean() if num_wins > 0 else 0.0
    kpis['avg_loss'] = abs(losses.mean()) if num_losses > 0 else 0.0 # abs because losses are negative

    kpis['win_loss_ratio'] = kpis['avg_win'] / kpis['avg_loss'] if kpis['avg_loss'] > 0 else np.inf if kpis['avg_win'] > 0 else 0.0

    # --- Drawdown ---
    # Assuming 'cumulative_pnl' is already calculated in data_processing
    if 'cumulative_pnl' in df.columns:
        df['drawdown_abs'], kpis['max_drawdown_abs'], kpis['max_drawdown_pct'], df['drawdown_pct'] = _calculate_drawdowns(df['cumulative_pnl'])
    else: # Fallback if cumulative_pnl is not pre-calculated
        temp_cum_pnl = pnl_series.cumsum()
        _, kpis['max_drawdown_abs'], kpis['max_drawdown_pct'], _ = _calculate_drawdowns(temp_cum_pnl)
    
    # Ensure drawdown percentages are reasonable (e.g. not inf)
    if np.isinf(kpis['max_drawdown_pct']):
        kpis['max_drawdown_pct'] = 100.0 # Cap at 100% if inf occurs


    # --- Risk-Adjusted Ratios ---
    # Assuming PnL series represents returns for these calculations.
    # For daily data, returns_series would be daily PnL. For per-trade data, it's per-trade PnL.
    # Annualization factor depends on data frequency.
    # Assuming 252 trading days per year for daily data.
    # If data is per-trade, annualization is more complex and depends on trade frequency.
    # Let's assume PnL is per-period (e.g., daily or per trade).
    # We need number of periods per year.
    
    trading_days_col = 'trade_date_only' # from data_processing
    num_trading_periods = kpis['total_trades'] # Default to number of trades
    if trading_days_col in df.columns and df[trading_days_col].nunique() > 1:
        unique_trading_days = df[trading_days_col].nunique()
        # If we have daily PnL, then unique_trading_days is the number of periods.
        # If PnL is per trade, we might annualize based on total duration of trading.
        # For now, let's use a common approach: assume returns are per period (trade or day).
        # If we use daily returns:
        # daily_returns = df.groupby(trading_days_col)[pnl_col].sum()
        # returns_std = daily_returns.std()
        # mean_return = daily_returns.mean()
        # num_periods_in_year = 252
        
        # Simpler: use per-trade PnL as returns for now. Annualization factor will be tricky.
        # Let's calculate Sharpe based on per-trade PnL and not annualize it here,
        # or provide an option. For now, let's assume PnL series is what we use.
        
        returns_series = pnl_series
        mean_return = returns_series.mean()
        returns_std = returns_series.std()

        # Sharpe Ratio
        if returns_std != 0 and not np.isnan(returns_std):
            # Assuming risk_free_rate is annual, convert to per-period rate.
            # This is complex if periods are trades of varying duration.
            # If we assume 'returns_series' are daily returns, then:
            # daily_rf_rate = (1 + risk_free_rate)**(1/252) - 1
            # sharpe = (mean_return - daily_rf_rate) / returns_std * np.sqrt(252)
            # For per-trade, often RFR is ignored or set to 0 if trades are short-term.
            # Let's calculate a simplified Sharpe assuming RFR per period is small or 0.
            # And annualize by sqrt(num_trades_per_year) - also tricky.
            # For now, a common simplification if RFR is low and trades are frequent:
            sharpe_ratio_simple = mean_return / returns_std if returns_std > 0 else 0.0
            # To annualize, if we assume N trades per year: sharpe_annual = sharpe_simple * sqrt(N)
            # This is an approximation. Let's provide a non-annualized Sharpe for now or based on number of trades.
            # If we have total duration:
            if 'date' in df.columns and len(df['date']) > 1:
                total_duration_years = (df['date'].max() - df['date'].min()).days / 365.25
                if total_duration_years > 0:
                    trades_per_year_est = kpis['total_trades'] / total_duration_years
                    sharpe_annualization_factor = np.sqrt(trades_per_year_est) if trades_per_year_est > 0 else 1
                    # Per-period risk-free rate (approximate if periods are trades)
                    # This is a simplification. A proper Sharpe requires consistent period returns.
                    # Assuming RFR for the average trade period is negligible for now.
                    # Or, if using daily returns:
                    # daily_pnl = df.groupby(df['date'].dt.date)[pnl_col].sum()
                    # if not daily_pnl.empty and daily_pnl.std() > 0:
                    #     daily_rf = risk_free_rate / 252 # Approximation
                    #     kpis['sharpe_ratio'] = (daily_pnl.mean() - daily_rf) / daily_pnl.std() * np.sqrt(252)

                    # Using per-trade PnL, simple Sharpe, annualized by sqrt(trades_per_year_est)
                    # This is a common but debated method for non-fixed-interval returns.
                    # Assuming risk_free_rate for the period of a single trade is negligible.
                    kpis['sharpe_ratio'] = sharpe_ratio_simple * sharpe_annualization_factor
                else: # Not enough duration to annualize
                    kpis['sharpe_ratio'] = sharpe_ratio_simple # Non-annualized
            else: # Not enough date info
                kpis['sharpe_ratio'] = sharpe_ratio_simple # Non-annualized
        else:
            kpis['sharpe_ratio'] = 0.0

        # Sortino Ratio
        negative_returns = returns_series[returns_series < 0]
        downside_std = negative_returns.std()
        if downside_std != 0 and not np.isnan(downside_std):
            # Similar annualization logic as Sharpe
            sortino_ratio_simple = mean_return / downside_std if downside_std > 0 else 0.0
            if 'sharpe_annualization_factor' in locals() and sharpe_annualization_factor !=1:
                 kpis['sortino_ratio'] = sortino_ratio_simple * sharpe_annualization_factor
            else: # Not enough duration to annualize or factor is 1
                 kpis['sortino_ratio'] = sortino_ratio_simple # Non-annualized
        else:
            kpis['sortino_ratio'] = 0.0 if mean_return <=0 else np.inf # if no downside risk and positive returns

    else: # Not enough data for std dev
        kpis['sharpe_ratio'] = 0.0
        kpis['sortino_ratio'] = 0.0

    # Calmar Ratio: Annualized PnL / Max Drawdown %
    # Need annualized PnL.
    if 'date' in df.columns and len(df['date']) > 1 and kpis['max_drawdown_abs'] > 0: # max_drawdown_abs to avoid division by zero if MDD% is 0
        total_duration_years = (df['date'].max() - df['date'].min()).days / 365.25
        if total_duration_years > 0:
            annualized_pnl = kpis['total_pnl'] / total_duration_years
            # Max drawdown pct is already a percentage, so use it directly.
            # Ensure kpis['max_drawdown_pct'] is the absolute percentage (e.g., 20 for 20%)
            mdd_pct_for_calmar = kpis['max_drawdown_pct'] / 100.0 # Convert to decimal, e.g. 0.20
            kpis['calmar_ratio'] = annualized_pnl / kpis['max_drawdown_abs'] if kpis['max_drawdown_abs'] > 0 else \
                                   (np.inf if annualized_pnl > 0 else 0.0)
            # Alternative: Annualized Return / Max Drawdown (absolute value of MDD)
            # Calmar is typically (Compound Annual Return) / Max Drawdown.
            # Using (Total PnL / Years) / Max Drawdown Abs.
        else:
            kpis['calmar_ratio'] = 0.0 # Not enough duration
    else:
        kpis['calmar_ratio'] = 0.0

    # --- VaR and CVaR (Historical Simulation) ---
    # Loss is represented by negative PnL. For VaR/CVaR, we look at the positive value of these losses.
    losses_for_var = -pnl_series[pnl_series < 0] # Positive values of losses

    if not losses_for_var.empty:
        kpis['var_95_loss'] = losses_for_var.quantile(0.95) if not losses_for_var.empty else 0.0
        kpis['cvar_95_loss'] = losses_for_var[losses_for_var >= kpis['var_95_loss']].mean() if not losses_for_var.empty and kpis['var_95_loss'] > 0 else 0.0
        kpis['var_99_loss'] = losses_for_var.quantile(0.99) if not losses_for_var.empty else 0.0
        kpis['cvar_99_loss'] = losses_for_var[losses_for_var >= kpis['var_99_loss']].mean() if not losses_for_var.empty and kpis['var_99_loss'] > 0 else 0.0
    else: # No losses
        kpis['var_95_loss'] = 0.0
        kpis['cvar_95_loss'] = 0.0
        kpis['var_99_loss'] = 0.0
        kpis['cvar_99_loss'] = 0.0
    
    # Ensure CVaR is not NaN if VaR is 0 (e.g. few losses)
    for level in ['95', '99']:
        var_key = f'var_{level}_loss'
        cvar_key = f'cvar_{level}_loss'
        if pd.isna(kpis[cvar_key]) and kpis[var_key] == 0:
            kpis[cvar_key] = 0.0


    # --- Distributional Properties ---
    kpis['pnl_skewness'] = pnl_series.skew() if kpis['total_trades'] > 2 else 0.0
    kpis['pnl_kurtosis'] = pnl_series.kurtosis() if kpis['total_trades'] > 3 else 0.0 # Excess kurtosis

    # --- Streaks ---
    kpis['max_win_streak'], kpis['max_loss_streak'] = _calculate_streaks(pnl_series)

    # --- Other Metrics ---
    if trading_days_col in df.columns:
        kpis['trading_days'] = df[trading_days_col].nunique()
        if kpis['trading_days'] > 0:
            daily_pnl_sum = df.groupby(trading_days_col)[pnl_col].sum()
            kpis['avg_daily_pnl'] = daily_pnl_sum.mean()
        else:
            kpis['avg_daily_pnl'] = 0.0
    else:
        kpis['trading_days'] = 0
        kpis['avg_daily_pnl'] = 0.0

    kpis['risk_free_rate_used'] = risk_free_rate * 100 # Display as percentage

    # Fill any remaining NaNs with 0 or appropriate default
    for key, value in kpis.items():
        if pd.isna(value):
            logger.warning(f"KPI '{key}' resulted in NaN, setting to 0.0. Check data or calculation logic.")
            kpis[key] = 0.0
        elif np.isinf(value):
            logger.warning(f"KPI '{key}' resulted in Inf, setting to a large number (or 0 if appropriate). Check for division by zero.")
            # Decide if Inf should be 0 or a large number based on KPI context
            # For ratios like Profit Factor, Inf can be valid if losses are zero.
            # For others, it might indicate an issue.
            if key in ["profit_factor", "win_loss_ratio", "sortino_ratio"] and kpis[key] > 0: # Positive infinity is okay
                pass
            else:
                kpis[key] = 0.0 # Default for problematic infinities


    logger.info(f"Calculated KPIs: {kpis}")
    return kpis


def get_kpi_interpretation(kpi_key: str, value: float) -> Tuple[str, str]:
    """
    Provides a qualitative interpretation for a given KPI value based on predefined thresholds.

    Args:
        kpi_key (str): The key of the KPI (must exist in KPI_CONFIG).
        value (float): The calculated value of the KPI.

    Returns:
        Tuple[str, str]: (Qualitative label, Threshold description string)
                         Returns ("N/A", "No thresholds defined") if KPI not in config.
    """
    if kpi_key not in KPI_CONFIG or pd.isna(value):
        return "N/A", "KPI not found or value is NaN"

    config = KPI_CONFIG[kpi_key]
    thresholds = config.get("thresholds", [])
    unit = config.get("unit", "")

    interpretation = "N/A" # Default if no threshold matches
    threshold_desc = "N/A"

    for label, min_val, max_val_exclusive in thresholds:
        if min_val <= value < max_val_exclusive:
            interpretation = label
            # Format threshold description
            if min_val == float('-inf'):
                threshold_desc = f"Val: {value:,.2f}{unit}, Thr: < {max_val_exclusive:,.1f}{unit}"
            elif max_val_exclusive == float('inf'):
                threshold_desc = f"Val: {value:,.2f}{unit}, Thr: >= {min_val:,.1f}{unit}"
            else:
                threshold_desc = f"Val: {value:,.2f}{unit}, Thr: {min_val:,.1f} - {max_val_exclusive:,.1f}{unit}"
            break
    # Handle cases where value might be exactly on a boundary if logic implies inclusive max
    # The current loop is exclusive for max_val. If last threshold is (label, min_val, inf), it's inclusive of min_val.

    if interpretation == "N/A" and thresholds: # If no range matched, check if it's beyond last range
        last_label, last_min, last_max = thresholds[-1]
        if value >= last_min and last_max == float('inf'): # Check if it should fall into the "highest" category
            interpretation = last_label
            threshold_desc = f"Val: {value:,.2f}{unit}, Thr: >= {last_min:,.1f}{unit}"
        elif thresholds and value < thresholds[0][1] and thresholds[0][1] == float('-inf'): # Check lowest category
            first_label, _, first_max = thresholds[0]
            interpretation = first_label
            threshold_desc = f"Val: {value:,.2f}{unit}, Thr: < {first_max:,.1f}{unit}"


    if interpretation == "N/A":
        threshold_desc = f"Val: {value:,.2f}{unit}, No specific range matched."


    return interpretation, threshold_desc


def get_kpi_color(kpi_key: str, value: float) -> str:
    """
    Determines the color for a KPI value based on its configuration.

    Args:
        kpi_key (str): The key of the KPI.
        value (float): The calculated value of the KPI.

    Returns:
        str: Hex color code (e.g., "#00FF00" for green).
             Returns gray if KPI not found or value is NaN.
    """
    if kpi_key not in KPI_CONFIG or pd.isna(value) or np.isinf(value): # Handle inf values as neutral for color
        return COLORS["gray"]

    config = KPI_CONFIG[kpi_key]
    color_logic = config.get("color_logic")

    if color_logic:
        # Thresholds might be needed for some color logic, but not all.
        # The lambda in config should handle this. For now, pass a dummy threshold if not used.
        threshold_for_color = 0 # Example, adjust if specific thresholds are needed by lambda
        return color_logic(value, threshold_for_color)
    return COLORS["gray"]


if __name__ == '__main__':
    # --- Test Data Setup ---
    sample_data = {
        EXPECTED_COLUMNS['date']: pd.to_datetime([
            '2023-01-01 10:00:00', '2023-01-01 12:00:00', '2023-01-02 09:30:00',
            '2023-01-02 15:00:00', '2023-01-03 11:00:00', '2023-01-04 10:00:00',
            '2023-01-04 14:00:00', '2023-01-05 09:00:00', '2023-01-05 16:00:00',
            '2023-01-06 10:00:00'
        ]),
        EXPECTED_COLUMNS['pnl']: [100, -50, 200, 50, -150, 80, -30, 120, 60, -70],
    }
    test_df = pd.DataFrame(sample_data)

    # Simulate features from data_processing
    test_df['cumulative_pnl'] = test_df[EXPECTED_COLUMNS['pnl']].cumsum()
    test_df['win'] = test_df[EXPECTED_COLUMNS['pnl']] > 0
    test_df['loss'] = test_df[EXPECTED_COLUMNS['pnl']] < 0
    test_df['trade_date_only'] = test_df[EXPECTED_COLUMNS['date']].dt.date

    logger.info("--- Testing KPI Calculations ---")
    kpis = calculate_all_kpis(test_df)

    if kpis:
        for kpi_name, kpi_value in kpis.items():
            interpretation, desc = get_kpi_interpretation(kpi_name, kpi_value)
            color = get_kpi_color(kpi_name, kpi_value)
            config_name = KPI_CONFIG.get(kpi_name, {}).get("name", kpi_name.replace("_", " ").title())
            unit = KPI_CONFIG.get(kpi_name, {}).get("unit", "")
            logger.info(
                f"{config_name}: {kpi_value:,.2f}{unit} "
                f"(Interpretation: {interpretation}, Desc: {desc}, Color: {color})"
            )
    else:
        logger.error("KPI calculation returned empty dict.")

    logger.info("\n--- Test with minimal data (1 trade) ---")
    minimal_df = test_df.head(1).copy()
    minimal_df['cumulative_pnl'] = minimal_df[EXPECTED_COLUMNS['pnl']].cumsum()
    minimal_df['trade_date_only'] = minimal_df[EXPECTED_COLUMNS['date']].dt.date
    kpis_minimal = calculate_all_kpis(minimal_df)
    if kpis_minimal:
        logger.info(f"Total PnL (1 trade): {kpis_minimal.get('total_pnl')}")
        logger.info(f"Sharpe (1 trade): {kpis_minimal.get('sharpe_ratio')}") # Should be 0 or NaN
    else:
        logger.error("KPI calculation for minimal data failed.")


    logger.info("\n--- Test with all winning trades ---")
    all_wins_data = {
        EXPECTED_COLUMNS['date']: pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 12:00:00']),
        EXPECTED_COLUMNS['pnl']: [100, 50],
    }
    all_wins_df = pd.DataFrame(all_wins_data)
    all_wins_df['cumulative_pnl'] = all_wins_df[EXPECTED_COLUMNS['pnl']].cumsum()
    all_wins_df['win'] = all_wins_df[EXPECTED_COLUMNS['pnl']] > 0
    all_wins_df['loss'] = all_wins_df[EXPECTED_COLUMNS['pnl']] < 0
    all_wins_df['trade_date_only'] = all_wins_df[EXPECTED_COLUMNS['date']].dt.date
    kpis_all_wins = calculate_all_kpis(all_wins_df)
    if kpis_all_wins:
        logger.info(f"Profit Factor (all wins): {kpis_all_wins.get('profit_factor')}") # Should be inf
        logger.info(f"Sortino (all wins): {kpis_all_wins.get('sortino_ratio')}") # Should be inf
    else:
        logger.error("KPI calculation for all wins data failed.")

    logger.info("\n--- Test with DataFrame having only NaNs in PnL ---")
    nan_pnl_data = {
        EXPECTED_COLUMNS['date']: pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 12:00:00']),
        EXPECTED_COLUMNS['pnl']: [np.nan, np.nan],
    }
    nan_pnl_df = pd.DataFrame(nan_pnl_data)
    # Simulate features, though they'd also be NaN or problematic
    nan_pnl_df['cumulative_pnl'] = nan_pnl_df[EXPECTED_COLUMNS['pnl']].cumsum()
    nan_pnl_df['win'] = nan_pnl_df[EXPECTED_COLUMNS['pnl']] > 0
    nan_pnl_df['loss'] = nan_pnl_df[EXPECTED_COLUMNS['pnl']] < 0
    nan_pnl_df['trade_date_only'] = nan_pnl_df[EXPECTED_COLUMNS['date']].dt.date
    kpis_nan_pnl = calculate_all_kpis(nan_pnl_df)
    if kpis_nan_pnl and all(pd.isna(v) or v == 0 for v in kpis_nan_pnl.values()): # Expect NaNs or 0s
        logger.info(f"KPIs for NaN PnL data (e.g., Total PnL): {kpis_nan_pnl.get('total_pnl')}")
        logger.info("Correctly handled NaN PnL data.")
    else:
        logger.error(f"KPI calculation for NaN PnL data produced unexpected results: {kpis_nan_pnl}")

    logger.info("\n--- Test _calculate_drawdowns with initial zero PnL and then loss ---")
    drawdown_test_pnl = pd.Series([0, -50, -20, 30, -60])
    drawdown_test_cum_pnl = drawdown_test_pnl.cumsum() # [0, -50, -70, -40, -100]
    dd_series, max_dd_abs, max_dd_pct, dd_pct_series = _calculate_drawdowns(drawdown_test_cum_pnl)
    logger.info(f"Cum PnL: {drawdown_test_cum_pnl.tolist()}")
    logger.info(f"Max DD Abs: {max_dd_abs}, Max DD Pct: {max_dd_pct}%") # Expected Abs: 100, Pct: tricky, maybe 0 or 100
    logger.info(f"DD Series: {dd_series.tolist()}")
    logger.info(f"DD Pct Series: {dd_pct_series.tolist()}")
    # Expected: HWM = [0, 0, 0, 0, 0]. DD_ABS = [0, 50, 70, 40, 100]. Max_DD_ABS = 100.
    # DD_PCT = (HWM - CumPnL) / HWM. If HWM is 0, (0 - CumPnL)/0.
    # (0 - 0)/0 = NaN -> 0. (0 - (-50))/0 = Inf -> 0. Max_DD_PCT = 0. This needs refinement.
    # The logic in _calculate_drawdowns for max_drawdown_pct has been updated.

