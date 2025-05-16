"""
config.py

Core configuration constants for the Trading Performance Dashboard.
This includes expected CSV column names, KPI thresholds, colors,
qualitative labels for KPI interpretation, and KPI groupings for display.
"""

from typing import Dict, List, Tuple

# --- General Settings ---
APP_TITLE: str = "Trading Performance Dashboard"
RISK_FREE_RATE: float = 0.02
FORECAST_HORIZON: int = 30

# --- Benchmark Configuration ---
DEFAULT_BENCHMARK_TICKER: str = "SPY"
AVAILABLE_BENCHMARKS: Dict[str, str] = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Dow Jones (DIA)": "DIA",
    "Russell 2000 (IWM)": "IWM",
    "Gold (GLD)": "GLD",
    "None": ""
}

# --- CSV Column Names ---
EXPECTED_COLUMNS: Dict[str, str] = {
    "trade_id": "trade_id", "date": "date", "symbol": "symbol_1",
    "entry_price": "entry", "exit_price": "exit", "pnl": "pnl",
    "risk": "risk", "notes": "lesson_learned", "strategy": "trade_model",
    "signal_confidence": "signal_confidence", "duration_minutes": "duration_mins",
}

# --- UI Colors ---
COLORS: Dict[str, str] = {
    "royal_blue": "#4169E1", "green": "#00FF00", "red": "#FF0000",
    "gray": "#808080", "orange": "#FFA500", "purple": "#8A2BE2",
    "dark_background": "#1C2526", "light_background": "#FFFFFF",
    "text_dark": "#E0E0E0", "text_light": "#333333",
    "text_muted_color": "#A0A0A0", "card_background_dark": "#273334",
    "card_border_dark": "#4169E1", "card_background_light": "#F0F2F6",
    "card_border_light": "#4169E1"
}

# --- KPI Definitions and Thresholds ---
# (KPI_CONFIG dictionary remains the same as your latest version - not repeated here for brevity,
# but ensure it includes all necessary KPIs, including benchmark-related ones)
KPI_CONFIG: Dict[str, Dict] = {
    "total_pnl": {"name": "Total PnL", "unit": "$", "interpretation_type": "higher_is_better", "thresholds": [("Negative", float('-inf'), 0), ("Slightly Positive", 0, 1000), ("Moderately Positive", 1000, 10000), ("Highly Positive", 10000, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 0 else (COLORS["red"] if v < 0 else COLORS["gray"])},
    "total_trades": {"name": "Total Trades", "unit": "", "interpretation_type": "neutral", "thresholds": [("Low", 0, 50), ("Moderate", 50, 200), ("High", 200, float('inf'))], "color_logic": lambda v, t: COLORS["gray"]},
    "win_rate": {"name": "Win Rate", "unit": "%", "interpretation_type": "higher_is_better", "thresholds": [("Very Low", 0, 30),("Low", 30, 40),("Acceptable", 40, 50),("Good", 50, 60),("Very Good", 60, 70),("Excellent", 70, 80),("Exceptional", 80, 101)], "color_logic": lambda v, t: COLORS["green"] if v >= 50 else COLORS["red"]},
    "loss_rate": {"name": "Loss Rate", "unit": "%", "interpretation_type": "lower_is_better", "thresholds": [("Exceptional", 0, 20),("Excellent", 20, 30),("Very Good", 30, 40),("Good", 40, 50),("Acceptable", 50, 60),("High", 60, 70),("Very High", 70, 101)], "color_logic": lambda v, t: COLORS["red"] if v > 50 else COLORS["green"]},
    "profit_factor": {"name": "Profit Factor", "unit": "", "interpretation_type": "higher_is_better", "thresholds": [("Negative", float('-inf'), 1.0), ("Break-even", 1.0, 1.01), ("Acceptable", 1.01, 1.5),("Good", 1.5, 2.0),("Very Good", 2.0, 3.0),("Exceptional", 3.0, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 1 else COLORS["red"]},
    "avg_trade_pnl": {"name": "Average Trade PnL", "unit": "$", "interpretation_type": "higher_is_better", "thresholds": [("Negative", float('-inf'), 0),("Neutral", 0, 1),("Positive", 1, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 0 else (COLORS["red"] if v < 0 else COLORS["gray"])},
    "avg_win": {"name": "Average Win", "unit": "$", "interpretation_type": "higher_is_better", "thresholds": [("Low", 0, 50), ("Moderate", 50, 200),("High", 200, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 0 else COLORS["gray"]},
    "avg_loss": {"name": "Average Loss", "unit": "$", "interpretation_type": "lower_is_better", "thresholds": [("Low", 0, 50),("Moderate", 50, 200),("High", 200, float('inf'))], "color_logic": lambda v, t: COLORS["red"] if v > 0 else COLORS["gray"]},
    "win_loss_ratio": {"name": "Win/Loss Ratio", "unit": "", "interpretation_type": "higher_is_better", "thresholds": [("Poor", 0, 1.0),("Acceptable", 1.0, 1.5),("Good", 1.5, 2.0),("Very Good", 2.0, 3.0),("Exceptional", 3.0, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 1 else COLORS["red"]},
    "max_drawdown_abs": {"name": "Max Drawdown Abs", "unit": "$", "interpretation_type": "lower_is_better", "thresholds": [("Low", 0, 1000),("Moderate", 1000, 5000),("High", 5000, float('inf'))], "color_logic": lambda v, t: COLORS["red"] if v > 0 else COLORS["gray"]}, # Corrected name
    "max_drawdown_pct": {"name": "Max Drawdown %", "unit": "%", "interpretation_type": "lower_is_better", "thresholds": [("Very Low", 0, 5),("Low", 5, 10),("Moderate", 10, 20),("High (Caution)", 20, 30),("Very High (Danger)", 30, 101)], "color_logic": lambda v, t: COLORS["red"] if v >= 20 else (COLORS["green"] if v < 10 else COLORS["gray"])}, # Corrected name
    "sharpe_ratio": {"name": "Sharpe Ratio", "unit": "", "interpretation_type": "higher_is_better", "thresholds": [("Poor", float('-inf'), 0),("Subpar", 0, 1.0),("Good", 1.0, 2.0),("Excellent", 2.0, 3.0),("Exceptional", 3.0, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 1 else (COLORS["red"] if v < 0 else COLORS["gray"])},
    "sortino_ratio": {"name": "Sortino Ratio", "unit": "", "interpretation_type": "higher_is_better", "thresholds": [("Poor", float('-inf'), 0),("Subpar", 0, 1.0),("Good", 1.0, 2.0),("Excellent", 2.0, 3.0),("Exceptional", 3.0, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 1 else (COLORS["red"] if v < 0 else COLORS["gray"])},
    "calmar_ratio": {"name": "Calmar Ratio", "unit": "", "interpretation_type": "higher_is_better", "thresholds": [("Poor", float('-inf'), 0),("Subpar", 0, 0.5),("Acceptable", 0.5, 1.0),("Good", 1.0, 2.0),("Excellent", 2.0, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 1 else (COLORS["red"] if v < 0 else COLORS["gray"])},
    "var_95_loss": {"name": "VaR 95% (Loss)", "unit": "$", "interpretation_type": "lower_is_better", "thresholds": [("Low Risk", 0, 500), ("Moderate Risk", 500, 2000), ("High Risk", 2000, float('inf'))], "color_logic": lambda v, t: COLORS["red"] if v > 1000 else (COLORS["gray"] if v == 0 else COLORS["orange"])},
    "cvar_95_loss": {"name": "CVaR 95% (Loss)", "unit": "$", "interpretation_type": "lower_is_better", "thresholds": [("Low Risk", 0, 500), ("Moderate Risk", 500, 2000), ("High Risk", 2000, float('inf'))], "color_logic": lambda v, t: COLORS["red"] if v > 1000 else (COLORS["gray"] if v == 0 else COLORS["orange"])},
    "var_99_loss": {"name": "VaR 99% (Loss)", "unit": "$", "interpretation_type": "lower_is_better", "thresholds": [("Low Risk", 0, 750), ("Moderate Risk", 750, 3000), ("High Risk", 3000, float('inf'))], "color_logic": lambda v, t: COLORS["red"] if v > 1500 else (COLORS["gray"] if v == 0 else COLORS["orange"])},
    "cvar_99_loss": {"name": "CVaR 99% (Loss)", "unit": "$", "interpretation_type": "lower_is_better", "thresholds": [("Low Risk", 0, 750), ("Moderate Risk", 750, 3000), ("High Risk", 3000, float('inf'))], "color_logic": lambda v, t: COLORS["red"] if v > 1500 else (COLORS["gray"] if v == 0 else COLORS["orange"])},
    "pnl_skewness": {"name": "PnL Skewness", "unit": "", "interpretation_type": "neutral", "thresholds": [("Highly Negative", float('-inf'), -1.0), ("Moderately Negative", -1.0, -0.5), ("Symmetric", -0.5, 0.5), ("Moderately Positive", 0.5, 1.0), ("Highly Positive", 1.0, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 0.5 else (COLORS["red"] if v < -0.5 else COLORS["gray"])},
    "pnl_kurtosis": {"name": "PnL Kurtosis (Excess)", "unit": "", "interpretation_type": "neutral", "thresholds": [("Platykurtic (Thin)", float('-inf'), -0.5),("Mesokurtic (Normal)", -0.5, 0.5),("Leptokurtic (Fat)", 0.5, 3.0),("Highly Leptokurtic (Very Fat)", 3.0, float('inf'))], "color_logic": lambda v, t: COLORS["red"] if v > 1 else COLORS["gray"]},
    "max_win_streak": {"name": "Max Win Streak", "unit": " trades", "interpretation_type": "higher_is_better", "thresholds": [("Low", 0, 3),("Moderate", 3, 7),("High", 7, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v >= 3 else COLORS["gray"]},
    "max_loss_streak": {"name": "Max Loss Streak", "unit": " trades", "interpretation_type": "lower_is_better", "thresholds": [("Low", 0, 3),("Moderate", 3, 7),("High", 7, float('inf'))], "color_logic": lambda v, t: COLORS["red"] if v >= 5 else COLORS["gray"]},
    "avg_daily_pnl": {"name": "Average Daily PnL", "unit": "$", "interpretation_type": "higher_is_better", "thresholds": [("Negative", float('-inf'), 0),("Neutral", 0, 1),("Positive", 1, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 0 else (COLORS["red"] if v < 0 else COLORS["gray"])},
    "trading_days": {"name": "Trading Days", "unit": "", "interpretation_type": "neutral", "thresholds": [("Short Period", 0, 21), ("Medium Period", 21, 63), ("Sufficient Period", 63, 252),("Long Period", 252, float('inf'))], "color_logic": lambda v, t: COLORS["gray"]},
    "risk_free_rate_used": {"name": "Risk-Free Rate Used", "unit": "%", "interpretation_type": "neutral", "thresholds": [("Standard Setting", 0, float('inf'))], "color_logic": lambda v, t: COLORS["gray"]},
    "benchmark_total_return": {"name": "Benchmark Total Return", "unit": "%", "interpretation_type": "neutral", "thresholds": [("Negative", float('-inf'), 0), ("Positive", 0, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 0 else (COLORS["red"] if v < 0 else COLORS["gray"])},
    "alpha": {"name": "Alpha (Annualized)", "unit": "%", "interpretation_type": "higher_is_better", "thresholds": [("Negative Alpha", float('-inf'), 0), ("Positive Alpha", 0, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 0 else (COLORS["red"] if v < 0 else COLORS["gray"])},
    "beta": {"name": "Beta", "unit": "", "interpretation_type": "neutral", "thresholds": [("Low Volatility", 0, 0.8), ("Market Volatility", 0.8, 1.2), ("High Volatility", 1.2, float('inf'))], "color_logic": lambda v, t: COLORS["gray"]},
    "benchmark_correlation": {"name": "Correlation to Benchmark", "unit": "", "interpretation_type": "neutral", "thresholds": [("Negative", -1.0, -0.5), ("Low", -0.5, 0.5), ("Positive", 0.5, 1.01)], "color_logic": lambda v, t: COLORS["purple"]},
    "tracking_error": {"name": "Tracking Error (Annualized)", "unit": "%", "interpretation_type": "lower_is_better", "thresholds": [("Low", 0, 5), ("Moderate", 5, 15), ("High", 15, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v < 5 else (COLORS["orange"] if v < 15 else COLORS["red"])},
    "information_ratio": {"name": "Information Ratio", "unit": "", "interpretation_type": "higher_is_better", "thresholds": [("Poor", float('-inf'), 0.0), ("Fair", 0.0, 0.5), ("Good", 0.5, 1.0), ("Excellent", 1.0, float('inf'))], "color_logic": lambda v, t: COLORS["green"] if v > 0.5 else (COLORS["orange"] if v > 0 else COLORS["red"])}
}


# --- KPI Groupings for Display ---
KPI_GROUPS_OVERVIEW: Dict[str, List[str]] = {
    "Overall Performance": ["total_pnl", "total_trades", "trading_days", "avg_daily_pnl"],
    "Profitability & Efficiency": ["win_rate", "loss_rate", "profit_factor", "avg_trade_pnl", "avg_win", "avg_loss", "win_loss_ratio"],
    "Risk-Adjusted Returns": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
    "Drawdown & Streaks": ["max_drawdown_abs", "max_drawdown_pct", "max_win_streak", "max_loss_streak"],
    "Benchmark Comparison": ["benchmark_total_return", "alpha", "beta", "benchmark_correlation", "tracking_error", "information_ratio"],
    "Distributional Properties": ["pnl_skewness", "pnl_kurtosis"]
}

KPI_GROUPS_RISK_DURATION: Dict[str, List[str]] = {
    "Drawdown Metrics": ["max_drawdown_abs", "max_drawdown_pct"],
    "Value at Risk (VaR & CVaR)": ["var_95_loss", "cvar_95_loss", "var_99_loss", "cvar_99_loss"],
    "Risk-Adjusted Ratios (vs. Self)": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
    "Market Risk & Relative Performance": ["beta", "alpha", "benchmark_correlation", "tracking_error", "information_ratio"],
    "Return Distribution Risk": ["pnl_skewness", "pnl_kurtosis"]
}
# This DEFAULT_KPI_DISPLAY_ORDER can still be useful for contexts where all KPIs are shown,
# or as a fallback if a KPI is not in a specific group.
DEFAULT_KPI_DISPLAY_ORDER: List[str] = [
    "total_pnl", "total_trades", "win_rate", "loss_rate", "profit_factor", "avg_trade_pnl",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_pct", "max_drawdown_abs",
    "benchmark_total_return", "alpha", "beta", "benchmark_correlation", "tracking_error", "information_ratio",
    "avg_win", "avg_loss", "win_loss_ratio",
    "var_95_loss", "cvar_95_loss", "var_99_loss", "cvar_99_loss",
    "pnl_skewness", "pnl_kurtosis",
    "max_win_streak", "max_loss_streak", "avg_daily_pnl", "trading_days", "risk_free_rate_used"
]


# --- Plotting Themes and Colors ---
PLOTLY_THEME_DARK: str = "plotly_dark"
PLOTLY_THEME_LIGHT: str = "plotly_white"
PLOT_BG_COLOR_DARK: str = COLORS["dark_background"]
PLOT_PAPER_BG_COLOR_DARK: str = COLORS["dark_background"]
PLOT_FONT_COLOR_DARK: str = COLORS["text_dark"]
PLOT_BG_COLOR_LIGHT: str = COLORS["light_background"]
PLOT_PAPER_BG_COLOR_LIGHT: str = COLORS["light_background"]
PLOT_FONT_COLOR_LIGHT: str = COLORS["text_light"]
PLOT_LINE_COLOR: str = COLORS["royal_blue"]
PLOT_BENCHMARK_LINE_COLOR: str = COLORS["purple"]
PLOT_MARKER_PROFIT_COLOR: str = COLORS["green"]
PLOT_MARKER_LOSS_COLOR: str = COLORS["red"]

# --- Advanced Analysis Defaults ---
BOOTSTRAP_ITERATIONS: int = 1000
CONFIDENCE_LEVEL: float = 0.95
DISTRIBUTIONS_TO_FIT: List[str] = ['norm', 't', 'laplace', 'johnsonsu', 'genextreme']
MARKOV_MAX_LAG: int = 1

# --- Logging Configuration ---
LOG_FILE: str = "logs/trading_dashboard_app.log"
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
