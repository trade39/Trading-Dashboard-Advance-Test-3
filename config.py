"""
config.py

Core configuration constants for the Trading Performance Dashboard.
This includes expected CSV column names, KPI thresholds, colors,
and qualitative labels for KPI interpretation.
"""

from typing import Dict, List, Tuple

# --- General Settings ---
APP_TITLE: str = "Trading Performance Dashboard"
RISK_FREE_RATE: float = 0.02  # Example: 2% annual risk-free rate
FORECAST_HORIZON: int = 30  # Default number of periods for forecasts

# --- CSV Column Names ---
# IMPORTANT: These MUST match the column names in your CSV *after* they have been
# processed by data_processing.py (lowercase, spaces to underscores, stripped).
EXPECTED_COLUMNS: Dict[str, str] = {
    "trade_id": "trade_id",       # From CSV "Trade ID" -> "trade_id"
    "date": "date",               # From CSV "Date" -> "date"
    "symbol": "symbol_1",         # From CSV "Symbol 1" -> "symbol_1" (will be cleaned further in data_processing)
    "entry_price": "entry",       # From CSV "Entry" -> "entry"
    "exit_price": "exit",         # From CSV "Exit" -> "exit"
    "pnl": "pnl",                 # From CSV "PnL" -> "pnl"
    "risk": "risk",               # This is engineered if not present. If your CSV has "Risk", map it.
    "notes": "lesson_learned",    # From CSV "Lesson learned" -> "lesson_learned"
    "strategy": "trade_model",    # CORRECTED: From CSV "Trade Model " -> "trade_model" (no trailing underscore)
    "signal_confidence": "signal_confidence", # Ensure this column exists in your CSV if used
    "duration_minutes": "duration_mins", # From CSV "Duration (mins)" -> "duration_mins"
}

# --- UI Colors ---
# Defines a palette for consistent styling across the application.
COLORS: Dict[str, str] = {
    "royal_blue": "#4169E1",
    "green": "#00FF00",
    "red": "#FF0000",
    "gray": "#808080",
    "orange": "#FFA500",  # Added for VaR/CVaR KPI color logic
    "dark_background": "#1C2526",
    "light_background": "#FFFFFF",
    "text_dark": "#E0E0E0",
    "text_light": "#333333",
    "text_muted_color": "#A0A0A0",
    "card_background_dark": "#273334", # Used for KPI cards, expanders in dark mode
    "card_border_dark": "#4169E1",    # Border for cards in dark mode
    "card_background_light": "#F0F2F6",# Used for KPI cards, expanders in light mode
    "card_border_light": "#4169E1"    # Border for cards in light mode
}

# --- KPI Definitions and Thresholds ---
# Configures each Key Performance Indicator (KPI) for display and interpretation.
# - "name": Display name of the KPI.
# - "unit": Unit of measurement (e.g., "$", "%").
# - "interpretation_type": "higher_is_better", "lower_is_better", or "neutral".
# - "thresholds": List of tuples (label, min_val, max_val_exclusive) for qualitative interpretation.
# - "color_logic": Lambda function determining the display color based on value and an optional threshold.
KPI_CONFIG: Dict[str, Dict] = {
    "total_pnl": {
        "name": "Total PnL", "unit": "$", "interpretation_type": "higher_is_better",
        "thresholds": [("Negative", float('-inf'), 0), ("Slightly Positive", 0, 1000), ("Moderately Positive", 1000, 10000), ("Highly Positive", 10000, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 0 else (COLORS["red"] if v < 0 else COLORS["gray"])
    },
    "total_trades": {
        "name": "Total Trades", "unit": "", "interpretation_type": "neutral",
        "thresholds": [("Low", 0, 50), ("Moderate", 50, 200), ("High", 200, float('inf'))],
        "color_logic": lambda v, t: COLORS["gray"]
    },
    "win_rate": {
        "name": "Win Rate", "unit": "%", "interpretation_type": "higher_is_better",
        "thresholds": [("Very Low", 0, 30),("Low", 30, 40),("Acceptable", 40, 50),("Good", 50, 60),("Very Good", 60, 70),("Excellent", 70, 80),("Exceptional", 80, 101)],
        "color_logic": lambda v, t: COLORS["green"] if v >= 50 else COLORS["red"]
    },
    "loss_rate": {
        "name": "Loss Rate", "unit": "%", "interpretation_type": "lower_is_better",
        "thresholds": [("Exceptional", 0, 20),("Excellent", 20, 30),("Very Good", 30, 40),("Good", 40, 50),("Acceptable", 50, 60),("High", 60, 70),("Very High", 70, 101)],
        "color_logic": lambda v, t: COLORS["red"] if v > 50 else COLORS["green"]
    },
    "profit_factor": {
        "name": "Profit Factor", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [("Negative", float('-inf'), 1.0), ("Break-even", 1.0, 1.01), ("Acceptable", 1.01, 1.5),("Good", 1.5, 2.0),("Very Good", 2.0, 3.0),("Exceptional", 3.0, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 1 else COLORS["red"]
    },
    "avg_trade_pnl": {
        "name": "Average Trade PnL", "unit": "$", "interpretation_type": "higher_is_better",
        "thresholds": [("Negative", float('-inf'), 0),("Neutral", 0, 1),("Positive", 1, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 0 else (COLORS["red"] if v < 0 else COLORS["gray"])
    },
    "avg_win": {
        "name": "Average Win", "unit": "$", "interpretation_type": "higher_is_better",
        "thresholds": [("Low", 0, 50), ("Moderate", 50, 200),("High", 200, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 0 else COLORS["gray"]
    },
    "avg_loss": {
        "name": "Average Loss", "unit": "$", "interpretation_type": "lower_is_better", # Value is absolute loss
        "thresholds": [("Low", 0, 50),("Moderate", 50, 200),("High", 200, float('inf'))],
        "color_logic": lambda v, t: COLORS["red"] if v > 0 else COLORS["gray"] # Red if any loss, gray if zero
    },
    "win_loss_ratio": { # Ratio of Avg Win / Avg Loss
        "name": "Win/Loss Ratio", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [("Poor", 0, 1.0),("Acceptable", 1.0, 1.5),("Good", 1.5, 2.0),("Very Good", 2.0, 3.0),("Exceptional", 3.0, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 1 else COLORS["red"]
    },
    "max_drawdown_abs": {
        "name": "Max Drawdown", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [("Low", 0, 1000),("Moderate", 1000, 5000),("High", 5000, float('inf'))], # Absolute value
        "color_logic": lambda v, t: COLORS["red"] if v > 0 else COLORS["gray"] # Red if any drawdown
    },
    "max_drawdown_pct": {
        "name": "Max Drawdown", "unit": "%", "interpretation_type": "lower_is_better",
        "thresholds": [("Very Low", 0, 5),("Low", 5, 10),("Moderate", 10, 20),("High (Caution)", 20, 30),("Very High (Danger)", 30, 101)],
        "color_logic": lambda v, t: COLORS["red"] if v >= 20 else (COLORS["green"] if v < 10 else COLORS["gray"])
    },
    "sharpe_ratio": {
        "name": "Sharpe Ratio", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [("Poor", float('-inf'), 0),("Subpar", 0, 1.0),("Good", 1.0, 2.0),("Excellent", 2.0, 3.0),("Exceptional", 3.0, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 1 else (COLORS["red"] if v < 0 else COLORS["gray"])
    },
    "sortino_ratio": {
        "name": "Sortino Ratio", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [("Poor", float('-inf'), 0),("Subpar", 0, 1.0),("Good", 1.0, 2.0),("Excellent", 2.0, 3.0),("Exceptional", 3.0, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 1 else (COLORS["red"] if v < 0 else COLORS["gray"])
    },
    "calmar_ratio": {
        "name": "Calmar Ratio", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [("Poor", float('-inf'), 0),("Subpar", 0, 0.5),("Acceptable", 0.5, 1.0),("Good", 1.0, 2.0),("Excellent", 2.0, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 1 else (COLORS["red"] if v < 0 else COLORS["gray"])
    },
    "var_95_loss": { # Value is positive loss amount
        "name": "VaR 95% (Loss)", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [("Low Risk", 0, 500), ("Moderate Risk", 500, 2000), ("High Risk", 2000, float('inf'))],
        "color_logic": lambda v, t: COLORS["red"] if v > 1000 else (COLORS["gray"] if v == 0 else COLORS["orange"])
    },
    "cvar_95_loss": { # Value is positive loss amount
        "name": "CVaR 95% (Loss)", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [("Low Risk", 0, 500), ("Moderate Risk", 500, 2000), ("High Risk", 2000, float('inf'))],
        "color_logic": lambda v, t: COLORS["red"] if v > 1000 else (COLORS["gray"] if v == 0 else COLORS["orange"])
    },
    "var_99_loss": { # Value is positive loss amount
        "name": "VaR 99% (Loss)", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [("Low Risk", 0, 750), ("Moderate Risk", 750, 3000), ("High Risk", 3000, float('inf'))],
        "color_logic": lambda v, t: COLORS["red"] if v > 1500 else (COLORS["gray"] if v == 0 else COLORS["orange"])
    },
    "cvar_99_loss": { # Value is positive loss amount
        "name": "CVaR 99% (Loss)", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [("Low Risk", 0, 750), ("Moderate Risk", 750, 3000), ("High Risk", 3000, float('inf'))],
        "color_logic": lambda v, t: COLORS["red"] if v > 1500 else (COLORS["gray"] if v == 0 else COLORS["orange"])
    },
    "pnl_skewness": {
        "name": "PnL Skewness", "unit": "", "interpretation_type": "neutral", # Can be good or bad depending on strategy
        "thresholds": [("Highly Negative", float('-inf'), -1.0), ("Moderately Negative", -1.0, -0.5), ("Symmetric", -0.5, 0.5), ("Moderately Positive", 0.5, 1.0), ("Highly Positive", 1.0, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 0.5 else (COLORS["red"] if v < -0.5 else COLORS["gray"])
    },
    "pnl_kurtosis": { # Excess Kurtosis
        "name": "PnL Kurtosis (Excess)", "unit": "", "interpretation_type": "neutral", # Higher means fatter tails
        "thresholds": [("Platykurtic (Thin)", float('-inf'), -0.5),("Mesokurtic (Normal)", -0.5, 0.5),("Leptokurtic (Fat)", 0.5, 3.0),("Highly Leptokurtic (Very Fat)", 3.0, float('inf'))],
        "color_logic": lambda v, t: COLORS["red"] if v > 1 else COLORS["gray"] # Red for fatter tails (often higher risk)
    },
    "max_win_streak": {
        "name": "Max Win Streak", "unit": " trades", "interpretation_type": "higher_is_better",
        "thresholds": [("Low", 0, 3),("Moderate", 3, 7),("High", 7, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v >= 3 else COLORS["gray"]
    },
    "max_loss_streak": {
        "name": "Max Loss Streak", "unit": " trades", "interpretation_type": "lower_is_better",
        "thresholds": [("Low", 0, 3),("Moderate", 3, 7),("High", 7, float('inf'))],
        "color_logic": lambda v, t: COLORS["red"] if v >= 5 else COLORS["gray"]
    },
    "avg_daily_pnl": {
        "name": "Average Daily PnL", "unit": "$", "interpretation_type": "higher_is_better",
        "thresholds": [("Negative", float('-inf'), 0),("Neutral", 0, 1),("Positive", 1, float('inf'))],
        "color_logic": lambda v, t: COLORS["green"] if v > 0 else (COLORS["red"] if v < 0 else COLORS["gray"])
    },
    "trading_days": {
        "name": "Trading Days", "unit": "", "interpretation_type": "neutral",
        "thresholds": [("Short Period", 0, 21), ("Medium Period", 21, 63), ("Sufficient Period", 63, 252),("Long Period", 252, float('inf'))],
        "color_logic": lambda v, t: COLORS["gray"]
    },
    "risk_free_rate_used": { # Informational KPI
        "name": "Risk-Free Rate Used", "unit": "%", "interpretation_type": "neutral",
        "thresholds": [("Standard Setting", 0, float('inf'))],
        "color_logic": lambda v, t: COLORS["gray"]
    }
}

# Default order for displaying KPIs in the cluster.
DEFAULT_KPI_DISPLAY_ORDER: List[str] = [
    "total_pnl", "total_trades", "win_rate", "loss_rate", "profit_factor", "avg_trade_pnl", "avg_win", "avg_loss",
    "win_loss_ratio", "max_drawdown_abs", "max_drawdown_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "var_95_loss", "cvar_95_loss", "var_99_loss", "cvar_99_loss", "pnl_skewness", "pnl_kurtosis",
    "max_win_streak", "max_loss_streak", "avg_daily_pnl", "trading_days", "risk_free_rate_used"
]

# --- Plotting Themes and Colors ---
PLOTLY_THEME_DARK: str = "plotly_dark"
PLOTLY_THEME_LIGHT: str = "plotly_white" # Standard Plotly light theme

# Custom colors for plots, aligned with UI colors
PLOT_BG_COLOR_DARK: str = COLORS["dark_background"]
PLOT_PAPER_BG_COLOR_DARK: str = COLORS["dark_background"]
PLOT_FONT_COLOR_DARK: str = COLORS["text_dark"]

PLOT_BG_COLOR_LIGHT: str = COLORS["light_background"]
PLOT_PAPER_BG_COLOR_LIGHT: str = COLORS["light_background"]
PLOT_FONT_COLOR_LIGHT: str = COLORS["text_light"]

PLOT_LINE_COLOR: str = COLORS["royal_blue"] # General line color for plots
PLOT_MARKER_PROFIT_COLOR: str = COLORS["green"]
PLOT_MARKER_LOSS_COLOR: str = COLORS["red"]

# --- Advanced Analysis Defaults ---
BOOTSTRAP_ITERATIONS: int = 1000
CONFIDENCE_LEVEL: float = 0.95 # For confidence intervals (e.g., 95% CI)
DISTRIBUTIONS_TO_FIT: List[str] = ['norm', 't', 'laplace', 'johnsonsu', 'genextreme'] # For PnL distribution fitting
MARKOV_MAX_LAG: int = 1 # For Markov chain analysis

# --- Logging Configuration ---
# These are defaults; app.py's setup_logger call will primarily use these.
LOG_FILE: str = "logs/trading_dashboard_app.log" # Path for the log file
LOG_LEVEL: str = "INFO" # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
