"""
components/kpi_display.py

This component is responsible for orchestrating the display of multiple
Key Performance Indicators (KPIs) in a structured layout, typically using cards.
It leverages the `display_kpi_card` utility for individual card rendering
and now explicitly handles confidence intervals.
"""
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple

# Assuming root level config and utils
try:
    from config import KPI_CONFIG, DEFAULT_KPI_DISPLAY_ORDER, APP_TITLE
    from utils.common_utils import display_kpi_card # display_kpi_card should handle CI display
    # calculations module provides interpretation and color logic
    from calculations import get_kpi_interpretation, get_kpi_color
except ImportError:
    # Fallback for standalone testing or if imports fail
    print("Warning (kpi_display.py): Could not import from root config/utils/calculations. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default"
    DEFAULT_KPI_DISPLAY_ORDER = ["total_pnl", "win_rate", "profit_factor", "sharpe_ratio"]
    KPI_CONFIG = { # Minimal placeholder
        "total_pnl": {"name": "Total PnL", "unit": "$"}, "win_rate": {"name": "Win Rate", "unit": "%"},
        "profit_factor": {"name": "Profit Factor"}, "sharpe_ratio": {"name": "Sharpe Ratio"}
    }
    # Placeholder for display_kpi_card if common_utils is not available
    def display_kpi_card(title, value, unit, interpretation, interpretation_desc, color, confidence_interval=None, key_suffix=""):
        ci_text = f" CI: [{confidence_interval[0]:.2f}-{confidence_interval[1]:.2f}]" if confidence_interval else ""
        st.metric(label=title, value=f"{value}{unit}{ci_text}", delta=interpretation_desc if interpretation_desc else interpretation)

    def get_kpi_interpretation(k_key, val): return "N/A", f"Val: {val}"
    def get_kpi_color(k_key, val): return "#808080" # gray

import logging
logger = logging.getLogger(APP_TITLE) # Get the main app logger

class KPIClusterDisplay:
    """
    A component to display a cluster of KPIs in a grid layout.
    """
    def __init__(
        self,
        kpi_results: Dict[str, Any],
        kpi_definitions: Dict[str, Dict] = KPI_CONFIG,
        kpi_order: List[str] = DEFAULT_KPI_DISPLAY_ORDER,
        # kpi_confidence_intervals: Keys should be the kpi_key (e.g., "avg_trade_pnl")
        # and value is a tuple (lower_bound, upper_bound).
        kpi_confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
        cols_per_row: int = 4
    ):
        """
        Initialize the KPIClusterDisplay.

        Args:
            kpi_results (Dict[str, Any]): Dictionary of calculated KPI values.
            kpi_definitions (Dict[str, Dict]): Configuration for each KPI.
            kpi_order (List[str]): Order in which to display KPIs.
            kpi_confidence_intervals (Optional[Dict[str, Tuple[float, float]]]):
                Pre-calculated 95% confidence intervals for KPIs.
                The keys should directly match the kpi_key (e.g., 'avg_trade_pnl').
            cols_per_row (int): Number of columns for the KPI card grid.
        """
        self.kpi_results = kpi_results if kpi_results else {}
        self.kpi_definitions = kpi_definitions
        self.kpi_order = kpi_order
        self.kpi_confidence_intervals = kpi_confidence_intervals if kpi_confidence_intervals else {}
        self.cols_per_row = max(1, cols_per_row)
        logger.debug(f"KPIClusterDisplay initialized with {len(self.kpi_results)} KPI results.")

    def render(self) -> None:
        """
        Renders the KPI cards in the specified layout.
        """
        if not self.kpi_results:
            logger.info("KPIClusterDisplay: No KPI results to display.")
            # User feedback for no KPIs should be handled by the calling script (app.py)
            return

        st_cols = st.columns(self.cols_per_row)
        current_col_idx = 0

        for kpi_key in self.kpi_order:
            if kpi_key in self.kpi_results:
                value = self.kpi_results[kpi_key]
                kpi_conf = self.kpi_definitions.get(kpi_key, {})

                name = kpi_conf.get("name", kpi_key.replace("_", " ").title())
                unit = kpi_conf.get("unit", "")

                interpretation, desc = get_kpi_interpretation(kpi_key, value)
                color = get_kpi_color(kpi_key, value)

                # Get confidence interval for this specific KPI
                ci_data = self.kpi_confidence_intervals.get(kpi_key) # Key directly matches kpi_key

                with st_cols[current_col_idx % self.cols_per_row]:
                    display_kpi_card(
                        title=name,
                        value=value,
                        unit=unit,
                        interpretation=interpretation,
                        interpretation_desc=desc,
                        color=color,
                        confidence_interval=ci_data, # Pass CI data to the card utility
                        key_suffix=f"cluster_{kpi_key}"
                    )
                current_col_idx += 1
            else:
                # Optionally display a placeholder or log if a KPI in order is missing from results
                logger.debug(f"KPIClusterDisplay: KPI key '{kpi_key}' from order not found in results. Skipping.")
        logger.debug("KPIClusterDisplay rendering complete.")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Test KPI Cluster Display (Updated for CIs)")

    # Mock data and configurations
    mock_kpi_results = {
        "total_pnl": 10500.75, "win_rate": 65.5, "profit_factor": 2.15,
        "sharpe_ratio": 1.8, "max_drawdown_pct": 15.2, "avg_trade_pnl": 50.20
    }
    # Assuming KPI_CONFIG, get_kpi_interpretation, get_kpi_color are available from imports
    # For standalone test, ensure calculations.py is in PYTHONPATH or provide fallbacks
    try:
        from calculations import get_kpi_interpretation, get_kpi_color
    except ImportError:
        print("Fallback: Defining dummy get_kpi_interpretation and get_kpi_color for kpi_display.py test.")
        def get_kpi_interpretation(kpi_key, value): return f"Interp for {kpi_key}", f"Val: {value:.2f}"
        def get_kpi_color(kpi_key, value): return "#00FF00" if value > 0 else "#FF0000" # Simplified

    mock_kpi_order = ["total_pnl", "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown_pct", "avg_trade_pnl"]
    mock_cis = {
        "total_pnl": (9500.0, 11500.0), # Key matches kpi_key
        "win_rate": (60.0, 70.0),
        "avg_trade_pnl": (45.0, 55.0)
        # Note: 'sharpe_ratio' CI is missing, display_kpi_card should handle ci_data=None
    }

    st.subheader("KPI Cluster with CIs (3 Columns)")
    kpi_cluster_3_cols = KPIClusterDisplay(
        kpi_results=mock_kpi_results,
        kpi_definitions=KPI_CONFIG, # Using imported or placeholder
        kpi_order=mock_kpi_order,
        kpi_confidence_intervals=mock_cis,
        cols_per_row=3
    )
    kpi_cluster_3_cols.render()
    logger.info("KPIClusterDisplay test with CIs complete.")
