"""
plotting.py

Contains functions to generate various interactive Plotly visualizations
for the Trading Performance Dashboard.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Any

# Assuming config.py is in the root directory
from config import (
    COLORS, PLOTLY_THEME_DARK, PLOTLY_THEME_LIGHT,
    PLOT_BG_COLOR_DARK, PLOT_PAPER_BG_COLOR_DARK, PLOT_FONT_COLOR_DARK,
    PLOT_BG_COLOR_LIGHT, PLOT_PAPER_BG_COLOR_LIGHT, PLOT_FONT_COLOR_LIGHT,
    PLOT_LINE_COLOR, PLOT_MARKER_PROFIT_COLOR, PLOT_MARKER_LOSS_COLOR,
    EXPECTED_COLUMNS, APP_TITLE
)

import logging
logger = logging.getLogger(APP_TITLE) # Get the main app logger


def _apply_custom_theme(fig: go.Figure, theme: str = 'dark') -> go.Figure:
    """Applies custom theme settings to a Plotly figure."""
    plotly_theme_template = PLOTLY_THEME_DARK if theme == 'dark' else PLOTLY_THEME_LIGHT
    bg_color = PLOT_BG_COLOR_DARK if theme == 'dark' else PLOT_BG_COLOR_LIGHT
    paper_bg_color = PLOT_PAPER_BG_COLOR_DARK if theme == 'dark' else PLOT_PAPER_BG_COLOR_LIGHT
    font_color = PLOT_FONT_COLOR_DARK if theme == 'dark' else PLOT_FONT_COLOR_LIGHT
    grid_color = COLORS.get('gray', '#808080') if theme == 'dark' else '#e0e0e0'

    fig.update_layout(
        template=plotly_theme_template,
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_bg_color,
        font_color=font_color,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=False),
        hoverlabel=dict(
            bgcolor=COLORS.get('card_background_dark', '#273334') if theme == 'dark' else COLORS.get('card_background_light', '#F0F2F6'),
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor=COLORS.get('royal_blue')
        )
    )
    return fig

def plot_equity_curve_and_drawdown(
    df: pd.DataFrame,
    date_col: str = EXPECTED_COLUMNS['date'],
    cumulative_pnl_col: str = 'cumulative_pnl',
    drawdown_pct_col: Optional[str] = 'drawdown_pct',
    theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty or date_col not in df.columns or cumulative_pnl_col not in df.columns:
        logger.warning("Equity curve plot: Data is insufficient.")
        return None
    if drawdown_pct_col and drawdown_pct_col not in df.columns:
        logger.warning(f"Drawdown column '{drawdown_pct_col}' not found. Plotting equity curve only.")
        drawdown_pct_col = None

    fig_rows = 2 if drawdown_pct_col else 1
    row_heights = [0.7, 0.3] if drawdown_pct_col else [1.0]
    subplot_titles = ("Equity Curve", "Drawdown (%)" if drawdown_pct_col else None)

    fig = make_subplots(
        rows=fig_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=[s for s in subplot_titles if s]
    )

    fig.add_trace(
        go.Scatter(x=df[date_col], y=df[cumulative_pnl_col], mode='lines', name='Equity Curve', line=dict(color=PLOT_LINE_COLOR, width=2)),
        row=1, col=1
    )

    if drawdown_pct_col:
        fig.add_trace(
            go.Scatter(x=df[date_col], y=df[drawdown_pct_col], mode='lines', name='Drawdown (%)', line=dict(color=COLORS.get('red', '#FF0000'), width=1.5), fill='tozeroy', fillcolor='rgba(255,0,0,0.2)'),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, tickformat=".2f")
        min_dd, max_dd = df[drawdown_pct_col].min(), df[drawdown_pct_col].max()
        if pd.isna(min_dd) or pd.isna(max_dd) or (min_dd == 0 and max_dd == 0) :
             fig.update_yaxes(range=[-1, 1], row=2, col=1) # Default range if no drawdown or NaN

    fig.update_layout(title_text='Equity Performance and Drawdown', hovermode='x unified')
    fig.update_yaxes(title_text="Cumulative PnL", row=1, col=1)
    return _apply_custom_theme(fig, theme)


def plot_pnl_distribution(
    df: pd.DataFrame, pnl_col: str = EXPECTED_COLUMNS['pnl'],
    title: str = "PnL Distribution (per Trade)", theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty or pnl_col not in df.columns or df[pnl_col].dropna().empty:
        logger.warning("PnL distribution plot: Data is insufficient.")
        return None
    fig = px.histogram(df, x=pnl_col, nbins=50, title=title, marginal="box", color_discrete_sequence=[PLOT_LINE_COLOR])
    fig.update_layout(xaxis_title="PnL per Trade", yaxis_title="Frequency")
    return _apply_custom_theme(fig, theme)

def plot_time_series_decomposition(
    decomposition_result: Any, title: str = "Time Series Decomposition", theme: str = 'dark'
) -> Optional[go.Figure]:
    if decomposition_result is None:
        logger.warning("Time series decomposition plot: No decomposition result provided.")
        return None
    try:
        observed = getattr(decomposition_result, 'observed', pd.Series())
        trend = getattr(decomposition_result, 'trend', pd.Series())
        seasonal = getattr(decomposition_result, 'seasonal', pd.Series())
        resid = getattr(decomposition_result, 'resid', pd.Series())
        
        if observed.empty:
            logger.warning("Time series decomposition plot: Observed series is empty.")
            return None
            
        x_axis = observed.index
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
        fig.add_trace(go.Scatter(x=x_axis, y=observed, mode='lines', name='Observed', line=dict(color=PLOT_LINE_COLOR)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=trend, mode='lines', name='Trend', line=dict(color=COLORS.get('green', '#00FF00'))), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=seasonal, mode='lines', name='Seasonal', line=dict(color=COLORS.get('royal_blue', '#4169E1'))), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=resid, mode='lines+markers', name='Residual', line=dict(color=COLORS.get('gray', '#808080')), marker=dict(size=3)), row=4, col=1)
        fig.update_layout(title_text=title, height=700, showlegend=False)
        return _apply_custom_theme(fig, theme)
    except AttributeError as e:
        logger.error(f"Decomposition result missing attributes: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error plotting decomposition: {e}", exc_info=True)
        return None

def plot_value_over_time(
    series: pd.Series, series_name: str, title: Optional[str] = None,
    x_axis_title: str = "Date / Time", y_axis_title: Optional[str] = None,
    theme: str = 'dark', line_color: str = PLOT_LINE_COLOR
) -> Optional[go.Figure]:
    if series is None or series.empty:
        logger.warning(f"Plot value over time for '{series_name}': Series is empty.")
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=series_name, line=dict(color=line_color)))
    fig.update_layout(title_text=title if title else series_name, xaxis_title=x_axis_title, yaxis_title=y_axis_title if y_axis_title else series_name)
    return _apply_custom_theme(fig, theme)

def plot_pnl_by_category(
    df: pd.DataFrame, category_col: str, pnl_col: str = EXPECTED_COLUMNS['pnl'],
    title_prefix: str = "Total PnL by", theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty or category_col not in df.columns or pnl_col not in df.columns:
        logger.warning(f"PnL by category plot: Data insufficient or missing columns ('{category_col}', '{pnl_col}').")
        return None
    grouped_pnl = df.groupby(category_col)[pnl_col].sum().reset_index().sort_values(by=pnl_col, ascending=False)
    fig = px.bar(grouped_pnl, x=category_col, y=pnl_col, title=f"{title_prefix} {category_col.replace('_', ' ').title()}",
                 color=pnl_col, color_continuous_scale=[COLORS.get('red', '#FF0000'), COLORS.get('gray', '#808080'), COLORS.get('green', '#00FF00')])
    fig.update_layout(xaxis_title=category_col.replace('_', ' ').title(), yaxis_title="Total PnL")
    return _apply_custom_theme(fig, theme)

def plot_win_rate_analysis(
    df: pd.DataFrame, category_col: str, win_col: str = 'win',
    title_prefix: str = "Win Rate by", theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty or category_col not in df.columns or win_col not in df.columns:
        logger.warning(f"Win rate analysis plot: Data insufficient or missing columns ('{category_col}', '{win_col}').")
        return None
    category_counts = df.groupby(category_col).size().rename('total_trades_in_cat')
    category_wins = df[df[win_col] == True].groupby(category_col).size().rename('wins_in_cat')
    win_rate_df = pd.concat([category_counts, category_wins], axis=1).fillna(0)
    win_rate_df['win_rate_pct'] = (win_rate_df['wins_in_cat'] / win_rate_df['total_trades_in_cat'] * 100).fillna(0)
    win_rate_df = win_rate_df.reset_index().sort_values(by='win_rate_pct', ascending=False)
    fig = px.bar(win_rate_df, x=category_col, y='win_rate_pct', title=f"{title_prefix} {category_col.replace('_', ' ').title()}",
                 color='win_rate_pct', color_continuous_scale=px.colors.sequential.Greens)
    fig.update_layout(xaxis_title=category_col.replace('_', ' ').title(), yaxis_title="Win Rate (%)", yaxis_ticksuffix="%")
    return _apply_custom_theme(fig, theme)

def plot_rolling_performance(
    df: pd.DataFrame, date_col: str, metric_series: pd.Series, metric_name: str,
    title: Optional[str] = None, theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty or date_col not in df.columns or metric_series.empty:
        logger.warning(f"Rolling performance plot for '{metric_name}': Data insufficient.")
        return None
    plot_x_data = df[date_col] if len(df[date_col]) == len(metric_series) else metric_series.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_x_data, y=metric_series, mode='lines', name=metric_name, line=dict(color=PLOT_LINE_COLOR)))
    fig.update_layout(title_text=title if title else f"Rolling {metric_name}",
                      xaxis_title="Date" if date_col in df.columns and len(df[date_col]) == len(metric_series) else "Trade Number / Period",
                      yaxis_title=metric_name)
    return _apply_custom_theme(fig, theme)

def plot_correlation_matrix(
    df: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
    title: str = "Correlation Matrix of Numeric Features", theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty:
        logger.warning("Correlation matrix plot: DataFrame is empty.")
        return None
    df_numeric = df[numeric_cols].copy() if numeric_cols else df.select_dtypes(include=np.number)
    if df_numeric.empty or df_numeric.shape[1] < 2:
        logger.warning("Correlation matrix plot: Not enough numeric columns (need at least 2).")
        return None
    corr_matrix = df_numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale='RdBu', zmin=-1, zmax=1, text=corr_matrix.round(2).astype(str),
        texttemplate="%{text}", hoverongaps=False ))
    fig.update_layout(title_text=title)
    return _apply_custom_theme(fig, theme)

def plot_bootstrap_distribution_and_ci(
    bootstrap_statistics: List[float],
    observed_statistic: float,
    lower_bound: float,
    upper_bound: float,
    statistic_name: str,
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Plots the distribution of bootstrap statistics with observed value and CIs.
    """
    if not bootstrap_statistics:
        logger.warning(f"Bootstrap distribution plot for '{statistic_name}': No bootstrap statistics provided.")
        return None

    fig = go.Figure()

    # Histogram of bootstrap statistics
    fig.add_trace(go.Histogram(
        x=bootstrap_statistics,
        name='Bootstrap<br>Distribution',
        marker_color=COLORS.get('royal_blue', '#4169E1'),
        opacity=0.75,
        histnorm='probability density' # Normalize for better comparison if ranges differ
    ))

    # Vertical line for observed statistic
    fig.add_vline(
        x=observed_statistic, line_width=2, line_dash="dash",
        line_color=COLORS.get('green', '#00FF00'),
        name=f'Observed<br>{statistic_name}<br>({observed_statistic:.4f})'
    )

    # Vertical lines for CI bounds
    fig.add_vline(
        x=lower_bound, line_width=2, line_dash="dot",
        line_color=COLORS.get('orange', '#FFA500'),
        name=f'Lower 95% CI<br>({lower_bound:.4f})'
    )
    fig.add_vline(
        x=upper_bound, line_width=2, line_dash="dot",
        line_color=COLORS.get('orange', '#FFA500'),
        name=f'Upper 95% CI<br>({upper_bound:.4f})'
    )
    
    fig.update_layout(
        title_text=f'Bootstrap Distribution for {statistic_name}',
        xaxis_title=statistic_name,
        yaxis_title='Density',
        bargap=0.1,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig, theme)
