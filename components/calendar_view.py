"""
components/calendar_view.py

This component aims to provide a P&L calendar heatmap view.
Due to Streamlit's native limitations for complex calendar heatmaps like GitHub's
contribution graph, this implementation will use Plotly to create a
heatmap of daily PnL over a year, which is a common way to visualize this.
A true interactive day-by-day calendar might require custom HTML/JS components.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Any

try:
    from config import EXPECTED_COLUMNS, COLORS, APP_TITLE
    from utils.common_utils import format_currency # For tooltips
except ImportError:
    print("Warning (calendar_view.py): Could not import from root config/utils. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default"
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}
    COLORS = {"green": "#00FF00", "red": "#FF0000", "gray": "#808080", "royal_blue": "#4169E1",
              "dark_background": "#1C2526", "text_dark": "#E0E0E0",
              "light_background": "#FFFFFF", "text_light": "#000000"}
    def format_currency(value, currency_symbol="$", decimals=2):
        if pd.isna(value): return "N/A"
        return f"{currency_symbol}{value:.{decimals}f}"

import logging
logger = logging.getLogger(APP_TITLE)

class PnLCalendarComponent:
    """
    Component to display a P&L calendar-style heatmap.
    This creates a heatmap of daily PnL, typically for a selected year.
    """
    def __init__(
        self,
        daily_pnl_df: pd.DataFrame,
        year: Optional[int] = None,
        date_col: str = 'date',
        pnl_col: str = 'pnl',
        plot_theme: str = 'dark'
    ):
        self.daily_pnl_df = daily_pnl_df
        self.year = year
        self.date_col = date_col
        self.pnl_col = pnl_col
        self.plot_theme = plot_theme
        self.calendar_data = self._prepare_calendar_data()
        logger.debug("PnLCalendarComponent initialized.")

    def _prepare_calendar_data(self) -> Optional[pd.DataFrame]:
        if self.daily_pnl_df is None or self.daily_pnl_df.empty:
            logger.warning("PnLCalendar: daily_pnl_df is empty.")
            return None
        if self.date_col not in self.daily_pnl_df.columns or self.pnl_col not in self.daily_pnl_df.columns:
            logger.error(f"PnLCalendar: Missing required columns '{self.date_col}' or '{self.pnl_col}'.")
            return None

        df = self.daily_pnl_df.copy()
        try:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        except Exception as e:
            logger.error(f"PnLCalendar: Could not convert date column to datetime: {e}")
            return None

        df = df.set_index(self.date_col)

        if self.year is None:
            self.year = df.index.max().year if not df.index.empty else pd.Timestamp.now().year
        
        start_date = pd.Timestamp(f'{self.year}-01-01')
        end_date = pd.Timestamp(f'{self.year}-12-31')
        all_days_in_year = pd.date_range(start_date, end_date, freq='D')
        
        calendar_df = df.reindex(all_days_in_year, fill_value=0).reset_index() # Use df directly
        calendar_df = calendar_df.rename(columns={'index': 'date', self.pnl_col: 'pnl'}) # Ensure 'pnl' is the target column name

        calendar_df['week_of_year'] = calendar_df['date'].dt.isocalendar().week.astype(int)
        calendar_df['day_of_week'] = calendar_df['date'].dt.dayofweek # Monday=0, Sunday=6
        calendar_df['month_text'] = calendar_df['date'].dt.strftime('%b')
        calendar_df['day_of_month'] = calendar_df['date'].dt.day

        # Adjust week numbers at year boundaries for consistent plotting within one year
        if not calendar_df.empty:
            if calendar_df['date'].iloc[0].month == 1 and calendar_df['week_of_year'].iloc[0] >= 52:
                calendar_df.loc[calendar_df['week_of_year'] >= 52, 'week_of_year'] = 0
            if calendar_df['date'].iloc[-1].month == 12 and calendar_df['week_of_year'].iloc[-1] == 1:
                 # If Dec dates fall into week 1 of next year, map them to the last week of current year (e.g. 53)
                 max_week_current_year = calendar_df.loc[calendar_df['date'].dt.month == 12, 'week_of_year'].max()
                 if max_week_current_year < 50: max_week_current_year = 52 # Ensure it's a high number
                 calendar_df.loc[(calendar_df['date'].dt.month == 12) & (calendar_df['week_of_year'] == 1), 'week_of_year'] = max_week_current_year + 1


        return calendar_df

    def _get_plotly_theme_layout(self) -> Dict:
        # ... (same as before)
        if self.plot_theme == 'dark':
            return {
                'plot_bgcolor': COLORS.get('dark_background', '#1C2526'),
                'paper_bgcolor': COLORS.get('dark_background', '#1C2526'),
                'font_color': COLORS.get('text_dark', '#E0E0E0'),
                'xaxis': dict(showgrid=False, zeroline=False),
                'yaxis': dict(showgrid=False, zeroline=False),
            }
        else: # light theme
            return {
                'plot_bgcolor': COLORS.get('light_background', '#FFFFFF'),
                'paper_bgcolor': COLORS.get('light_background', '#FFFFFF'),
                'font_color': COLORS.get('text_light', '#000000'),
                'xaxis': dict(showgrid=False, zeroline=False),
                'yaxis': dict(showgrid=False, zeroline=False),
            }

    def render(self) -> None:
        st.subheader(f"P&L Calendar Heatmap for {self.year}")
        if self.calendar_data is None or self.calendar_data.empty:
            st.info(f"No P&L data available to display for the year {self.year}.")
            return

        z_data = np.full((7, 53), np.nan)
        hover_text_data = [[None for _ in range(53)] for _ in range(7)]

        for _, row in self.calendar_data.iterrows():
            day_idx = int(row['day_of_week'])
            week_idx = int(row['week_of_year']) # This should be 0-indexed for array
            if 0 <= day_idx < 7 and 0 <= week_idx < 53:
                z_data[day_idx, week_idx] = row['pnl']
                hover_text_data[day_idx][week_idx] = (
                    f"Date: {row['date'].strftime('%Y-%m-%d')} ({row['date'].strftime('%a')})<br>"
                    f"PnL: {format_currency(row['pnl'])}<br>"
                    f"Day: {row['day_of_month']}<br>"
                    f"Week: {week_idx + 1}" # Display week as 1-indexed
                )
        
        min_pnl, max_pnl = np.nanmin(z_data), np.nanmax(z_data)
        if pd.isna(min_pnl) or pd.isna(max_pnl) : min_pnl, max_pnl = 0,0 # Handle all NaN case
        abs_max = max(abs(min_pnl), abs(max_pnl), 0.01)

        colorscale = [
            [0.0, COLORS.get('red', '#FF0000')],
            [0.5, COLORS.get('gray', '#DDDDDD') if self.plot_theme == 'light' else '#444444'],
            [1.0, COLORS.get('green', '#00FF00')]
        ]

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=list(range(1, 54)), # Weeks 1 to 53 for x-axis labels
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            colorscale=colorscale,
            zmin=-abs_max, zmax=abs_max,
            xgap=2, ygap=2,
            showscale=True,
            # Fixed: 'titleside' to 'title.side' or just 'title.text'
            colorbar=dict(title=dict(text='Daily PnL', side='right')),
            customdata=hover_text_data,
            hovertemplate="%{customdata}<extra></extra>"
        ))
        
        # Month labels logic (can be kept as is or refined)
        month_positions = []
        last_month = ""
        # Ensure calendar_data is not empty before trying to access iloc
        if not self.calendar_data.empty:
            for week_num_idx in range(53): # Iterate through 0-indexed weeks
                week_data = self.calendar_data[self.calendar_data['week_of_year'] == week_num_idx]
                if not week_data.empty:
                    current_month_in_week = week_data['month_text'].iloc[0]
                    if current_month_in_week != last_month:
                        month_positions.append({'week_label': week_num_idx + 1, 'month': current_month_in_week}) # week_label is 1-indexed for plot
                        last_month = current_month_in_week
        
        for pos in month_positions:
             fig.add_annotation(
                x=pos['week_label'], y=-0.7, # Use week_label for x position
                text=pos['month'], showarrow=False,
                font=dict(size=10, color=self._get_plotly_theme_layout()['font_color']),
                xanchor='center'
            )

        fig.update_layout(
            title_text=f'Daily P&L Heatmap - {self.year}',
            xaxis_title='Week of Year', yaxis_title='Day of Week',
            yaxis_autorange='reversed', height=350,
            margin=dict(t=50, l=50, b=80, r=30),
            **self._get_plotly_theme_layout()
        )
        fig.update_xaxes(side="top", tickvals=[w for w in range(1,54,4)])
        st.plotly_chart(fig, use_container_width=True)

# (No __main__ block needed here as it's a component)
