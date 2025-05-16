"""
pages/6_🔮_Stochastic_Models.py
Explore stochastic process models. UI/UX enhanced.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

try:
    from config import APP_TITLE, EXPECTED_COLUMNS
    from utils.common_utils import display_custom_message
    from services.analysis_service import AnalysisService
    from plotting import _apply_custom_theme
except ImportError as e:
    st.error(f"Stochastic Models Page Error: Critical module import failed: {e}.")
    APP_TITLE = "TradingDashboard_Error"; logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR: {e}", exc_info=True); st.stop()

logger = logging.getLogger(APP_TITLE)
analysis_service = AnalysisService()

def show_stochastic_models_page():
    st.title("🔮 Stochastic Process Modeling")
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Upload data to access stochastic models.", "info"); return
    
    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    cum_pnl_col = 'cumulative_pnl'

    if filtered_df.empty: display_custom_message("No data matches filters.", "info"); return

    # --- GBM Simulation ---
    st.subheader("Geometric Brownian Motion (GBM) Simulation")
    with st.expander("Configure GBM Simulation", expanded=True):
        if cum_pnl_col not in filtered_df.columns:
            st.warning(f"'{cum_pnl_col}' not found for GBM initial value.")
            s0_gbm_default = 1000.0
        else:
            s0_gbm_default = float(filtered_df[cum_pnl_col].iloc[-1] if not filtered_df[cum_pnl_col].empty else 1000.0)
        
        with st.form("gbm_simulation_form_v3"):
            s0_gbm_input = st.number_input("Initial Value (S0):", value=s0_gbm_default, format="%.2f", key="gbm_s0_v3")
            mu_gbm = st.slider("Annualized Drift (μ):", -0.50, 0.50, 0.05, 0.01, key="gbm_mu_v3", format="%.2f")
            sigma_gbm = st.slider("Annualized Volatility (σ):", 0.01, 1.00, 0.20, 0.01, key="gbm_sigma_v3", format="%.2f")
            n_steps_gbm = st.number_input("Steps (days):", 30, 730, 252, key="gbm_steps_v3")
            n_sims_gbm = st.number_input("Simulations:", 1, 100, 10, key="gbm_sims_v3")
            submit_gbm_button = st.form_submit_button("Simulate GBM Paths")

    if submit_gbm_button: # Button is part of the form, logic runs on submit
        with st.spinner(f"Simulating {n_sims_gbm} GBM paths for {n_steps_gbm} days..."):
            gbm_result = analysis_service.run_gbm_simulation(s0_gbm_input, mu_gbm, sigma_gbm, dt=1/252, n_steps=n_steps_gbm, n_sims=n_sims_gbm)
        
        if gbm_result and 'paths' in gbm_result:
            gbm_paths = gbm_result['paths']
            fig_gbm = go.Figure()
            for i in range(min(n_sims_gbm, 20)): fig_gbm.add_trace(go.Scatter(y=gbm_paths[i,:], mode='lines', name=f'Path {i+1}'))
            fig_gbm.update_layout(title="GBM Simulated Equity Paths", xaxis_title=f"Days (from S0: {s0_gbm_input:.2f})", yaxis_title="Simulated Value")
            st.plotly_chart(_apply_custom_theme(fig_gbm, plot_theme), use_container_width=True)
            st.success("GBM simulation complete!")
        elif gbm_result: display_custom_message(f"GBM Error: {gbm_result.get('error', 'Unknown')}", "error")
        else: display_custom_message("GBM simulation failed.", "error")

    # --- OU Process Fitting ---
    # ... (Similar form/spinner pattern) ...

    # --- Markov Chain Analysis ---
    st.subheader("Markov Chain Analysis for Trade Sequences")
    with st.expander("Configure Markov Chain Analysis", expanded=False):
        if not pnl_col or pnl_col not in filtered_df.columns:
            st.warning(f"PnL column ('{pnl_col}') not found for Markov chain analysis.")
            run_mc_button = False # Disable button
        else:
            with st.form("markov_chain_form_v3"):
                mc_n_states = st.selectbox("Number of States:", [2, 3], index=0, format_func=lambda x: f"{x} (W/L)" if x==2 else f"{x} (W/L/B)", key="mc_states_v3")
                run_mc_button = st.form_submit_button("Analyze Trade Sequence")
    
    if run_mc_button and pnl_col and pnl_col in filtered_df.columns:
        pnl_series_for_mc = filtered_df[pnl_col].dropna()
        if len(pnl_series_for_mc) < 20: display_custom_message("Need >= 20 trades for Markov chain.", "info")
        else:
            with st.spinner("Fitting Markov chain..."):
                mc_results = analysis_service.analyze_markov_chain_trades(pnl_series_for_mc, n_states=mc_n_states)
            if mc_results and 'error' not in mc_results:
                st.success("Markov chain analysis complete!")
                st.write(f"**State Labels:** `{mc_results.get('state_labels', [])}`")
                if 'transition_matrix' in mc_results:
                    st.write("**Transition Matrix (P(Row -> Col)):**")
                    st.dataframe(pd.DataFrame(mc_results['transition_matrix']).style.format("{:.3%}").background_gradient(cmap='Blues', axis=1))
                # ... (display initial distribution) ...
                # TODO: Add heatmap visualization for transition matrix
                if 'transition_matrix' in mc_results:
                    tm_df = pd.DataFrame(mc_results['transition_matrix'])
                    fig_tm = go.Figure(data=go.Heatmap(
                               z=tm_df.values,
                               x=tm_df.columns,
                               y=tm_df.index,
                               colorscale='Blues',
                               text=tm_df.applymap(lambda x: f"{x:.2%}").values, # Format text as percentage
                               texttemplate="%{text}",
                               hoverongaps=False))
                    fig_tm.update_layout(title="Transition Matrix Heatmap", xaxis_title="To State", yaxis_title="From State")
                    st.plotly_chart(_apply_custom_theme(fig_tm, plot_theme), use_container_width=True)

            elif mc_results: display_custom_message(f"Markov Error: {mc_results.get('error')}", "error")
            else: display_custom_message("Markov analysis failed.", "error")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_stochastic_models_page()
