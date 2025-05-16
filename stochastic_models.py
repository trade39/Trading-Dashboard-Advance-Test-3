"""
stochastic_models.py

Implementations or wrappers for stochastic process models relevant to trading,
such as Geometric Brownian Motion (GBM) for equity curve simulation,
Ornstein-Uhlenbeck for mean-reverting series, Jump-Diffusion models,
and Markov chains for modeling trade sequences or market regimes.
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any, Optional

# Assuming config.py is in the root directory if needed for constants
from config import MARKOV_MAX_LAG, EXPECTED_COLUMNS

import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@st.cache_data(show_spinner="Simulating Geometric Brownian Motion...", ttl=3600)
def simulate_gbm(
    s0: float, mu: float, sigma: float, dt: float, n_steps: int, n_sims: int = 1
) -> np.ndarray:
    """
    Simulates stock price paths using Geometric Brownian Motion (GBM).
    S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z is a standard normal random variable.

    Args:
        s0: Initial stock price.
        mu: Annualized drift (expected return).
        sigma: Annualized volatility.
        dt: Time step (e.g., 1/252 for daily steps if mu, sigma are annualized).
        n_steps: Number of time steps to simulate.
        n_sims: Number of simulation paths.

    Returns:
        np.ndarray: Array of simulated price paths (n_sims x n_steps+1).
    """
    if s0 <= 0 or sigma < 0 or dt <= 0 or n_steps <= 0:
        logger.error("Invalid parameters for GBM simulation.")
        return np.array([[]]) # Return empty for error

    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = s0

    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_sims)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z
        paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)

    return paths

@st.cache_data(show_spinner="Fitting Ornstein-Uhlenbeck process...", ttl=3600)
def fit_ornstein_uhlenbeck(
    series: pd.Series
) -> Optional[Dict[str, float]]:
    """
    Estimates parameters (theta, mu, sigma) for an Ornstein-Uhlenbeck process.
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    This is a simplified estimation using linear regression on the discretized form:
    X(t+1) - X(t) = theta*mu*dt - theta*X(t)*dt + sigma*sqrt(dt)*epsilon
    Let y = X(t+1) - X(t) and x = X(t).
    y = beta_0 + beta_1 * x + error
    beta_1 = -theta*dt  => theta = -beta_1/dt
    beta_0 = theta*mu*dt => mu = beta_0 / (theta*dt) = beta_0 / (-beta_1)
    sigma = std(residuals) / sqrt(dt)

    Args:
        series: Pandas Series representing the time series data. Assumes dt=1 if not specified.

    Returns:
        Optional[Dict[str, float]]: Dictionary with 'theta' (mean reversion speed),
                                     'mu' (long-term mean), 'sigma' (volatility of noise),
                                     or None if fitting fails.
    """
    series = series.dropna()
    if len(series) < 10: # Need sufficient data
        logger.warning("Ornstein-Uhlenbeck: Not enough data points to fit.")
        return None

    dt = 1 # Assuming dt=1 for simplicity (e.g., daily data)
           # If dt is different, it should be passed or inferred.

    y = series.diff().dropna()
    x = series[:-1].loc[y.index] # Align x with y

    if len(y) < 2:
        logger.warning("Ornstein-Uhlenbeck: Not enough data after differencing.")
        return None

    try:
        # Using statsmodels for regression to get more details if needed
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        beta_0, beta_1 = model.params

        if dt == 0: return None # Avoid division by zero
        
        theta = -beta_1 / dt
        mu = beta_0 / (-beta_1) if beta_1 != 0 else np.mean(series) # if beta_1 is 0, no mean reversion to mu from regression
        
        residuals = model.resid
        sigma = np.std(residuals) / np.sqrt(dt)

        # Constraints: theta > 0 for mean reversion
        if theta < 0:
            logger.warning(f"OU fit resulted in theta < 0 ({theta:.4f}), indicating non-mean-reverting behavior under this model.")
            # Return None or the parameters with a warning. For now, return them.
            # return None

        return {"theta": theta, "mu": mu, "sigma_ou": sigma, "dt_assumed": dt}

    except Exception as e:
        logger.error(f"Error fitting Ornstein-Uhlenbeck process: {e}", exc_info=True)
        return None


@st.cache_data(show_spinner="Simulating Jump-Diffusion process (Merton model)...", ttl=3600)
def simulate_merton_jump_diffusion(
    s0: float, mu: float, sigma: float, # GBM parameters
    lambda_jump: float, mu_jump: float, sigma_jump: float, # Jump parameters
    dt: float, n_steps: int, n_sims: int = 1
) -> np.ndarray:
    """
    Simulates price paths using Merton's Jump-Diffusion model.
    Combines GBM with Poisson-driven jumps.
    S(t+dt) = S(t) * exp((mu - 0.5*sigma^2 - lambda_jump*k)*dt + sigma*sqrt(dt)*Z) * J
    where k = E[J-1] = exp(mu_jump + 0.5*sigma_jump^2) - 1 (compensator for jump risk)
    J is the jump size, log(J) ~ N(mu_jump, sigma_jump^2)
    Number of jumps in dt follows Poisson(lambda_jump*dt)

    Args:
        s0, mu, sigma: Parameters for the GBM component.
        lambda_jump: Average number of jumps per year.
        mu_jump: Mean of the log-jump size.
        sigma_jump: Standard deviation of the log-jump size.
        dt, n_steps, n_sims: Simulation parameters.

    Returns:
        np.ndarray: Array of simulated price paths.
    """
    if s0 <=0 or sigma < 0 or lambda_jump < 0 or sigma_jump < 0 or dt <= 0 or n_steps <=0:
        logger.error("Invalid parameters for Merton Jump-Diffusion simulation.")
        return np.array([[]])

    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = s0

    # Compensator for the jump part to ensure martingale property under risk-neutral measure (if mu is risk-neutral)
    # Here, mu is the actual drift. The term adjusts the drift for the expected impact of jumps.
    k = np.exp(mu_jump + 0.5 * sigma_jump**2) - 1
    adjusted_drift = mu - lambda_jump * k

    for t in range(1, n_steps + 1):
        # GBM part
        z_gbm = np.random.standard_normal(n_sims)
        gbm_drift = (adjusted_drift - 0.5 * sigma**2) * dt
        gbm_diffusion = sigma * np.sqrt(dt) * z_gbm
        paths[:, t] = paths[:, t-1] * np.exp(gbm_drift + gbm_diffusion)

        # Jump part
        poisson_arrivals = np.random.poisson(lambda_jump * dt, n_sims)
        for sim_idx in range(n_sims):
            if poisson_arrivals[sim_idx] > 0:
                for _ in range(poisson_arrivals[sim_idx]): # Can have multiple jumps in one dt
                    jump_size_log = np.random.normal(mu_jump, sigma_jump)
                    paths[sim_idx, t] *= np.exp(jump_size_log)
    return paths


@st.cache_data(show_spinner="Fitting Markov chain to trade sequences...", ttl=3600)
def fit_markov_chain_trade_sequence(
    pnl_series: pd.Series,
    n_states: int = 2, # e.g., 2 for Win/Loss, 3 for Win/Loss/Breakeven
    max_lag: int = MARKOV_MAX_LAG
) -> Optional[Dict[str, Any]]:
    """
    Fits a Markov chain to a sequence of trade outcomes (e.g., Win/Loss).
    Estimates transition probabilities.

    Args:
        pnl_series: Series of PnL values for trades.
        n_states: Number of states to define from PnL (e.g., 2 for W/L).
        max_lag: Maximum lag to consider for state definition (currently supports 1).

    Returns:
        Optional[Dict[str, Any]]: Dictionary with 'states', 'transition_matrix',
                                   'initial_distribution', or None if fitting fails.
    """
    pnl_series = pnl_series.dropna()
    if len(pnl_series) < 5: # Need some data
        logger.warning("Markov Chain: Not enough PnL data.")
        return None

    # Define states based on PnL
    if n_states == 2: # Win (W), Loss (L)
        states = np.array(['W'] * len(pnl_series))
        states[pnl_series <= 0] = 'L' # Breakeven treated as Loss for 2-state
        state_labels = ['L', 'W']
    elif n_states == 3: # Win (W), Loss (L), Breakeven (B)
        states = np.array(['B'] * len(pnl_series))
        states[pnl_series > 0] = 'W'
        states[pnl_series < 0] = 'L'
        state_labels = ['L', 'B', 'W']
    else:
        logger.error(f"Markov Chain: Unsupported number of states {n_states}. Choose 2 or 3.")
        return None
    
    num_distinct_states = len(state_labels)

    if max_lag != 1:
        logger.warning("Markov Chain: Currently only supports max_lag=1 for simple transition matrix.")
        # Higher order Markov chains are more complex to estimate and represent.

    # Estimate transition matrix P_ij = P(S_t = j | S_{t-1} = i)
    transition_counts = pd.DataFrame(
        np.zeros((num_distinct_states, num_distinct_states)),
        index=state_labels, columns=state_labels
    )

    for i in range(len(states) - 1):
        prev_state = states[i]
        current_state = states[i+1]
        transition_counts.loc[prev_state, current_state] += 1

    # Normalize counts to get probabilities
    transition_matrix = transition_counts.apply(lambda row: row / row.sum() if row.sum() > 0 else 0, axis=1)
    
    # Initial state distribution (empirical)
    initial_dist_counts = pd.Series(states).value_counts()
    initial_distribution = (initial_dist_counts / initial_dist_counts.sum()).reindex(state_labels).fillna(0)

    return {
        "state_labels": state_labels,
        "trade_states_sequence": states.tolist(),
        "transition_matrix": transition_matrix.to_dict('index'),
        "transition_counts": transition_counts.to_dict('index'),
        "initial_distribution": initial_distribution.to_dict()
    }


if __name__ == '__main__':
    logger.info("--- Testing Stochastic Models ---")

    # 1. GBM Simulation
    logger.info("\n--- GBM Simulation ---")
    gbm_paths = simulate_gbm(s0=100, mu=0.05, sigma=0.2, dt=1/252, n_steps=252, n_sims=3)
    if gbm_paths.size > 0:
        logger.info(f"GBM Simulation (3 paths, 252 steps) - End prices: {gbm_paths[:, -1]}")
    else:
        logger.error("GBM simulation failed.")

    # 2. Ornstein-Uhlenbeck Fitting
    logger.info("\n--- Ornstein-Uhlenbeck Fitting ---")
    # Simulate some OU data: dX = -0.5 * (0 - X) dt + 0.1 dW  (theta=0.5, mu=0, sigma=0.1)
    ou_test_data = [0]
    for _ in range(200):
        ou_test_data.append(ou_test_data[-1] + 0.5 * (0 - ou_test_data[-1]) * (1/252) + 0.1 * np.sqrt(1/252) * np.random.randn())
    ou_series = pd.Series(ou_test_data)
    ou_params = fit_ornstein_uhlenbeck(ou_series)
    if ou_params:
        logger.info(f"OU Fitted Parameters: theta={ou_params['theta']:.4f}, mu={ou_params['mu']:.4f}, sigma_ou={ou_params['sigma_ou']:.4f}")
    else:
        logger.warning("OU fitting failed or returned None.")

    # 3. Merton Jump-Diffusion Simulation
    logger.info("\n--- Merton Jump-Diffusion Simulation ---")
    merton_paths = simulate_merton_jump_diffusion(
        s0=100, mu=0.05, sigma=0.2,
        lambda_jump=0.5, mu_jump=-0.05, sigma_jump=0.1,
        dt=1/252, n_steps=252, n_sims=2
    )
    if merton_paths.size > 0:
        logger.info(f"Merton Simulation (2 paths, 252 steps) - End prices: {merton_paths[:, -1]}")
    else:
        logger.error("Merton simulation failed.")

    # 4. Markov Chain for Trade Sequences
    logger.info("\n--- Markov Chain for Trade Sequences ---")
    sample_pnl_trades = pd.Series([10, -5, 2, 8, -3, -6, 5, 1, -2, 0, 7, -4])
    markov_results_2state = fit_markov_chain_trade_sequence(sample_pnl_trades, n_states=2)
    if markov_results_2state:
        logger.info(f"Markov Chain (2 states - W/L) Transition Matrix:\n{pd.DataFrame(markov_results_2state['transition_matrix']).round(3)}")
        logger.info(f"Initial Distribution: {markov_results_2state['initial_distribution']}")
    else:
        logger.warning("Markov chain (2-state) fitting failed.")

    markov_results_3state = fit_markov_chain_trade_sequence(sample_pnl_trades, n_states=3)
    if markov_results_3state:
        logger.info(f"Markov Chain (3 states - W/L/B) Transition Matrix:\n{pd.DataFrame(markov_results_3state['transition_matrix']).round(3)}")
    else:
        logger.warning("Markov chain (3-state) fitting failed.")

    logger.info("--- Stochastic Models Testing Complete ---")
