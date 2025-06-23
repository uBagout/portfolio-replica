import numpy as np

def apply_constraints(original_weights, constraint_funcs, weights_history, replica_returns, constraints_history, 
                      constraint_params={
    'constraint_gross_exposure': [2.0],
    'constraint_var_historical': [0.08, 0.01, 4],
    'constraint_turnover_band': [0.02, 0.10]
                      }):
    total_rescale = 1.0
    if constraint_funcs is not None:
        for fn in constraint_funcs:
            original_weights, metadata = fn(original_weights, weights_history, replica_returns, **constraint_params.get(fn.__name__, {}))
            constraints_history[fn.__name__].append(metadata)
            total_rescale *= metadata.get('rescale_factor')
    return original_weights, total_rescale


def constraint_gross_exposure(weights, weights_history, replica_returns, max_gross=2):
    """
    Constraint to limit the gross exposure of the portfolio.
    If the gross exposure exceeds `max_gross`, rescale the weights.

    Parameters:
    - weights: Current weights of the portfolio.
    - weights_history: Historical weights of the portfolio.
    - replica_returns: Returns of the replica assets.
    - max_gross: Maximum allowed gross exposure (default is 2).

    Returns:
    - Tuple containing:
        - Weights after applying constraints.
        - Metadata dictionary with information about the constraint application.
    """
    gross = np.sum(np.abs(weights))
    metadata = {
        'activated': False,
        'violation_amount': 0.0,
        'rescale_factor': 1.0,
        'gross_exposure': gross
    }
    if gross > max_gross:
        metadata['activated'] = True
        metadata['violation_amount'] = gross - max_gross
        rescale = max_gross / gross
        weights = weights * rescale
        metadata['rescale_factor'] = rescale

        #segnare che gross exposure diventa max_gross?
    return weights, metadata

def constraint_turnover_band(
    weights: np.ndarray,
    weights_history: list,
    replica_returns: list,
    min_turnover: float = 0.02,
    max_turnover: float = 0.10
) -> tuple[np.ndarray, dict]:
    """
    Enforce both a minimum no-trade threshold and a maximum turnover cap.

    Parameters
    ----------
    weights : np.ndarray
        Proposed new weights.
    weights_history : list of np.ndarray
        Past weights; last entry is w_{t-1}.
    replica_returns : list
        Past returns (unused here).
    min_turnover : float
        Below this turnover, skip rebalancing entirely (no-trade band).
    max_turnover : float
        Above this turnover, scale changes down to this level.

    Returns
    -------
    new_weights, metadata : (np.ndarray, dict)
    """
    metadata = {
        'activated_min': False,
        'activated_max': False,
        'turnover': 0.0,
        'viol_min': 0.0,
        'viol_max': 0.0,
        'rescale_factor': 1.0
    }
    if not weights_history:
        return weights, metadata

    prev = weights_history[-1]
    turnover = np.sum(np.abs(weights - prev))
    metadata['turnover'] = turnover

    # No-trade band
    if turnover < min_turnover:
        metadata['activated_min'] = True
        metadata['viol_min'] = min_turnover - turnover
        return prev.copy(), metadata  # stay at old weights

    # Max-turnover cap
    if turnover > max_turnover:
        metadata['activated_max'] = True
        metadata['viol_max'] = turnover - max_turnover
        factor = max_turnover / turnover
        weights = prev + factor * (weights - prev)
        metadata['rescale_factor'] = factor

    return weights, metadata


import numpy as np
from scipy.stats import norm


def calculate_var(returns, method: str, confidence: float = 0.01, horizon: int = 4) -> float:
    """
    Calculate Value at Risk (VaR) over a given time horizon.

    Parameters
    ----------
    returns : array-like
        Historical returns (e.g. weekly P&L series).
    method : {'gaussian', 'historical', 'cornish-fisher'}
        VaR calculation method.
    confidence : float
        Tail probability (e.g. 0.01 for 1% VaR).
    horizon : int
        Time horizon (in same units as returns, e.g. weeks).

    Returns
    -------
    var : float
        Positive number representing the loss at the given confidence/day horizon.
    """
    r = np.asarray(returns)
    if method == 'gaussian':
        mu, sigma = np.mean(r), np.std(r)
        # z for one-sided quantile
        #z = abs(norm.ppf(confidence))
        z = abs(np.percentile(np.random.standard_normal(10_000), confidence * 100))
        var = -(mu + z * sigma) * np.sqrt(horizon)
    elif method == 'historical':
        # empirical quantile (these returns are negative if losses)
        hist_q = np.percentile(r, confidence * 100)
        var = -hist_q * np.sqrt(horizon)
    else:
        raise ValueError(f"Unknown VaR method '{method}'")
    return var

def constraint_var_historical(
    weights: np.ndarray,
    weights_history: list,
    replica_returns: list,
    max_var: float = 0.08,
    var_confidence: float = 0.01,
    var_horizon: int = 4,
    lookback: int = 20
) -> tuple[np.ndarray, dict]:
    """
    Project weights to satisfy a maximum historical VaR constraint.

    Parameters
    ----------
    weights : np.ndarray
        Proposed portfolio weights.
    weights_history : list of np.ndarray
        Past weights (unused here).
    replica_returns : list of float
        Past one-step replica returns.
    max_var : float
        Maximum allowed VaR (positive number, e.g. 0.08 for 8%).
    var_confidence : float
        Tail probability for VaR (e.g. 0.01 for 1%).
    var_horizon : int
        Horizon over which to scale VaR (e.g. 4 weeks).
    lookback : int
        Number of past returns to use for historical simulation.

    Returns
    -------
    new_weights, metadata : (np.ndarray, dict)
        Possibly rescaled weights and a metadata dict describing the adjustment.
    """
    metadata = {
        'activated': False,
        'estimated_var': np.nan,
        'violation': 0.0,
        'rescale_factor': 1.0
    }
    if len(replica_returns) < lookback:
        # Not enough history to estimate VaR
        return weights, metadata

    hist = replica_returns[-lookback:]
    est_var = calculate_var(hist, method='historical', confidence=var_confidence, horizon=var_horizon)
    metadata['estimated_var'] = est_var

    if est_var > max_var:
        metadata['activated'] = True
        metadata['violation'] = est_var - max_var
        factor = max_var / est_var
        weights = weights * factor
        metadata['rescale_factor'] = factor

    return weights, metadata
