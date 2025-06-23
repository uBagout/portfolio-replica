import pandas as pd
from typing import Dict, Tuple
import numpy as np
from src.cclasses import BacktestResult
from typing import Optional, List, Generator
import pickle
from typing import Dict, List, Tuple
from typing import Tuple
from typing import Optional
import matplotlib.pyplot as plt


def load_data():
    #
    filepath = "data/processed/"
    df_cleaned_imp_LLL1 = pd.read_parquet(filepath + "df_cleaned_imp_LLL1.parquet")

    futures_cleaned_imp_LLL1 = pd.read_parquet(filepath + "futures_cleaned_imp_LLL1.parquet")
    indices = pd.read_parquet(filepath + "indices.parquet")
    
    with open(filepath + "tickers_name_dict.pkl", 'rb') as f:
        tickers_name_dict = pickle.load(f)

    return df_cleaned_imp_LLL1, futures_cleaned_imp_LLL1, indices, tickers_name_dict


def compute_index_returns(
    df_idx: pd.DataFrame,
    index_weights: Dict[str, float] = None
) -> pd.Series:
    """
    Compute weighted composite index returns from raw index price levels.

    Parameters
    ----------
    df_idx : pd.DataFrame
        DataFrame of index price levels (columns are index names).
    index_weights : dict, optional
        Mapping from index column to weight. If None, uses equal weights.

    Returns
    -------
    y : pd.Series
        Composite index returns.
    """
    # Determine which indices to use
    cols = df_idx.columns.tolist()
    if index_weights is None:
        # Equal weights if not provided
        index_weights = {col: 1.0 / len(cols) for col in cols}
    else:
        # Validate provided keys
        missing = set(index_weights) - set(cols)
        if missing:
            raise KeyError(f"Index weights refer to unknown columns: {missing}")

    # Compute simple returns
    ret = df_idx[list(index_weights.keys())].pct_change(fill_method=None).dropna()

    # Apply weights
    weighted = pd.DataFrame({col: ret[col] * weight
                             for col, weight in index_weights.items()},
                            index=ret.index)

    # Sum to get composite
    y = weighted.sum(axis=1)
    y.name = "Target_Index"
    return y

def compute_futures_returns(
    df_fut: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute returns for futures price levels.

    Parameters
    ----------
    df_fut : pd.DataFrame
        DataFrame of futures price levels (columns are futures names).

    Returns
    -------
    X : pd.DataFrame
        Futures returns DataFrame.
    """
    X = df_fut.pct_change(fill_method=None).dropna()
    return X

def align_features_target(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align feature and target on common datetime index.

    Parameters
    ----------
    X : pd.DataFrame
        Feature returns with datetime index.
    y : pd.Series
        Target returns with datetime index.

    Returns
    -------
    X_aligned : pd.DataFrame
    y_aligned : pd.Series
        Subsets of X and y sharing the same index.
    """
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    return X_aligned, y_aligned

def prepare_data(X, y, train_idx, test_idx):
    """
    Extract training and test sets for a single backtest iteration.

    Returns:
        X_train, y_train: training features and target
        X_next, y_next: test features and target (next step)
        date_next: date/index of test step
    """
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_next = X.iloc[test_idx].values.flatten()
    y_next = y.iloc[test_idx].iloc[0]
    date_next = y.index[test_idx][0]
    return X_train, y_train, X_next, y_next, date_next


def prepare_X_y(
    df_indices: pd.DataFrame,
    df_futures: pd.DataFrame,
    index_weights: Dict[str, float] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    High-level wrapper: compute and align index and futures returns.

    Parameters
    ----------
    df_indices : pd.DataFrame
        Raw index price levels.
    df_futures : pd.DataFrame
        Raw futures price levels.
    index_weights : dict, optional
        Composite index weights. Default: equal weights.

    Returns
    -------
    X, y : aligned returns ready for modeling
    """
    y = compute_index_returns(df_indices, index_weights=index_weights)
    X = compute_futures_returns(df_futures)
    X_aligned, y_aligned = align_features_target(X, y)
    return X_aligned, y_aligned

def compute_transaction_cost(
    current_weights: np.ndarray,
    previous_weights: Optional[np.ndarray] = None,
    cost_rate: float = 0.0004
) -> float:
    """
    Compute round‑trip transaction cost based on turnover.

    Parameters
    ----------
    current_weights : np.ndarray
        New portfolio weights w_t.
    previous_weights : np.ndarray or None
        Prior portfolio weights w_{t-1}. If None, assumed zero (initial alloc).
    cost_rate : float
        Round‑trip cost per unit turnover (e.g. 0.0004 for 4 bps).

    Returns
    -------
    cost : float
        Transaction cost to be subtracted from portfolio return.
    """
    if previous_weights is None:
        prev = np.zeros_like(current_weights)
    else:
        prev = previous_weights
    turnover = np.sum(np.abs(current_weights - prev))
    return turnover * cost_rate



def unscale_weights(model):
    """
    Extract model weights and unscale them if scaler present.

    Returns:
        original_weights: scaled-back portfolio weights
        scale: scaling factors from scaler or ones
    """
    scaler = model.named_steps.get('scaler', None)
    reg = model.named_steps['regressor']
    normalized_weights = reg.coef_.copy()
    if scaler is not None and hasattr(scaler, 'scale_'):
        scale = scaler.scale_
        original_weights = normalized_weights / scale
    else:
        scale = np.ones_like(normalized_weights)
        original_weights = normalized_weights.copy()
    return original_weights, scale

def compute_step_metrics(X_next, original_weights):
    """
    Compute per-step metrics.
    """
    replica_return = float(np.dot(X_next, original_weights))
    gross_exposure = np.sum(np.abs(original_weights))
    return replica_return, gross_exposure

def return_figures():
    """
    Load and return figures for the EDA section.
    """
    figurenames = [
        "lll1imputed.png",
    ]

    filepath = "figures/"
    figures = []
    for fname in figurenames:
        figures.append(plt.imread(filepath + fname))
        
    return figures

def return_eval_figures() -> List[plt.Figure]:
    """
    Load and return figures for the Model Training section.
    """
    figurenames = [
        "cumulative_returns_Pipeline.png",
        "drawdowns_Pipeline.png",
        "gross_exposure_Pipeline.png",
        "portfolio_weights_Pipeline.png",
        "rolling_correlation_Pipeline.png",
        "scaling_factors_Pipeline.png",
        "var_Pipeline.png",
        "weekly_returns_Pipeline.png"
    ]

    filepath = "./results/Pipeline/"
    figures = []
    for fname in figurenames:
        figures.append(plt.imread(filepath + fname))

    return figures

def prepare_plotting_config(result_model: BacktestResult, config_params: dict) -> dict:
    """
    Converts a BacktestResult instance into the structure expected by plot_detailed_results.
    """
     # Start from aggregate_metrics
    config = dict(result_model.aggregate_metrics)
    # Add/override with extra info
    config.update({
        'model_name': result_model.model_name,
        'model_params': result_model.model_params,
        'rolling_window': config_params.get("window", None),
        'step': config_params.get("step", None),

    })
    return config