import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from pykalman import KalmanFilter
from sktime.forecasting.model_selection import SlidingWindowSplitter, ExpandingWindowSplitter
import pandas as pd
from typing import Generator, Tuple
from src.cclasses import BacktestResult
from src.utilities import prepare_data, unscale_weights, compute_step_metrics, compute_transaction_cost
from src.constraints import apply_constraints
from src.evaluation import compute_aggregate_metrics

models_config = {

    'linear': {
        'alphas': [None],
        'l1_ratios': [None],
        'rolling_windows': [52, 104,156]
    },
    'ridge': {
        'alphas': [0.01, 0.1, 1.0, 10.0],
        'l1_ratios': [None],
        'rolling_windows': [52, 104,156]
    },
    'lasso': {
        'alphas': [0.0001, 0.001, 0.01],
        'l1_ratios': [None],
        'rolling_windows': [52, 104,156]
    },

    'elasticnet': {
        'alphas': [0.0001, 0.001, 0.01],
        'l1_ratios': [0.0, 0.5, 1.0],
        'rolling_windows': [52, 104,156]
    },


    'kalman': {
        'alphas': [None],
        'l1_ratios': [None],
        "rolling_windows": [52, 104, 156]
    }
}

def train_model(build_model, model_params, X_train, y_train):
    """
    Build and fit model pipeline on training data.
    """
    model = build_model(**model_params)
    model.fit(X_train, y_train)
    return model


class KalmanRegressor(BaseEstimator, RegressorMixin):

    def __init__(self,
                 Q_val=0.0004,
                 R_val=0.0016,
                 initial_weights=None,
                 lookback=5,
                 min_Q=1e-7,
                 max_Q=1.0,
                 min_R=1e-7,
                 max_R=1.0):

        self.Q_val = Q_val
        self.R_val = R_val
        self.initial_weights = initial_weights
        self.lookback = lookback
        self.min_Q = min_Q
        self.max_Q = max_Q
        self.min_R = min_R
        self.max_R = max_R
        self.kf = None
        self.filtered_state_means = None

        self.coef_ = None
        self.model_name = "kalman"
        self.model_params = {"Q_val": Q_val, "R_val": R_val}

        self.weights_history = []
        self.residuals_history = []

    def _estimate_dynamic_Q(self, n_features):
        """Estimate dynamic process noise covariance Q_t"""
        if len(self.weights_history) >= self.lookback:
            recent_weights = np.array(self.weights_history[-self.lookback:])
            if len(recent_weights) > 1:
                weight_diffs = np.diff(recent_weights, axis=0)
                Q_diag = np.var(weight_diffs, axis=0)
                # Avoid zeros and clip to bounds
                Q_diag = np.clip(Q_diag, self.min_Q, self.max_Q)
                return np.diag(Q_diag)

        # Fallback to initial Q
        return self.Q_val * np.eye(n_features)

    def _estimate_dynamic_R(self):
        """Estimate dynamic observation noise covariance R_t"""
        if len(self.residuals_history) >= self.lookback:
            recent_residuals = self.residuals_history[-self.lookback:]
            R_t = np.var(recent_residuals)
            return np.clip(R_t, self.min_R, self.max_R)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        n_timesteps, n_features = X.shape

        # Estimate dynamic parameters
        Q_matrix = self._estimate_dynamic_Q(n_features)
        R_value = self._estimate_dynamic_R()

        # Use X as observation matrix
        observation_matrices = X.reshape((n_timesteps, 1, n_features))

        # Identity transition (weights evolve independently)
        transition_matrices = np.eye(n_features)

        initial_state_mean = (
            self.initial_weights if self.initial_weights is not None
            else np.zeros(n_features)
        )

        self.kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            transition_covariance=Q_matrix,
            observation_covariance=R_value if R_value is not None else self.R_val,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=np.eye(n_features)
        )

        self.filtered_state_means, filtered_state_covariances = self.kf.filter(y)
        self.weights_history = self.filtered_state_means
        self.coef_ = self.filtered_state_means[-1]

        # Calculate residuals for R estimation
        predictions = np.array([np.dot(X[i], self.filtered_state_means[i])
                              for i in range(n_timesteps)])
        residuals = y - predictions
        self.residuals_history.extend(residuals.tolist())

        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.filtered_state_means is None:
            raise RuntimeError("Model must be fitted before prediction.")


        return np.dot(X, self.coef_)

def get_model(model_name: str, **params):
    """
    Return a scikit-learn pipeline with a scaler and the selected regression model.

    Parameters
    ----------
    model_name : str
        One of ['linear', 'ridge', 'lasso', 'elasticnet','kalman']
    **params : keyword arguments passed to the regressor

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    regressors = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elasticnet': ElasticNet,
        'kalman': KalmanRegressor
    }

    if model_name not in regressors:
        raise ValueError(f"Model '{model_name}' not supported.")

    # Special handling for KalmanRegressor parameters
    if model_name == 'kalman':
        if 'alpha' in params:
            params['Q_val'] = params.pop('alpha')
        if 'l1_ratio' in params:
            params['R_val'] = params.pop('l1_ratio')

    regressor = regressors[model_name](**params)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", regressor)
    ])

def simulate_backtest(
    X, y, splitter, build_model, model_params, constraint_funcs=None, constraint_params=None
):
    """
    Run one-step-ahead replication backtest.

    Returns BacktestResult with weights, returns, metrics, etc.
    """
    # Initialize records
    scale_history = []
    weights_history = []
    gross_exposures = []
    rescale_history = []
    var=[]
    replica_returns_gross = []
    replica_returns_net = []
    target_returns = []
    dates = []
    transaction_costs = []

    # Initialize constraints history
    constraints_history = {fn.__name__: [] for fn in constraint_funcs} if constraint_funcs else {}


    for train_idx, test_idx in splitter:
        # Get the proper section of data X, y
        X_train, y_train, X_next, y_next, date_next = prepare_data(X, y, train_idx, test_idx)

        # Train the model
        model = train_model(build_model, model_params, X_train, y_train)

        # Get the scale factor and scaled-back weights to portfolio units
        original_weights, scale = unscale_weights(model)

        # Apply check and constraints
        original_weights,rescale_factor = apply_constraints(original_weights, constraint_funcs, weights_history, replica_returns_net, constraints_history, constraint_params)

        # Compute per-step metrics
        replica_return_gross, gross_exposure = compute_step_metrics(X_next, original_weights)

        # Compute VaR (Value at Risk) for the current step
        if "constraint_var_historical" in constraints_history:
            latest_metadata = constraints_history["constraint_var_historical"][-1]
            var_value = latest_metadata.get("estimated_var", np.nan)
            var.append(var_value)
        else:
            var.append(np.nan)

        prev_w = weights_history[-1] if weights_history else None
        transaction_cost = compute_transaction_cost(
            current_weights=original_weights,
            previous_weights=prev_w,
            cost_rate=0.0004
        )

        transaction_costs.append(transaction_cost)

        replica_return_net = replica_return_gross - transaction_cost

        # Record data
        scale_history.append(scale)
        rescale_history.append(rescale_factor)
        weights_history.append(original_weights)
        gross_exposures.append(gross_exposure)

        replica_returns_gross.append(replica_return_gross)
        replica_returns_net.append(replica_return_net)
        target_returns.append(y_next)
        dates.append(date_next)

    # Transform time-series data
    weights_history = np.vstack(weights_history)
    replica_returns_gross = pd.Series(replica_returns_gross, index=dates, name='replica_returns')
    replica_returns_net = pd.Series(replica_returns_net, index=dates, name='replica_returns')
    target_returns = pd.Series(target_returns, index=dates, name='target_returns')
    transaction_costs = pd.Series(transaction_costs, index=dates, name='transaction_costs')

    # Compute aggregate metrics
    aggregate_metrics = compute_aggregate_metrics(
        target_returns=target_returns,
        replica_returns=replica_returns_net,
        gross_exposures=gross_exposures,
        var=var,
        scaling_factors=scale_history,
        rescale_history=rescale_history,
        weights_history=weights_history,
        config_params=model_params,
        transaction_costs=transaction_costs,
    )
    # Get the model name from the actual regressor within the pipeline
    model_name_in_pipeline = type(model.named_steps['regressor']).__name__

    # Return an experiment summary class
    reg_name = type(model.named_steps['regressor']).__name__
    return BacktestResult(
        model_name=reg_name,
        model_params=model_params,
        weights_history=weights_history,
        gross_exposures=gross_exposures,
        scale_history=scale_history,
        rescale_history=rescale_history,
        var=var,
        replica_returns=replica_returns_net,
        target_returns=target_returns,
        aggregate_metrics=aggregate_metrics,
        constraints_history=constraints_history,
        transaction_costs=transaction_costs
    )

def generate_backtest_splits(
    y: pd.Series,
    strategy: str = "sliding",
    window_length: int = 52,
    step_length: int = 1
) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
    """
    One-step-ahead backtest splits for replication.

    At each fold:
      - Train on `window_length` points (or all past for expanding)
      - Test on the single point immediately after that window (t+1)

    Parameters
    ----------
    y : pd.Series
        Target series (index only used for length).
    strategy : {'sliding', 'expanding'}
        'sliding'   → fixed-size rolling window of length `window_length`.
        'expanding' → growing window that starts at size `window_length` and then increases.
    window_length : int
        For 'sliding': # samples in each train window.
        For 'expanding': # samples in initial train window.
    step_length : int
        How many periods to move forward between fits.

    Yields
    ------
    train_idx : np.ndarray
        Integer positions for training (length = window_length or growing).
    test_idx : np.ndarray
        Single-element array containing the one-step-ahead index.
    """
    # always one-step ahead
    fh = [1]
    if strategy == "sliding":
        splitter = SlidingWindowSplitter(
            window_length=window_length,
            step_length=step_length,
            fh=fh
        )
    elif strategy == "expanding":
        splitter = ExpandingWindowSplitter(
            initial_window=window_length,
            step_length=step_length,
            fh=fh
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    for train_idx, test_idx in splitter.split(y):
        yield train_idx, test_idx


def run_single_backtest_experiment(config: dict) -> BacktestResult:
    """
    Run a backtest experiment from a unified configuration dictionary.

    Expected keys in config:
        - model_name: str
        - model_params: dict
        - split_strategy: str
        - window: int
        - step: int
        - constraint_funcs: list or None
        - constraint_params: dict
    """
    model_name = config['model_name']
    model_params = config['model_params']
    build_model = lambda **p: get_model(model_name, **p)

    splitter = generate_backtest_splits(
        y=config['y'],
        strategy=config.get('split_strategy', 'sliding'),
        window_length=config['window'],
        step_length=config['step']
    )

    return simulate_backtest(
        X=config['X'],
        y=config['y'],
        splitter=splitter,
        build_model=build_model,
        model_params=model_params,
        constraint_funcs=config.get('constraint_funcs', None),
        constraint_params=config.get('constraint_params', {})
    )
