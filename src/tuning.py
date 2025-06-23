from itertools import product
import pandas as pd
import joblib
import os
import numpy as np

from src.training import run_single_backtest_experiment          
# Path to save results
folder_path = './'

# Define paths
data_raw_path = folder_path+"data/raw/"
data_interim_path = folder_path+"data/interim/"
data_processed_path = folder_path+"data/processed/"

#define directory for models
save_dir=folder_path+"results/"

def tune_hyperparameters_model(
    model_name,
    alphas,
    l1_ratios,
    rolling_windows,
    X,
    y,
    constraint_funcs,
    constraint_params=None,
    step=1,
    metric_key='information_ratio',
    save_path=None
):
    """
    Tune hyperparameters for a single model by running backtests across all combinations.

    Parameters:
    - model_name: str, name of the regression model (e.g. 'ridge', 'lasso', 'elasticnet', 'linear')
    - alphas: list of float or [None], values for the alpha regularization parameter
    - l1_ratios: list of float or [None], values for l1_ratio (only relevant for ElasticNet)
    - rolling_windows: list of int, different window sizes for the backtest
    - X: np.ndarray or pd.DataFrame, feature matrix
    - y: np.ndarray or pd.Series, target returns
    - constraint_funcs: list of callable, constraints applied during optimization
    - step: int, step size for the sliding window
    - metric_key: str, metric to optimize (e.g. 'information_ratio')
    - save_path: str or None, path to save best result and config using joblib

    Returns:
    - result_model: BacktestResult object, the best performing model configuration
    """

    print(f"\nTuning hyperparameters for {model_name} regression...")

    result_model = None
    result_metric = {}


    for window, alpha, l1_ratio in product(rolling_windows, alphas, l1_ratios):
        model_params = {}
        if alpha is not None:
            model_params["alpha"] = alpha
        if l1_ratio is not None and model_name == 'elasticnet':
            model_params["l1_ratio"] = l1_ratio

        config = {
            "model_name": model_name,
            "model_params": model_params,
            "X": X,
            "y": y,
            "split_strategy": "sliding",
            "window": window,
            "step": step,
            "constraint_funcs": constraint_funcs,
            "constraint_params": constraint_params
        }

        result = run_single_backtest_experiment(config)
        metrics = result.aggregate_metrics



        # Check for best model
        if metrics.get(metric_key, -np.inf) > result_metric.get(metric_key, -np.inf):
            result_model = result
            result_metric = metrics

    best_config = {
        "model_name": result_model.model_name,
        "model_params": result_model.model_params,
        "rolling_window": config["window"],
        "step": config["step"],
        "metric_key": metric_key,
        "metric_value": result_metric[metric_key],
    }


    if save_path:
        joblib.dump({
            "best_result": result_model,
            "best_config": best_config

        }, save_path)
        print(f"\nSaved best config and result to {save_path}")


    return result_model, best_config

def tune_all_models(
    models_config,
    X,
    y,
    constraint_funcs,
    constraint_params,
    step=1,
    metric_key='information_ratio',
    save_dir=None
):
    """
    Tune hyperparameters across multiple model types and return a summary.

    Parameters:
    - models_config: dict, keys are model names, values are dicts with 'alphas', 'l1_ratios', 'rolling_windows'
    - X: np.ndarray or pd.DataFrame, input features
    - y: np.ndarray or pd.Series, target returns
    - constraint_funcs: list of constraint functions to apply during optimization
    - step: int, step size for the rolling/sliding window
    - metric_key: str, which metric to sort and evaluate models by (e.g. 'information_ratio')
    - save_dir: str or None, directory to save each model’s best result using joblib

    Returns:
    - df_results: pd.DataFrame, summary table of best models sorted by metric_key
    """
    best_results = []
    best_configs = []

    for model_name, param_grid in models_config.items():
        alphas = param_grid.get('alphas', [None])
        l1_ratios = param_grid.get('l1_ratios', [None])
        rolling_windows = param_grid.get('rolling_windows', [52])  # default 1 year

        if save_dir is not None:
            save_folder = f"{save_dir}/{model_name}"
            os.makedirs(save_folder, exist_ok=True)
            save_path = f"{save_folder}/{model_name}_best_model.joblib"

        best_result, best_config = tune_hyperparameters_model(
            model_name=model_name,
            alphas=alphas,
            l1_ratios=l1_ratios,
            rolling_windows=rolling_windows,
            X=X,
            y=y,
            constraint_funcs=constraint_funcs,
            constraint_params=constraint_params,
            step=step,
            metric_key=metric_key,
            save_path=save_path #we have our own results, user does this in site cache
        )

        

        summary = best_result.summary()
        summary.update({
            'model_name': model_name,
            'model_params': best_result.model_params,
            'window': best_result.model_params.get("window", "unknown")
        })
        best_results.append(summary)
        best_configs.append(best_config)

    # Sort results by the selected metric
    df_results = pd.DataFrame(best_results)
    df_results = df_results.sort_values(by='IR', ascending=False)

    print("\n=== Best Model per Type ===")
    print(df_results[['model_name', 'IR', 'sharpe', 'TE', 'corr']])

    return df_results, best_configs