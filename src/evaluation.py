import os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from src.cclasses import BacktestResult
import streamlit as st
import numpy as np

import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def apply_theme(fig, ax, theme): #transparency with streamlit
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    if theme == "Dark":
        fig.patch.set_facecolor('#222222')
        ax.set_facecolor('#222222')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.title.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')

    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

def plot_correlation_matrix_st(df: pd.DataFrame, method: str = "pearson", title: str = None, theme="Light"):
    corr = df.corr(method=method)
    fig, ax = plt.subplots(figsize=(8, 6))
    apply_theme(fig, ax, theme)
    sns.heatmap(corr, annot=False, fmt=".2f", square=True, ax=ax)
    ax.set_title(title or f"{method.title()} Correlation Matrix")
    plt.tight_layout()
    return fig

def plot_series_st(df: pd.DataFrame, col: str, title: str = None, theme="Light"):
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_theme(fig, ax, theme)

    df[col].plot(ax=ax, alpha=1.0)
    ax.set_title(title or f"Time Series: {col}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    plt.tight_layout()
    return fig

def plot_acf_pacf_st(series: pd.Series, lags: int, title_prefix: str = None, theme="Light"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    for ax in axes:
        apply_theme(fig, ax, theme)
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[0].set_title((title_prefix or "") + " ACF")
    axes[1].set_title((title_prefix or "") + " PACF")
    plt.tight_layout()
    return fig

def plot_rolling_stats_st(series: pd.Series, window: int = 20, title: str = None, theme="Light"):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    fig, ax = plt.subplots()
    apply_theme(fig, ax, theme)
    series.plot(ax=ax, alpha=0.5, label="Original")
    rolling_mean.plot(ax=ax, label=f"Mean ({window})")
    rolling_std.plot(ax=ax, label=f"Std ({window})")
    ax.set_title(title or f"Rolling Mean & Std (window={window})")
    ax.legend()
    plt.tight_layout()
    return fig

def compute_aggregate_metrics(
    target_returns: pd.Series,
    replica_returns: pd.Series,
    gross_exposures: list,
    var: list,
    scaling_factors: list,
    rescale_history: list,
    weights_history: list,
    config_params: dict
) -> dict:
    """Compute overall evaluation metrics for backtest."""
    # Compute cumulative returns for both target and replica
    cumulative_target = (1 + target_returns).cumprod()
    cumulative_replica = (1 + replica_returns).cumprod()

    # Annualized return and volatility
    target_mean_return = target_returns.mean() * 52
    replica_mean_return = replica_returns.mean() * 52
    target_vol = target_returns.std() * np.sqrt(52)
    replica_vol = replica_returns.std() * np.sqrt(52)

    # Sharpe ratio
    target_sharpe = target_mean_return / target_vol if target_vol > 0 else np.nan
    replica_sharpe = replica_mean_return / replica_vol if replica_vol > 0 else np.nan

    # Drawdons
    target_drawdown = 1 - cumulative_target / cumulative_target.cummax()
    replica_drawdown = 1 - cumulative_replica / cumulative_replica.cummax()

    # TE, IR, Corr
    tracking_error = (replica_returns - target_returns).std() * np.sqrt(52)
    information_ratio = (replica_mean_return - target_mean_return) / tracking_error if tracking_error > 0 else np.nan
    correlation = replica_returns.corr(target_returns)

    return {
        **config_params,
        'target_returns': target_returns,
        'replica_returns': replica_returns,
        'cumulative_target': cumulative_target,
        'cumulative_replica': cumulative_replica,
        'target_mean_return': target_mean_return,
        'replica_mean_return': replica_mean_return,
        'target_vol': target_vol,
        'replica_vol': replica_vol,
        'target_sharpe': target_sharpe,
        'replica_sharpe': replica_sharpe,
        'target_max_drawdown': target_drawdown.max(),
        'replica_max_drawdown': replica_drawdown.max(),
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'correlation': correlation,
        'gross_exposures': gross_exposures,
        'avg_gross_exposure': np.mean(gross_exposures),
        'var':var,
        'avg_var':np.nanmean(var),
        'scaling_factors': scaling_factors,
        'rescale_history': rescale_history,
        'weights_history': weights_history
    }



def display_backtest_result(result, theme="Light"):
    """
    Display aggregate metrics and plot cumulative returns for a backtest result.
    """
    print("=== Aggregate Metrics ===")
    for k, v in result.aggregate_metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")
        elif isinstance(v, str):
            print(f"{k}: {v}")
        # Skip arrays or series here

    cum_target = (1 + result.target_returns).cumprod()
    cum_replica = (1 + result.replica_returns).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))
    apply_theme(fig, ax, theme)
    ax.plot(cum_target, label="Target Index")
    ax.plot(cum_replica, label="Replica Portfolio")
    ax.set_title("Cumulative Returns — Backtest Result")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

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

def plot_detailed_results(best_config, X, max_var_threshold, save_dir=None, theme="Light"):
    """
    Plots detailed metrics and diagnostic charts for the best configuration using Streamlit.
    """
    model_name = best_config.get('model_name', 'unknown')
    if save_dir is not None:
        save_path = save_dir + model_name
        os.makedirs(save_path, exist_ok=True)

    # --- Detailed Metrics Table ---
    metrics_normalized = pd.DataFrame({
        'Metric': ['Annualized return', 'Annualized volatility', 'Sharpe ratio',
                   'Max Drawdown', 'Tracking Error', 'Information ratio',
                   'Correlation', 'Average gross exposure', 'Average VaR (1%, 1M)',
                   'Rolling Window'],
        'Target': [f"{best_config['target_mean_return']*100:.2f}%",
                   f"{best_config['target_vol']*100:.2f}%",
                   f"{best_config['target_sharpe']:.2f}",
                   f"{best_config['target_max_drawdown']*100:.2f}%",
                   "N/A",
                   "N/A",
                   "N/A",
                   "N/A",
                   "N/A",
                   "N/A"],
        'Replica': [f"{best_config['replica_mean_return']*100:.2f}%",
                    f"{best_config['replica_vol']*100:.2f}%",
                    f"{best_config['replica_sharpe']:.2f}",
                    f"{best_config['replica_max_drawdown']*100:.2f}%",
                    f"{best_config['tracking_error']*100:.2f}%",
                    f"{best_config['information_ratio']:.2f}",
                    f"{best_config['correlation']:.4f}",
                    f"{best_config['avg_gross_exposure']:.4f}",
                    f"{best_config['avg_var']*100:.2f}%",
                    f"{best_config['rolling_window']}"]
        })

    st.markdown("#### Detailed metrics for the best configuration (normalized returns):")
    st.dataframe(metrics_normalized)

    # --- Cumulative Returns ---
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    ax.plot(best_config['cumulative_target'], label='Target index', color='blue')
    ax.plot(best_config['cumulative_replica'], label='Replica portfolio', color='red')
    ax.set_title('Cumulative returns: target vs replica')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative return')
    ax.legend(fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Drawdowns ---
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    target_drawdown = 1 - best_config['cumulative_target'] / best_config['cumulative_target'].cummax()
    replica_drawdown = 1 - best_config['cumulative_replica'] / best_config['cumulative_replica'].cummax()
    ax.plot(target_drawdown, label='Target index', color='blue')
    ax.plot(replica_drawdown, label='Replica portfolio', color='red')
    ax.set_title('Drawdowns: target vs replica')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.legend(fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Gross Exposure over Time ---
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    gross_exposure_series = pd.Series(best_config['gross_exposures'], index=best_config['replica_returns'].index)
    ax.plot(gross_exposure_series, color='purple')
    ax.set_title('Gross exposure over time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Gross exposure')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # --- VaR over Time ---
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    var_series = pd.Series(best_config['var'], index=best_config['replica_returns'].index)
    ax.plot(var_series, color='orange')
    ax.axhline(y=max_var_threshold, color='r', linestyle='--',
                label=f'VaR threshold ({max_var_threshold*100:.2f}%)')
    ax.set_title('Value at Risk (VaR) over time')
    ax.set_xlabel('Date')
    ax.set_ylabel('VaR (1%, 1M)')
    ax.legend(fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Scaling Factors over Time ---
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    scaling_series = pd.Series(best_config['rescale_history'], index=best_config['replica_returns'].index)
    ax.plot(scaling_series, color='green')
    ax.set_title('Risk scaling factors over time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Scaling factor')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Portfolio Weights over Time ---
    weights_history = best_config['weights_history']
    weights_df = pd.DataFrame(weights_history, index=best_config['replica_returns'].index)
    weights_df.columns = X.columns
    top_weights = weights_df.abs().mean().sort_values(ascending=False).head(10).index

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    for col in top_weights:
        ax.plot(weights_df[col], label=col)
    ax.set_title('Top 10 portfolio weights over time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight')
    ax.legend(fontsize='xx-small')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Weekly Returns: Target vs Replica ---
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    ax.scatter(best_config['target_returns'], best_config['replica_returns'], alpha=0.5)
    ax.plot([-0.1, 0.1], [-0.1, 0.1], 'r--')  # Diagonal line
    ax.set_title('Weekly returns: target vs replica')
    ax.set_xlabel('Target returns')
    ax.set_ylabel('Replica returns')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Rolling Correlation ---
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    rolling_corr = best_config['replica_returns'].rolling(window=26).corr(best_config['target_returns'])
    ax.plot(rolling_corr, color='blue')
    ax.set_title('Rolling 26-Week correlation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Correlation')
    ax.axhline(y=best_config['correlation'], color='r', linestyle='--',
                label=f'Overall Correlation: {best_config["correlation"]:.4f}')
    ax.legend(fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)



