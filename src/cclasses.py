from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class BacktestResult:
    model_name: str
    model_params: dict
    weights_history: np.ndarray
    gross_exposures: list
    scale_history: list
    rescale_history: list
    var:list
    replica_returns: pd.Series
    target_returns: pd.Series
    aggregate_metrics: dict
    constraints_history: dict
    transaction_costs: pd.Series

    def summary(self):
        return {
            'model': self.model_name,
            'sharpe': self.aggregate_metrics.get('replica_sharpe'),
            'IR': self.aggregate_metrics.get('information_ratio'),
            'TE': self.aggregate_metrics.get('tracking_error'),
            'corr': self.aggregate_metrics.get('correlation'),
            'transaction_costs': self.aggregate_metrics.get('transaction_costs_sum'),
        }
    
