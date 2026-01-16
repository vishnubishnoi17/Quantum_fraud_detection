"""
Utility functions for Quantum Fraud Detection project
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def balance_dataset(
    X: np.ndarray, 
    y: np.ndarray, 
    n_samples: int = 400
) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Balance dataset by sampling equal number from each class. 
    
    Args:
        X: Feature matrix
        y: Labels
        n_samples: Total samples (split equally between classes)
        
    Returns: 
        Balanced X, y arrays
    """
    fraud_idx = np.where(y == 1)[0]
    non_fraud_idx = np.where(y == 0)[0]
    
    n_per_class = n_samples // 2
    
    fraud_sample = np.random.choice(
        fraud_idx, min(n_per_class, len(fraud_idx)), replace=False
    )
    non_fraud_sample = np.random.choice(
        non_fraud_idx, min(n_per_class, len(non_fraud_idx)), replace=False
    )
    
    idx = np.concatenate([fraud_sample, non_fraud_sample])
    np.random.shuffle(idx)
    
    return X[idx], y[idx]


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: np.ndarray = None
) -> dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba:  Predicted probabilities (for AUC-ROC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None: 
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            logger.warning("Cannot calculate AUC-ROC (single class in y_true)")
            metrics['auc_roc'] = np.nan
            
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def save_model_results(
    model_name: str,
    metrics: dict,
    save_path: str = 'results'
):
    """Save model metrics to CSV."""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    df = pd.DataFrame([metrics])
    df.insert(0, 'Model', model_name)
    
    filepath = os.path.join(save_path, f"{model_name.lower().replace(' ', '_')}_results.csv")
    df.to_csv(filepath, index=False)
    logger.info(f"Saved results to {filepath}")


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Importance"
):
    """Plot feature importance bar chart."""
    import matplotlib.pyplot as plt
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    return plt


class EarlyStopping:
    """Early stopping for quantum training."""
    
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = loss
            self. counter = 0
            
        return self.should_stop
