```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

class MetricsCalculator:
    """Calculate and store model performance metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculate all classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision':  precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, model_name="Model"):
        """Pretty print metrics"""
        print(f"\n{'='*50}")
        print(f"{model_name} Performance Metrics")
        print(f"{'='*50}")
        for metric, value in metrics. items():
            print(f"{metric. upper():15s}: {value:.4f}")
        print(f"{'='*50}\n")

class Visualizer:
    """Visualization utilities for model comparison"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_model_comparison(results_df):
        """Compare multiple models"""
        fig, axes = plt. subplots(1, 2, figsize=(15, 5))
        
        # Bar plot
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        x = np.arange(len(results_df))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                axes[0].bar(x + i*width, results_df[metric], width, label=metric. upper())
        
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x + width * 2)
        axes[0].set_xticklabels(results_df['model'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Heatmap
        heatmap_data = results_df[metrics].T
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                    xticklabels=results_df['model'], yticklabels=metrics, ax=axes[1])
        axes[1].set_title('Performance Metrics Heatmap')
        
        plt.tight_layout()
        return fig

def save_results(results, filename='results/model_results.csv'):
    """Save results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return df
```