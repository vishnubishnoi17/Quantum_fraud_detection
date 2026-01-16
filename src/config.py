"""
Configuration settings for Quantum Fraud Detection
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = PROJECT_ROOT / 'figures'

# Create directories if they don't exist
for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Data preprocessing
N_FEATURES_SELECTED = 4
TRAIN_TEST_SPLIT = 0.2
SCALER_RANGE = (0, 1)

# Classical models hyperparameters
CLASSICAL_MODELS_CONFIG = {
    'logistic_regression': {
        'C':  0.1,
        'max_iter':  500,
        'random_state': RANDOM_SEED
    },
    'random_forest': {
        'n_estimators': 50,
        'max_depth':  5,
        'random_state': RANDOM_SEED,
        'min_samples_split': 10
    },
    'xgboost': {
        'n_estimators': 50,
        'max_depth':  4,
        'learning_rate': 0.05,
        'random_state':  RANDOM_SEED
    },
    'decision_tree': {
        'max_depth': 3,
        'random_state': RANDOM_SEED,
        'min_samples_split': 20
    }
}

# Quantum model hyperparameters
VQC_CONFIG = {
    'n_qubits': 4,
    'feature_map':  {
        'type': 'ZZFeatureMap',
        'reps': 2,
        'entanglement': 'linear'
    },
    'ansatz': {
        'type': 'EfficientSU2',
        'reps': 2,
        'entanglement':  'circular'
    },
    'optimizer': {
        'type': 'SPSA',
        'maxiter': 100
    },
    'training_samples': {
        'train':  400,
        'test': 150
    }
}

# Visualization settings
PLOT_STYLE = 'whitegrid'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
