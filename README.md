# Quantum Machine Learning for Fraud Classification

A comprehensive implementation of Quantum Machine Learning (QML) classifiers for credit card fraud detection, comparing quantum approaches with classical machine learning baselines.

## 🎯 Project Overview

This project implements and benchmarks **Variational Quantum Classifiers (VQC)** against classical machine learning models on a fraud detection task. The goal is to demonstrate the potential of quantum computing for classification problems while maintaining rigorous comparison with state-of-the-art classical methods.

### Key Features

- ✅ **Data Preprocessing**: Dimensionality reduction from 8 to 4 features using multiple techniques
- ✅ **Classical Baselines**:  Logistic Regression, Random Forest, XGBoost, Neural Networks
- ✅ **Quantum Feature Maps**: ZZ Feature Map, Pauli Feature Map, Data Re-uploading
- ✅ **VQC Implementation**: Variational Quantum Classifier with multiple ansatz options
- ✅ **Comprehensive Analysis**: Detailed performance comparison and visualization

## 📁 Project Structure

```
qml-fraud-classification/
├── data/
│   ├── dataset.csv                    # Original dataset
│   └── processed/                     # Preprocessed data (generated)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Data cleaning and feature reduction
│   ├── 02_classical_baselines.ipynb   # Classical ML models
│   ├── 03_quantum_feature_maps.ipynb  # Quantum encoding strategies
│   ├── 04_vqc_classifier.ipynb        # Variational Quantum Classifier
│   └── 05_results_comparison.ipynb    # Comprehensive analysis
├── src/
│   └── utils.py                       # Helper functions and utilities
├── models/                            # Saved models (generated)
├── results/                           # Results and metrics (generated)
├── figures/                           # Generated visualizations (generated)
├── requirements.txt                   # Python dependencies
├── setup.sh                          # Setup script
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to your project directory
cd "~/serious bkc/qml-fraud-classification"

# Make setup script executable
chmod +x setup. sh

# Run setup (creates virtual environment and installs dependencies)
./setup.sh

# Activate virtual environment
source qml_env/bin/activate
```

### 2. Manual Installation (Alternative)

```bash
# Create virtual environment
python3 -m venv qml_env
source qml_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p src results figures models data/processed
```

### 3. Run the Analysis

Execute notebooks in order:

```bash
# Start Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

Then run notebooks sequentially: 
1. `01_data_preprocessing.ipynb` - Clean data and reduce features
2. `02_classical_baselines.ipynb` - Train classical models
3. `03_quantum_feature_maps.ipynb` - Design quantum encodings
4. `04_vqc_classifier.ipynb` - Train quantum classifier
5. `05_results_comparison.ipynb` - Analyze and compare results

## 📊 Methodology

### 1. Data Preprocessing

- **Original Features**: 8
- **Target**:  Binary classification (fraud vs legitimate)
- **Feature Reduction**: 8 → 4 features using: 
  - Correlation analysis
  - Mutual Information
  - **Random Forest Feature Importance** (selected method)
  - PCA (alternative approach)
- **Normalization**: RobustScaler (handles outliers)
- **Train-Test Split**: 80-20 with stratification

### 2. Classical Baselines

| Model | Parameters | Description |
|-------|-----------|-------------|
| **Logistic Regression** | 4 | Linear baseline |
| **Random Forest** | ~1000s | Ensemble method |
| **XGBoost** | ~1000s | Gradient boosting |
| **Neural Network** | ~200 | MLP (16-8 architecture) |

### 3. Quantum Feature Maps

| Feature Map | Qubits | Depth | Description |
|-------------|--------|-------|-------------|
| **Angle Encoding** | 4 | Low | Simple RY rotations |
| **ZZ Feature Map** | 4 | Medium | Second-order Pauli-Z evolution |
| **Pauli Feature Map** | 4 | High | Z and ZZ strings with full entanglement |
| **Data Re-uploading** | 4 | High | Multiple uploads with entanglement |

### 4. Variational Quantum Classifier (VQC)

- **Feature Map**: ZZ Feature Map (2 repetitions)
- **Ansatz**: RealAmplitudes (3 repetitions, linear entanglement)
- **Optimizer**: COBYLA (gradient-free, max 200 iterations)
- **Backend**: Qiskit Aer Simulator
- **Trainable Parameters**: 12-24 (depending on ansatz)

## 📈 Results

Results will be generated after running all notebooks.  Expected outputs: 

### Performance Metrics

All models evaluated on: 
- **Accuracy**: Overall correctness
- **Precision**:  Fraud prediction accuracy
- **Recall**:  Fraud detection rate
- **F1 Score**: Harmonic mean of precision/recall
- **AUC-ROC**: Area under ROC curve

### Visualizations

Generated figures include:
- Feature importance comparison
- Model performance comparison (bar charts, radar charts, heatmaps)
- Confusion matrices
- ROC curves
- VQC training convergence
- Quantum vs classical analysis

## 🔬 Quantum Advantage Analysis

The project investigates: 

1. **Parameter Efficiency**: VQC uses significantly fewer parameters than classical NNs
2. **Performance**:  Competitive accuracy with simpler architecture
3. **NISQ Limitations**: Current quantum hardware constraints
4. **Future Potential**: Scalability with fault-tolerant quantum computers

## 📝 Key Files Generated

After running all notebooks: 

```
results/
├── classical_baselines_results.csv
├── all_models_results.csv
├── vqc_training_history.csv
├── FINAL_SUMMARY_REPORT. txt
├── RECOMMENDATIONS.txt
├── results_table.tex (for LaTeX)
└── results_table.md (for Markdown)

figures/
├── class_distribution.png
├── correlation_matrix.png
├── feature_importance_comparison.png
├── classical_models_comparison.png
├── zz_feature_map.png
├── complete_vqc_circuit.png
├── vqc_training_progress.png
├── quantum_vs_classical_comparison.png
├── radar_chart_comparison.png
└── performance_heatmap.png

models/
├── logistic_regression. pkl
├── random_forest. pkl
├── xgboost.pkl
├── neural_network.pkl
├── vqc_classifier.pkl
└── feature_maps.pkl
```

## 🎓 Dependencies

### Core Libraries

- **Qiskit** (1.0.0): Quantum computing framework
- **Qiskit Aer** (0.13.3): High-performance simulator
- **Qiskit Machine Learning** (0.7.2): QML algorithms
- **PennyLane** (0.35.0): Alternative quantum ML framework

### Classical ML

- **scikit-learn** (1.4.0): Classical ML algorithms
- **XGBoost** (2.0.3): Gradient boosting
- **imbalanced-learn** (0.12.0): Handling imbalanced data

### Data & Visualization

- **pandas**, **numpy**:  Data manipulation
- **matplotlib**, **seaborn**, **plotly**: Visualization
- **jupyter**: Interactive notebooks

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure virtual environment is activated
source qml_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**2. Memory Issues**
```python
# In notebooks, reduce training samples
n_train_samples = 200  # Instead of 500
n_test_samples = 100   # Instead of 200
```

**3. Qiskit Version Conflicts**
```bash
# Use exact versions from requirements.txt
pip install qiskit==1.0.0 qiskit-aer==0.13.3
```

## 📚 References

### Quantum Machine Learning

1.  Schuld, M., & Petruccione, F. (2018). *Supervised Learning with Quantum Computers*.  Springer.
2. Biamonte, J., et al. (2017). "Quantum machine learning." *Nature*, 549(7671), 195-202.
3. Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567(7747), 209-212.

### VQC and Ansatz Design

4. Cerezo, M., et al. (2021). "Variational quantum algorithms." *Nature Reviews Physics*, 3(9), 625-644.
5. Sim, S., et al. (2019). "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms." *Advanced Quantum Technologies*, 2(12), 1900070.

### Quantum Feature Maps

6. Pérez-Salinas, A., et al. (2020). "Data re-uploading for a universal quantum classifier." *Quantum*, 4, 226.
7. Schuld, M., & Killoran, N. (2019). "Quantum machine learning in feature Hilbert spaces." *Physical Review Letters*, 122(4), 040504.

## 🎯 Bonus Problem:  Quantum Neural Networks (QNN)

For the bonus task, explore: 
- Implementing QNN with multiple quantum layers
- Comparing QNN vs VQC performance
- Testing on additional datasets (medical, cybersecurity)

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different ansatz designs
- Try alternative optimizers
- Test on different datasets
- Implement noise models for realistic simulations

## 📄 License

This project is for educational purposes as part of a quantum computing assignment. 

## 👤 Author

**vishnubishnoi17**
- GitHub: [@vishnubishnoi17](https://github.com/vishnubishnoi17)

## 🙏 Acknowledgments

- IBM Quantum for Qiskit framework
- Xanadu for PennyLane
- UCI Machine Learning Repository for datasets

---

## 📊 Quick Results Summary

*This section will be populated after running all notebooks*

### Best Model by Metric

| Metric | Model | Score |
|--------|-------|-------|
| Accuracy | TBD | TBD |
| Precision | TBD | TBD |
| Recall | TBD | TBD |
| F1 Score | TBD | TBD |
| AUC-ROC | TBD | TBD |

### Quantum vs Classical

| Model Type | Best Accuracy | Parameters |
|------------|---------------|------------|
| Classical | TBD | ~1000s |
| Quantum (VQC) | TBD | 12-24 |

---

**Next Steps**:  Run the notebooks in order and update this README with your results! 
