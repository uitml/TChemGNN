# TChemGNN Research Implementation Guide

Code for the paper [Efficient Learning of Molecular Properties Using Graph Neural Networks Enhanced with Chemistry Knowledge](https://pubs.acs.org/doi/10.1021/acsomega.5c07178) by Tetiana Lutchyn, Marie Mardal, Benjamin Ricaud, [ChemRxiv version](https://chemrxiv.org/engage/chemrxiv/article-details/68693d7cc1cb1ecda0442439).

We build a simple GNN-based model that integrates chemistry knowledge that GNNs may have difficulties to learn. We show that this combination greatly enhances the accuracy compared with the pure GNN approach. With a simple approach, this study highlights some limitations of GNNs and the crucial benefit of giving GNN models easy access to global information about the graph in the context of applications to chemistry. We focus on regression tasks at the molecular level, on small-molecule data sets. 

## 🚀 Quick Reproduction

### 1. Environment Setup
```bash
# Automated setup
python setup_script.py

# Or manual installation
pip install torch torch-geometric rdkit numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 2. Dataset Preparation
```bash
# Download all benchmark datasets
python download_datasets.py

# Verify datasets
python download_datasets.py --verify_only
```

### 3. Single Experiment (Paper Results)
```bash
# Run individual datasets
python main.py --dataset esol      # Expected: 0.7844 RMSE
python main.py --dataset freesolv  # Expected: 1.0124 RMSE
python main.py --dataset lipophilicity # Expected: 1.0221 RMSE
python main.py --dataset bace      # Expected: 0.9586 RMSE
```

### 4. Complete Paper Reproduction
```bash
# Run ALL experiments from paper
python run_all_experiments.py

# Quick validation (reduced epochs)
python run_all_experiments.py --quick
```

---

## 📊 **Key Research Components**

### Core Model Implementation
```python
# models/gsr.py - TChemGNN architecture
class GSR(nn.Module):
    """
    35 features → 5 GAT layers → Node predictions
    Key: tanh activation, 28 hidden dims, ~3.7K params
    """
```

### Feature Engineering (35D vector)
```python
# data/preprocessing.py - Paper-specified features
def create_atom_features(smiles_list):
    """
    Returns 35 features per atom:
    - 14 atomic-level (degree, atomic_number, etc.)
    - 15 molecular-level (rings, aromaticity, etc.) 
    - 6 global 3D (volume, dipole, angle, etc.)
    Note: logP excluded (key paper finding)
    """
```

### Training Configuration
```python
# Paper-aligned training settings
SETTINGS = {
    'optimizer': 'RMSprop',
    'lr': 0.00075,
    'scheduler': None,        # Paper doesn't use
    'grad_clipping': None,    # Paper doesn't use
    'activation': 'tanh',
    'hidden_dim': 28,
    'epochs': 5000
}
```

---

## 🧪 **Research Experiments Available**

### 1. Main Benchmarks (Tables 1-4)
- **Command**: `python main.py --dataset {esol,freesolv,lipophilicity,bace}`
- **Purpose**: Reproduce state-of-the-art results
- **Expected**: Match or exceed large foundation models

### 2. Ablation Studies (Table 2)
- **Command**: `python ablation_studies.py --experiment ablation`
- **Purpose**: Analyze GNN architecture impact
- **Key Finding**: GAT + 3D features = optimal

### 3. Pooling Analysis (Table 5)
- **Command**: `python ablation_studies.py --experiment node_positions`
- **Purpose**: Compare pooling vs. no-pooling strategies
- **Key Finding**: Last node best for ESOL/FreeSolv

### 4. Feature Importance (Table 6)
- **Command**: Compare `--with/without` global features
- **Purpose**: Quantify 3D feature contribution
- **Key Finding**: 3D features crucial for performance

### 5. Random Forest Baseline
- **Command**: `python ablation_studies.py --experiment random_forest`
- **Purpose**: Validate against classical ML
- **Key Finding**: Expert features competitive with deep learning

---

## 🔍 **Research Insights & Extensions**

### Key Paper Contributions Implemented:
1. **Global 3D Features**: Molecular geometry critical for GNN performance
2. **No-Pooling Strategy**: SMILES-guided node selection outperforms pooling
3. **Efficiency**: 3.7K parameters match/beat million-parameter models
4. **Expert Knowledge**: Chemistry-informed design beats pure ML approaches

### Extension Opportunities:
```python
# Modify for your research
class CustomGSR(GSR):
    def __init__(self, custom_features=None):
        # Add your custom molecular features
        # Experiment with different architectures
        # Test on new datasets
```

### Research Questions Addressed:
- ✅ Can small models beat large foundation models?
- ✅ How important are 3D molecular features?
- ✅ When is pooling detrimental to GNN performance?
- ✅ What's the optimal GNN architecture for chemistry?

---

## 📈 **Performance Validation**

### Expected Results (RMSE):
| Dataset | Paper | Implementation |
|---------|-------|----------------|
| ESOL | 0.7844 | ✅ Reproducible |
| FreeSolv | 1.0124 | ✅ Reproducible |
| Lipophilicity | 1.0221 | ✅ Reproducible |
| BACE | 0.9586 | ✅ Reproducible |

### Computational Requirements:
- **Training Time**: ~30 minutes per dataset (CPU)
- **Memory**: <2GB RAM
- **Storage**: <1GB for all datasets
- **Hardware**: Runs on laptop (GPU optional)

---

## 🛠️ **Research Modifications**

### Easy Customizations:
```bash
# Different architectures
python main.py --dataset esol --hidden_dim 64 --heads 4

# Training variations  
python main.py --dataset esol --epochs 1000 --lr 0.001

# Pooling experiments
python main.py --dataset esol --pooling_strategy mean
```

### Advanced Research:
- **New Features**: Modify `create_atom_features()` in `data/preprocessing.py`
- **New Architectures**: Extend `GSR` class in `models/gsr.py`  
- **New Datasets**: Add config to `DATASET_CONFIG` in `main.py`
- **New Experiments**: Follow patterns in `ablation_studies.py`

---

## 📚 **Research Output**

### Generated Artifacts:
- **Training Curves**: Loss/RMSE progression
- **Predictions**: Molecule-level results with errors
- **Visualizations**: Interactive training animations
- **Analysis**: Node-level prediction breakdowns
- **Reports**: Comprehensive experiment summaries

### Publication-Ready Results:
- All tables from paper reproducible
- Statistical significance validated
- Performance comparisons with SOTA
- Ablation studies quantified
- Runtime/efficiency analysis included

---

## ⚠️ **Research Notes**

### Critical Implementation Details:

1. **SMILES Ordering**: Leverages encoding structure for node selection
2. **Molecular fingerprint Computation**: Uses RDKit for molecular properties and add them as node features

### Reproducibility:
- Fixed random seeds throughout
- Deterministic operations
- Version-controlled dependencies
- Paper-exact hyperparameters

This implementation provides a complete research platform for exploring GNN-based molecular property prediction with chemistry-informed design principles.

---

## Credits

This code was written by Tetiana Lutchyn and refactored by Sebastian Iversen. The models were designed by Tetiana Lutchyn and Benjamin Ricaud. Training and experiments of the model(s) were performed by Tetiana Lutchyn. We thank Claude 3.7 for its help cleaning and refactoring the code.

