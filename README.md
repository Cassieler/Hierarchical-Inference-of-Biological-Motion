# Hierarchical-Inference-of-Biological-Motion
Hierarchical Bayesian model for biological motion recognition with strong inductive biases.  Achieves 70% accuracy with just 5 training samples per action class.

# Hierarchical Motion Recognition: The Power of Correct Inductive Biases

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A hierarchical Bayesian approach to biological motion recognition that demonstrates 
the critical importance of correct structural priors for sample-efficient learning.

**Key Result**: Achieves **Achieves 70% accuracy with just 5 training samples per action class** by 
encoding domain knowledge about human kinematics into the model architecture.

---

## 🎯 Overview

This repository contains the implementation and experiments for our research on 
hierarchical motion recognition. We demonstrate that:

1. **Correct hierarchical priors enable few-shot learning**: 95% accuracy with 2 samples/class
2. **Wrong structural assumptions cannot be compensated by data**: 5-10× data efficiency gap
3. **Interpretable failures**: Ablation studies reveal mechanistic failure modes

### Model Architecture
```
Action → Global Motion → Limb Dynamics → Joint Positions
         [velocity,        [amplitudes,      [trajectories]
          frequency,        phases,
          oscillation]      coordination]
```

**Key Design Principles:**
- ✅ Pelvis as stable reference frame
- ✅ Two-layer hierarchy (global → limb)
- ✅ Periodic motion priors (sinusoidal fitting)
- ✅ Rigid body constraints
- ✅ Forward motion as primary direction

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/hierarchical-motion-recognition.git
cd hierarchical-motion-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Basic Usage
```python
from models import HierarchicalMotionModel
from biomation_utils import load_dataset

# Load data
data = load_dataset('your_path/biomation_full.npz')

# Prepare training data (just 2 samples per action!)
train_data = prepare_data(data, n_samples=2, seed=42)
test_data = prepare_data(data, n_samples=10, seed=999)

# Train model
model = HierarchicalMotionModel(n_components_limb=2)
model.fit(train_data)

# Evaluate
accuracy = evaluate_model(model, test_data)
print(f"Accuracy: {accuracy:.3f}")  # Expected: ~0.95

# Predict new sample
trajectory = test_data[0]['trajectory']  # (T, J, 2)
prediction = model.predict(trajectory)
probabilities = model.predict_proba(trajectory)
```

---

## 📁 Repository Structure
```
hierarchical-motion-recognition/
├── models.py                      # Core model implementations
│   ├── HierarchicalMotionModel    # Our hierarchical model
│   ├── GraphicalHierarchicalModel # Full Bayesian version
│   ├── FlatMotionModel            # PCA baseline
│   └── FlatBayesianModel          # Minimal baseline
│
├── wrong_hierarchy_models.py      # Ablation models
│   ├── WrongSkeletonModel         # Random reference frame
│   └── WrongKinematicsModel       # Vertical-first assumption
│
├── biomation_utils.py             # Data generation and utilities
│   ├── generate_walking()
│   ├── generate_running()
│   ├── generate_jumping()
│   └── ...
│
├── run_ablation_study.py          # Main ablation experiments
├── run_four_model_comparison.py   # Full model comparison
│
├── experiments.py                 # Experiment scripts
│
├── results/                       # Output directory (generated)
│   ├── ablation_results.pkl       # Saved experimental results
│   ├── ablation_study_comparison.png
│   ├── confusion_matrices_ablation.png
│   └── per_action_analysis.png
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # MIT License
```

## 🎓 Key Concepts

### Why Hierarchical?

**Problem:** Biological motion is high-dimensional (60 frames × 10 joints × 2D = 1,200 dimensions)

**Solution:** Hierarchical decomposition
```
Global motion (3D) → describes overall trajectory
  ↓
Limb motion (10 × 8D) → describes relative movements
  ↓
Joint positions (1,200D) → emerges from hierarchy
```

**Benefit:** Reduces effective dimensionality from O(T×J) to O(J), enabling learning from 2-10 samples.

### Why Correct Priors Matter

| Prior | Correct | Wrong | Impact |
|-------|---------|-------|--------|
| **Reference frame** | Pelvis (stable) | Random joint | -40% accuracy |
| **Motion direction** | Horizontal primary | Vertical primary | -30% accuracy |
| **Hierarchy depth** | 2 layers | 0 or 5+ layers | -20-40% accuracy |
| **Body model** | Rigid skeleton | Flexible/circular | -15-25% accuracy |

**Key Insight:** Wrong priors create fundamental representational mismatches that 
cannot be compensated by simply adding more training data.
