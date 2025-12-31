# 🛣️ Lane Estimation Using GPS Probe Data for OSM Imputation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Predicting road lane counts from GPS probe data to fill missing lane information in OpenStreetMap (OSM)**

## 📋 Project Overview

OpenStreetMap (OSM) is missing lane count information for **~70% of road segments** in many regions. This project develops a machine learning pipeline to **predict lane counts** using GPS probe data distributions, enabling large-scale imputation of missing lane data.

### 🎯 Key Results

| Model | Accuracy | Within ±1 Lane |
|-------|----------|----------------|
| **Random Forest** | **90.11%** | 98.7% |
| LightGBM | 84.56% | 97.2% |
| XGBoost | 85.0% | 96.8% |
| GMM (Baseline) | 22.0% | - |

## 🏗️ Architecture

```
GPS Probe Data → Lateral Distribution (40 bins) → ML Model → Lane Count Prediction
                                                      ↓
                                               OSM Imputation
```

## 📁 Project Structure

```
Lane-Estimation/
├── 📓 Notebooks
│   ├── Estimation_V4 (2).ipynb      # Main model training & evaluation
│   └── Florida_AADT_Analysis.ipynb  # AADT correlation analysis
│
├── 📊 Datasets
│   ├── la_link_dist_vs_lanes_estimation_dataset_v1.parquet  # LA GPS data
│   ├── florida_lanes_data/          # Florida DOT lane counts
│   └── florida_aadt_data/           # Florida DOT traffic data
│
├── 🐍 Scripts
│   ├── complete_analysis.py         # Error analysis & insights
│   ├── florida_aadt_model.py        # AADT-based modeling
│   ├── lightgbm_model.py            # LightGBM implementation
│   └── improve_accuracy_analysis.py # Accuracy improvement strategies
│
├── 📑 Presentations
│   └── Lane_Estimation_Presentation.pptx
│
└── 📄 Documentation
    └── README.md
```

## 🔬 Methodology

### Feature Engineering

1. **GPS Lateral Distribution (40 bins)**
   - Divide road width into 40 equal bins
   - Calculate ratio of GPS points in each bin
   - Captures lane patterns (peaks indicate lane centers)

2. **One-way Flag**
   - Binary indicator for one-way roads
   - Affects expected lane distribution patterns

3. **Traffic Volume (AADT)** *(Enhancement)*
   - Annual Average Daily Traffic
   - Strong correlation with lane count (r > 0.7)

### Model Training

```python
from sklearn.ensemble import RandomForestClassifier

# Features: 40 GPS distribution bins + oneway flag
features = [str(i) for i in range(40)] + ['oneway']

# Random Forest with class balancing
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
```

## 📈 Error Analysis

### Common Misclassifications

| True Lanes | Predicted | Count | Root Cause |
|------------|-----------|-------|------------|
| 2 | 3 | 145 | Turn lanes, parking |
| 3 | 2 | 132 | Low traffic density |
| 4 | 3 | 89 | GPS noise in median |

### Key Insights

- **2 vs 3 lane confusion**: Most common error (25% of mistakes)
- **Higher lanes = Lower accuracy**: 6+ lanes drop to ~60% accuracy
- **Traffic volume matters**: AADT can resolve ambiguous cases

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Quick Start

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_parquet('la_link_dist_vs_lanes_estimation_dataset_v1.parquet')

# Filter for quality
df = df[df['total_count'] > 1000]
df = df[df['lanes_int'] <= 7]

# Prepare features
features = [str(i) for i in range(40)] + ['oneway']
X = df[features]
y = df['lanes_int']

# Train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
```

## 📊 Datasets

### LA GPS Probe Data
- **Source**: Proprietary GPS data
- **Records**: ~50,000 road segments
- **Features**: GPS lateral distribution, total count, one-way flag
- **Labels**: Ground truth lane counts (1-7 lanes)

### Florida DOT Data
- **Source**: [Florida Geographic Data Library](https://www.fgdl.org/)
- **Lane Counts**: `number_of_lanes_oct25.shp` (86,880 segments)
- **Traffic Data**: `aadt_oct23.shp` (Annual Average Daily Traffic)

## 🔮 Future Work

1. **AADT Integration**: Add real traffic volume data for +3-5% accuracy
2. **Multi-Region Training**: Combine LA, Florida, and other regions
3. **Deep Learning**: CNN on GPS heatmaps
4. **Real-time Pipeline**: Streaming GPS processing

## 📚 References

1. He et al., "Lane-Level Street Map Extraction From Aerial Imagery", WACV 2022
2. OpenStreetMap Contributors, [openstreetmap.org](https://www.openstreetmap.org)
3. Florida DOT RCI Database, [fdot.gov](https://www.fdot.gov)

## 👥 Contributors

- Research Team - Lane Estimation Project

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>🛣️ Making maps more complete, one lane at a time</b>
</p>

