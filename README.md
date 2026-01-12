# tidal-forecasting-hierarchical-attention
Code and data for PLOS ONE paper on tidal forecasting
# Hierarchical Attention Network for Multi-Horizon Tidal Water Level Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Official implementation of the paper:**

> **"A hierarchical attention network with multi-scale temporal encoders for interpretable multi-horizon tidal water level forecasting"**  
> Yuchen Zhang  
> *Submitted to PLOS ONE, 2026*

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 🌊 Overview

This repository contains the complete implementation of a hierarchical attention network for accurate and interpretable tidal water level forecasting. Our method addresses key limitations of traditional harmonic analysis and existing deep learning approaches by:

- **Multi-scale temporal encoding**: Explicitly captures patterns at 6-hour (semidiurnal), 24-hour (diurnal), and 168-hour (fortnightly) scales
- **Hierarchical attention fusion**: Dynamically combines information across temporal scales
- **Horizon-specific prediction**: Adapts to different forecasting horizons (1-168 hours)
- **Physical interpretability**: Attention patterns align with established tidal constituents

### Performance Highlights

- **3.76% MAPE** for 24-hour predictions (14.1% improvement over PatchTST)
- **46%+ improvement** over traditional machine learning methods
- **Robust cross-station generalization** (σ=0.15% MAPE across 5 stations)
- **36.3% error reduction** in meteorological components

## ✨ Key Features

### Architecture
```
├── Multi-Scale Encoders (Layer 1-3)
│   ├── Layer 1: 6-hour receptive field (semidiurnal tides)
│   ├── Layer 2: 24-hour receptive field (diurnal tides)
│   └── Layer 3: 168-hour receptive field (fortnightly patterns)
├── Hierarchical Attention Fusion
│   └── Dynamic weight allocation across scales
└── Horizon-Specific Prediction Heads
    └── Specialized outputs for different forecast horizons
```

### Technical Highlights
- ✅ PyTorch implementation with GPU acceleration
- ✅ Modular design for easy extension
- ✅ Comprehensive evaluation metrics
- ✅ Visualization tools for interpretability analysis
- ✅ Pre-trained model checkpoints
- ✅ Reproducible experiments with fixed seeds

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration, optional)
- 8GB+ RAM (16GB+ recommended)

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yczhang-npu/tidal-forecasting-hierarchical-attention.git
cd tidal-forecasting-hierarchical-attention

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/yczhang-npu/tidal-forecasting-hierarchical-attention.git
cd tidal-forecasting-hierarchical-attention

# Create conda environment
conda env create -f environment.yml
conda activate tidal-forecasting

# Install the package
pip install -e .
```

## 🚀 Quick Start

### 1. Download Data

All data are publicly available from NOAA CO-OPS:

```bash
# Download NOAA tide gauge data for 5 stations
python scripts/download_data.py \
    --stations boston newyork charleston keywest sandiego \
    --start 2010-01-01 \
    --end 2023-12-31 \
    --output data/raw
```

**Data Sources:**
- **NOAA CO-OPS API**: https://api.tidesandcurrents.noaa.gov/api/prod/
- **NOAA Tides & Currents**: https://tidesandcurrents.noaa.gov/

### 2. Preprocess Data

```bash
# Preprocess and create train/val/test splits
python scripts/preprocess_data.py \
    --input data/raw \
    --output data/processed \
    --train-years 2010-2019 \
    --val-years 2020-2021 \
    --test-years 2022-2023
```

### 3. Train Model

```bash
# Train the hierarchical attention model
python src/training/train.py --config configs/model_config.yaml

# Or use the provided script
bash scripts/train_model.sh
```

### 4. Evaluate

```bash
# Evaluate on test set
python src/evaluation/evaluate.py \
    --checkpoint results/checkpoints/best_model.pth \
    --data data/processed/test_split.npz

# Generate visualizations
python src/evaluation/visualize_results.py
```

### 5. Interactive Demo

```bash
# Launch Jupyter notebook for interactive exploration
jupyter notebook notebooks/02_model_training.ipynb
```

## 📊 Data

### Data Sources

All data are publicly available from **NOAA CO-OPS** (Center for Operational Oceanographic Products and Services):

**Official Website**: https://tidesandcurrents.noaa.gov/  
**API Documentation**: https://api.tidesandcurrents.noaa.gov/api/prod/  
**Data License**: Public Domain (U.S. Government Work)

| Station | Location | Station ID | Latitude | Longitude | Data Period |
|---------|----------|------------|----------|-----------|-------------|
| Boston | Boston, MA | 8443970 | 42.354°N | 71.053°W | 2010-2023 |
| New York | New York, NY | 8518750 | 40.700°N | 74.015°W | 2010-2023 |
| Charleston | Charleston, SC | 8665530 | 32.782°N | 79.925°W | 2010-2023 |
| Key West | Key West, FL | 8724580 | 24.551°N | 81.808°W | 2010-2023 |
| San Diego | San Diego, CA | 9410170 | 32.714°N | 117.173°W | 2010-2023 |

### Data Features

Each sample includes:
- **Water Level**: Verified hourly height (meters, MLLW datum)
- **Predicted Tide**: Harmonic analysis predictions
- **Meteorological**: Wind speed/direction, air pressure, air temperature
- **Temporal**: Hour of day, day of week, day of year

### Data Access

#### Automated Download
```bash
python scripts/download_data.py --stations boston --start 2010-01-01 --end 2023-12-31
```

#### Manual Download via NOAA API
```python
import requests

def download_noaa_data(station_id, start_date, end_date, product='hourly_height'):
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        'station': station_id,
        'begin_date': start_date,
        'end_date': end_date,
        'product': product,
        'datum': 'MLLW',
        'time_zone': 'GMT',
        'units': 'metric',
        'format': 'json',
        'application': 'research_tidal_forecasting'
    }
    response = requests.get(base_url, params=params)
    return response.json()

# Example: Download Boston data
data = download_noaa_data('8443970', '20100101', '20231231')
```

#### Manual Download via Web Interface
1. Visit: https://tidesandcurrents.noaa.gov/
2. Click "Data Retrieval"
3. Select station and date range
4. Choose products (Water Levels, Predictions, Meteorological)
5. Download in CSV or JSON format

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for detailed documentation.

## 📖 Usage

### Training

```python
from src.models.hierarchical_attention import HierarchicalAttentionModel
from src.data.dataloader import TidalDataLoader
from src.training.train import Trainer

# Load data
dataloader = TidalDataLoader(
    data_path='data/processed',
    batch_size=32,
    sequence_length=168
)

# Initialize model
model = HierarchicalAttentionModel(
    input_dim=8,
    hidden_dim=128,
    num_layers=3,
    num_heads=8,
    dropout=0.1
)

# Train
trainer = Trainer(model, dataloader, device='cuda')
trainer.train(epochs=100, lr=0.001)
```

### Inference

```python
from src.models.hierarchical_attention import HierarchicalAttentionModel
import torch

# Load trained model
model = HierarchicalAttentionModel.load_from_checkpoint(
    'results/checkpoints/best_model.pth'
)

# Make predictions
with torch.no_grad():
    predictions = model(input_sequence)
```

### Visualization

```python
from src.evaluation.visualize import plot_attention_weights, plot_predictions

# Visualize attention patterns
plot_attention_weights(
    model, 
    sample_input,
    save_path='results/figures/attention_patterns.png'
)

# Plot predictions vs ground truth
plot_predictions(
    predictions, 
    ground_truth,
    save_path='results/figures/predictions.png'
)
```

See [docs/usage.md](docs/usage.md) for more examples.

## 📈 Results

### Quantitative Results

**24-Hour Forecast Performance (Multi-Station Average)**

| Model | MAPE (%) | RMSE (m) | MAE (m) | R² |
|-------|----------|----------|---------|-----|
| **Ours (Proposed)** | **3.76** | **0.142** | **0.089** | **0.985** |
| PatchTST | 4.38 | 0.165 | 0.103 | 0.979 |
| TimesNet | 4.52 | 0.171 | 0.107 | 0.977 |
| Autoformer | 4.89 | 0.184 | 0.116 | 0.973 |
| FEDformer | 4.78 | 0.180 | 0.113 | 0.974 |
| LSTM | 5.32 | 0.201 | 0.126 | 0.968 |
| XGBoost | 7.05 | 0.266 | 0.167 | 0.942 |
| Random Forest | 7.45 | 0.281 | 0.176 | 0.935 |

### Key Findings

1. **Multi-horizon performance**: Consistent improvements across all forecast horizons (1-168 hours)
2. **Cross-station generalization**: Low variance (σ=0.15% MAPE) across diverse tidal regimes
3. **Interpretability**: Attention patterns correlate with tidal constituents (M2: r=0.89, K1: r=0.82)
4. **Meteorological component**: 36.3% error reduction in non-astronomical variations

See [results/README.md](results/README.md) for detailed results and visualizations.

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{zhang2026tidal,
  title={A hierarchical attention network with multi-scale temporal encoders for interpretable multi-horizon tidal water level forecasting},
  author={Zhang, Yuchen},
  journal={PLOS ONE},
  year={2026},
  note={Submitted}
}
```

### BibTeX for Code Repository

```bibtex
@software{zhang2026tidal_code,
  author={Zhang, Yuchen},
  title={Hierarchical Attention Network for Tidal Forecasting - Code and Data},
  year={2026},
  publisher={GitHub},
  url={https://github.com/yczhang-npu/tidal-forecasting-hierarchical-attention},
  doi={10.5281/zenodo.XXXXXXX}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data License**: 
- Raw NOAA data: Public Domain (U.S. Government Work)
- Preprocessed data: CC BY 4.0 (Creative Commons Attribution 4.0 International)

**Code License**: MIT License

```
MIT License

Copyright (c) 2026 Yuchen Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 🙏 Acknowledgments

- **Data**: NOAA Center for Operational Oceanographic Products and Services for providing freely accessible tide gauge data (https://tidesandcurrents.noaa.gov/)
- **Baseline Implementations**: 
  - PatchTST: https://github.com/yuqinie98/PatchTST
  - TimesNet: https://github.com/thuml/TimesNet
  - Autoformer: https://github.com/thuml/Autoformer
  - FEDformer: https://github.com/MAZiqing/FEDformer

## 📧 Contact

**Author**: Yuchen Zhang  
**Email**: 2627556529@qq.com  
**Corresponding Author**: Yuchen Zhang (2627556529@qq.com)

**Issues**: Please use [GitHub Issues](https://github.com/yczhang-npu/tidal-forecasting-hierarchical-attention/issues) for bug reports and feature requests.

## 🔄 Updates

- **2026-01-12**: Initial release with complete code, data, and documentation

## 📚 Related Work

This work builds upon and compares with:

- [PatchTST](https://github.com/yuqinie98/PatchTST) - Nie et al. (2023)
- [TimesNet](https://github.com/thuml/TimesNet) - Wu et al. (2023)
- [Autoformer](https://github.com/thuml/Autoformer) - Wu et al. (2021)
- [FEDformer](https://github.com/MAZiqing/FEDformer) - Zhou et al. (2022)

## ⭐ Star History

If you find this work useful, please consider giving us a star ⭐!

---

**Maintained by**: Yuchen Zhang (2627556529@qq.com)  
**Last Updated**: January 12, 2026  
**Version**: 1.0.0
