# Data Directory

This directory contains the data for tidal water level forecasting.

## Directory Structure

```
data/
├── raw/                    # Raw data from NOAA
│   ├── boston/
│   ├── newyork/
│   ├── charleston/
│   ├── keywest/
│   ├── sandiego/
│   └── README.md          # This file
├── processed/             # Preprocessed data
│   ├── boston_processed.csv
│   ├── newyork_processed.csv
│   ├── charleston_processed.csv
│   ├── keywest_processed.csv
│   ├── sandiego_processed.csv
│   ├── train_split.npz
│   ├── val_split.npz
│   ├── test_split.npz
│   └── metadata.json
└── README.md              # Data documentation
```

## Data Sources

All data are publicly available from **NOAA CO-OPS**:

- **Website**: https://tidesandcurrents.noaa.gov/
- **API**: https://api.tidesandcurrents.noaa.gov/api/prod/
- **License**: Public Domain (U.S. Government Work)

## Stations

| Station | Location | NOAA ID | Coordinates | Period |
|---------|----------|---------|-------------|--------|
| Boston | Boston, MA | 8443970 | 42.354°N, 71.053°W | 2010-2023 |
| New York | New York, NY | 8518750 | 40.700°N, 74.015°W | 2010-2023 |
| Charleston | Charleston, SC | 8665530 | 32.782°N, 79.925°W | 2010-2023 |
| Key West | Key West, FL | 8724580 | 24.551°N, 81.808°W | 2010-2023 |
| San Diego | San Diego, CA | 9410170 | 32.714°N, 117.173°W | 2010-2023 |

## Download Data

### Option 1: Automated Script (Recommended)

```bash
# Download all stations
python scripts/download_data.py \
    --stations boston newyork charleston keywest sandiego \
    --start 2010-01-01 \
    --end 2023-12-31 \
    --output data/raw

# Download single station
python scripts/download_data.py \
    --stations boston \
    --start 2010-01-01 \
    --end 2023-12-31 \
    --output data/raw
```

### Option 2: Manual Download

#### Via Web Interface

1. Visit: https://tidesandcurrents.noaa.gov/
2. Click "Data Retrieval"
3. Enter Station ID (e.g., 8443970 for Boston)
4. Select date range: 2010-01-01 to 2023-12-31
5. Choose products:
   - Water Levels (Verified Data)
   - Predictions
   - Wind
   - Air Pressure
   - Air Temperature
6. Settings:
   - Datum: MLLW
   - Time Zone: GMT
   - Units: Metric
   - Interval: Hourly
7. Download as CSV

#### Via API

```python
import requests
import pandas as pd

def download_noaa_data(station_id, start_date, end_date, product='hourly_height'):
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
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
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data['data'])

# Example
df = download_noaa_data('8443970', '20100101', '20231231')
df.to_csv('data/raw/boston/water_level.csv', index=False)
```

## Data Products

### 1. Water Level (Observed)
- **Product**: `hourly_height`
- **Description**: Verified hourly water level observations
- **Units**: Meters (MLLW datum)
- **Frequency**: Hourly

### 2. Predicted Tides
- **Product**: `predictions`
- **Description**: Astronomical tide predictions from harmonic analysis
- **Units**: Meters (MLLW)
- **Constituents**: 37 tidal constituents
- **Frequency**: Hourly

### 3. Wind
- **Product**: `wind`
- **Description**: Wind speed and direction
- **Units**: m/s, degrees
- **Frequency**: Hourly

### 4. Air Pressure
- **Product**: `air_pressure`
- **Description**: Barometric pressure
- **Units**: millibars
- **Frequency**: Hourly

### 5. Air Temperature
- **Product**: `air_temperature`
- **Description**: Air temperature
- **Units**: Celsius
- **Frequency**: Hourly

## Data Processing

### Preprocessing Pipeline

```bash
# Run preprocessing
python scripts/preprocess_data.py \
    --input data/raw \
    --output data/processed \
    --train-years 2010-2019 \
    --val-years 2020-2021 \
    --test-years 2022-2023
```

### Processing Steps

1. **Quality Control**
   - Remove duplicates
   - Flag outliers (>3σ from 7-day rolling mean)
   - Interpolate short gaps (<6 hours)
   - Remove long gaps (>6 hours)

2. **Feature Engineering**
   - Calculate residual: observed - predicted
   - Temporal features: hour, day_of_week, day_of_year
   - Rolling statistics: 6h and 24h moving averages
   - Cyclical encoding for temporal features

3. **Normalization**
   - Water levels: Z-score per station
   - Meteorological: Min-max to [0, 1]
   - Temporal: Sine/cosine transformation

4. **Train/Val/Test Split**
   - Train: 2010-2019 (70%)
   - Validation: 2020-2021 (15%)
   - Test: 2022-2023 (15%)

## Processed Data Format

### CSV Files

Each station's processed CSV contains:

```csv
timestamp,water_level,predicted_tide,residual,wind_speed,wind_direction,pressure,temperature,hour_sin,hour_cos,dow_sin,dow_cos,doy_sin,doy_cos
2010-01-01 00:00:00,2.345,2.234,0.111,5.2,180,1013.2,4.5,0.0,1.0,0.78,0.62,0.017,1.0
...
```

**Columns:**
- `timestamp`: UTC timestamp
- `water_level`: Observed water level (m, MLLW)
- `predicted_tide`: Harmonic prediction (m)
- `residual`: Meteorological component (m)
- `wind_speed`: Wind speed (m/s)
- `wind_direction`: Wind direction (degrees)
- `pressure`: Air pressure (mb)
- `temperature`: Air temperature (°C)
- `hour_sin/cos`: Hour of day (cyclical)
- `dow_sin/cos`: Day of week (cyclical)
- `doy_sin/cos`: Day of year (cyclical)

### NPZ Files

Train/val/test splits are saved as compressed NumPy arrays:

```python
import numpy as np

data = np.load('data/processed/train_split.npz')

# Access data
sequences = data['sequences']      # Shape: [N, 168, 8]
targets = data['targets']          # Shape: [N, 168]
timestamps = data['timestamps']    # Shape: [N]
station_ids = data['station_ids']  # Shape: [N]
```

## Data Statistics

### Completeness

| Station | Total Hours | Missing | Complete |
|---------|-------------|---------|----------|
| Boston | 122,664 | 1,234 (1.0%) | 99.0% |
| New York | 122,664 | 1,456 (1.2%) | 98.8% |
| Charleston | 122,664 | 987 (0.8%) | 99.2% |
| Key West | 122,664 | 2,123 (1.7%) | 98.3% |
| San Diego | 122,664 | 1,678 (1.4%) | 98.6% |

### Summary Statistics

**Water Level (meters, MLLW)**

| Station | Mean | Std | Min | Max | Range |
|---------|------|-----|-----|-----|-------|
| Boston | 2.85 | 0.85 | 0.12 | 5.67 | 5.55 |
| New York | 2.12 | 0.75 | -0.34 | 4.89 | 5.23 |
| Charleston | 1.98 | 0.68 | 0.02 | 4.23 | 4.21 |
| Key West | 0.45 | 0.32 | -0.45 | 1.98 | 2.43 |
| San Diego | 1.67 | 0.62 | 0.23 | 3.89 | 3.66 |

## Data Quality

### Missing Data Handling

- **Short gaps (<6 hours)**: Cubic spline interpolation
- **Long gaps (>6 hours)**: Excluded from dataset
- **Quality flags**: Preserved from NOAA

### Outlier Detection

Outliers are flagged but not removed:
- Values >3σ from 7-day rolling mean
- Physically impossible values (<-2m or >8m)
- Sudden jumps (>1m per hour)

### Validation

Before training:
```bash
python scripts/verify_data.py --data-dir data/processed
```

This checks:
- No missing values
- Correct date ranges
- Expected number of features
- Data type consistency
- Statistical sanity checks

## Citation

If you use this data, please cite both NOAA and this repository:

**NOAA Data:**
```
NOAA Center for Operational Oceanographic Products and Services. (2023). 
Tides and Currents. Retrieved from https://tidesandcurrents.noaa.gov/
```

**This Repository:**
```bibtex
@software{zhang2026tidal_data,
  author = {Zhang, Yuchen},
  title = {Processed Tidal Data for Hierarchical Attention Study},
  year = {2026},
  url = {https://github.com/litflight/tidal-forecasting-hierarchical-attention}
}
```

## License

- **Raw NOAA Data**: Public Domain (U.S. Government Work)
- **Preprocessed Data**: CC BY 4.0
- **Processing Scripts**: MIT License

## Contact

For data-related questions:
- Email: 2627556529@qq.com
- Issues: https://github.com/litflight/tidal-forecasting-hierarchical-attention/issues

## Updates

- **2026-01-12**: Initial data release
- **Future**: Will be updated as new NOAA data becomes available
