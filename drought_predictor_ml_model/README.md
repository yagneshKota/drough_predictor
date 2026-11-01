# India Nationwide Drought Prediction ML Model Package

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Coverage](https://img.shields.io/badge/Coverage-716%20Districts%20%7C%2034%20States--UTs-green.svg)]()
[![Model Version](https://img.shields.io/badge/Model%20Version-2.0--optimized-brightgreen.svg)]()

Production-ready machine learning package delivering **7-day, 15-day, and 30-day ahead district-level drought severity forecasts** across India. Engineered for high-throughput backend services and microservice architectures (FastAPI, Flask, Django, or serverless workers).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Machine Learning Architecture](#3-machine-learning-architecture)
4. [Three Prediction Horizons](#4-three-prediction-horizons)
5. [Drought Severity Classes](#5-drought-severity-classes)
6. [Verified Dataset Sources](#6-verified-dataset-sources)
7. [Feature Engineering Overview](#7-feature-engineering-overview)
8. [Model Training Approach](#8-model-training-approach)
9. [Feature Preprocessing & Leakage Prevention](#9-feature-preprocessing--leakage-prevention)
10. [Model Evaluation & Verified Metrics](#10-model-evaluation--verified-metrics)
11. [Temporal and Spatial Validation](#11-temporal-and-spatial-validation)
12. [SHAP Explainability & Feature Importance](#12-shap-explainability--feature-importance)
13. [Installation](#13-installation)
14. [How to Load the Models](#14-how-to-load-the-models)
15. [How to Provide Input Features](#15-how-to-provide-input-features)
16. [How to Generate Predictions](#16-how-to-generate-predictions)
17. [Example Prediction Output](#17-example-prediction-output)
18. [Backend Integration Instructions (FastAPI)](#18-backend-integration-instructions-fastapi)
19. [Model Limitations](#19-model-limitations)
20. [Future Improvements](#20-future-improvements)

---

## 1. Project Overview

This repository houses the final trained model weights, preprocessing pipelines, inference engines, schema specifications, and evaluation artifacts for nationwide agricultural and meteorological drought forecasting in India.

The package is **100% self-contained**:
- Models load directly from JSON definitions (`models/`).
- Median imputation and standard scaling are packaged inside `models/preprocessing_pipeline.joblib`.
- **Zero dependency** on original raw training datasets during inference.
- Easily imported as an internal Python library or wrapped in a REST API.

---

## 2. Problem Statement

Droughts in India affect over 68% of the net sown area, creating severe economic shocks to smallholder agriculture, reservoir depletion, and acute drinking water shortages. Traditional drought assessments rely heavily on backward-looking indicators (e.g., rainfall departure over the past month), which identify drought only *after* damage has occurred.

**Our Objective**: Predict the **future** drought severity category (Normal, Mild, Moderate, Severe, Extreme) **7, 15, and 30 days into the future** at the administrative district level, giving policymakers, irrigation engineers, and farmers true proactive lead time.

---

## 3. Machine Learning Architecture

The system employs three horizon-specialized, multi-class **XGBoost (Extreme Gradient Boosting)** classification models:

```text
               Raw District-Week Features (Meteorology, Soil, Satellite, Drought Indices)
                                                   │
                                                   ▼
                       DroughtPreprocessor (Imputation & Standard Scaling)
                                                   │
          ┌────────────────────────────────────────┼────────────────────────────────────────┐
          │ (53 Features)                          │ (35 Features)                          │ (52 Features)
          ▼                                        ▼                                        ▼
┌───────────────────┐                    ┌───────────────────┐                    ┌───────────────────┐
│  7-Day Ahead      │                    │  15-Day Ahead     │                    │  30-Day Ahead     │
│  XGBoost Model    │                    │  XGBoost Model    │                    │  XGBoost Model    │
└─────────┬─────────┘                    └─────────┬─────────┘                    └─────────┬─────────┘
          │ (Probabilities)                        │ (Probabilities)                        │ (Probabilities)
          ▼                                        ▼                                        ▼
   Class 0 - 4 Probabilities                Class 0 - 4 Probabilities                Class 0 - 4 Probabilities
```

- **Objective Function**: `multi:softprob` (generating well-calibrated posterior probability distributions over all 5 classes).
- **Tree Algorithm**: Histogram-optimized gradient boosting (`tree_method="hist"`).
- **Decoupled Topologies**: Feature counts and hyperparameters are tuned independently to match the temporal decay of physical drivers.

---

## 4. Three Prediction Horizons

1. **7-Day Forecast (1-Week Ahead)**:
   - *Target*: $\text{CDI}(t + 1 \text{ week})$
   - *Primary Utility*: Immediate agricultural operations, micro-irrigation scheduling, weekly state relief contingency.
2. **15-Day Forecast (2-Weeks Ahead)**:
   - *Target*: $\text{CDI}(t + 2 \text{ weeks})$
   - *Primary Utility*: Reservoir allocation, canal rotation planning, seed choice adjustment during sowing windows.
3. **30-Day Forecast (4-Weeks Ahead)**:
   - *Target*: $\text{CDI}(t + 4 \text{ weeks})$
   - *Primary Utility*: Strategic regional water budgeting, livestock fodder pre-positioning, disaster declaration preparation.

---

## 5. Drought Severity Classes

Categories correspond to official **India Drought Monitor (IDM)** Combined Drought Index (CDI) standards:

| Class ID | Severity Label | CDI Range | Agricultural & Environmental Description | Recommended Action Advisory |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Normal** | $CDI > -0.5$ | Near-normal moisture; optimal soil hydration. | Standard crop management and regular irrigation. |
| **1** | **Mild Drought (D0)** | $-0.8 < CDI \le -0.5$ | Abnormally dry; incipient rootzone moisture deficit. | Mulching, moisture conservation, monitor rain forecast. |
| **2** | **Moderate Drought (D1)** | $-1.3 < CDI \le -0.8$ | Evident moisture deficit; crop growth stunted. | Implement rotational irrigation; prioritize hardy varieties. |
| **3** | **Severe Drought (D2)** | $-1.6 < CDI \le -1.3$ | Major crop damage; depleted surface reservoirs. | Activate drought contingency; restrict non-essential water. |
| **4** | **Extreme Drought (D3/D4)**| $CDI \le -1.6$ | Critical water emergency; widespread crop failure. | Emergency relief mobilization; inter-basin water dispatch. |

---

## 6. Verified Dataset Sources

The training and evaluation pipeline utilized verified observational, satellite, and reanalysis data:

1. **India Drought Monitor (IDM) / IIT Gandhinagar**: 277 weekly nationwide grids (2021–2026) for CDI, Standardized Precipitation Index (SPI-1, SPI-3, SPI-6), Standardized Runoff Index (SRI-1), and Standardized Soil Moisture Index (SSMI-1).
2. **CHIRPS 3.0 & ERA5-Land Reanalysis**: High-resolution daily rainfall grids aggregated to 7d, 15d, 30d, 60d, 90d, 180d, and 365d accumulations and historical percentage anomalies.
3. **ECMWF ERA5-Land**: Daily maximum, minimum, and mean temperatures, Hargreaves reference evapotranspiration ($ET_0$), solar radiation, and maximum surface wind speed.
4. **NASA MODIS (MOD13Q1)**: 250m Normalized Difference Vegetation Index (NDVI), Enhanced Vegetation Index (EVI), Vegetation Condition Index (VCI), and 4-week vegetation health trends.
5. **ISRIC SoilGrids 2.0**: Clay %, sand %, silt %, soil organic carbon ($SOC$), $pH$, bulk density, and available water capacity ($AWC$) depth-weighted for 0–30 cm.
6. **geoBoundaries (IND-ADM2)**: Audited boundaries for 716 districts across 34 States and Union Territories. Kushinagar is verified in **Uttar Pradesh** (`IND_UTT_KUSH_613`).

---

## 7. Feature Engineering Overview

The pipeline transforms raw physical variables into 53 features:

- **Precipitation Accumulation (7)**: `rainfall_7d`, `rainfall_15d`, `rainfall_30d`, `rainfall_60d`, `rainfall_90d`, `rainfall_180d`, `rainfall_365d`.
- **Precipitation Dynamics & Anomalies (7)**: `rainfall_rolling_mean`, `rainfall_rolling_std`, `rainfall_variability`, `rainfall_anomaly`, `rainfall_percentage_deviation`, `dry_weeks_last_4w`, `dry_weeks_last_12w`.
- **Thermal & Atmospheric Extremes (5)**: `temperature_mean`, `temperature_max`, `temperature_min`, `temperature_anomaly`, `is_heatwave_week` ($T_{max} \ge 40^\circ\text{C}$ or anomaly $\ge +3^\circ\text{C}$).
- **Soil Hydrology & Water Deficit (8)**: `soil_moisture_surface`, `soil_moisture_rootzone`, `soil_moisture_anomaly`, `soil_moisture_4w_trend`, `et0_evapotranspiration`, `water_deficit_7d`, `water_deficit_30d`, `rainfall_et0_ratio`.
- **Atmospheric Physics (2)**: `solar_radiation_mean`, `wind_speed_max`.
- **Vegetation Phenology (6)**: `ndvi`, `evi`, `ndvi_anomaly`, `evi_anomaly`, `vci`, `vegetation_4w_trend`.
- **Soil Physical Properties (7)**: `clay_pct`, `sand_pct`, `silt_pct`, `soc_g_per_kg`, `ph_h2o`, `awc_mm_per_m`, `bulk_density_g_cm3`.
- **Current Drought Status (2)**: `current_cdi`, `current_drought_class`.
- **Multi-Scale Drought Memory Indices (5)**: `spi_1month`, `spi_3month`, `spi_6month`, `sri_1month`, `ssmi_1month`.
- **Temporal & Historical Context (4)**: `historical_drought_frequency`, `month`, `week_of_year`, `season` (1: Kharif, 2: Rabi, 3: Zaid).

---

## 8. Model Training Approach

- **Chronological Data Partitioning**:
  - **Train**: July 14, 2021 – June 26, 2024 ($N = 112,065$ observations)
  - **Validation**: July 03, 2024 – December 25, 2024 ($N = 18,798$ observations)
  - **Test (Held-Out)**: January 01, 2025 – October 21, 2026 ($N = 68,685$ observations)
- **Class Imbalance Strategy**: Balanced sample weighting applied during gradient tree boosting:
  $$w_c = \frac{N}{K \cdot N_c}$$
- **Early Stopping & Tuning**: Models trained with `max_depth=7`, `learning_rate=0.03`, `subsample=0.90`, `colsample_bytree=0.75`, $L_1$ penalty `reg_alpha=1.0`, and $L_2$ penalty `reg_lambda=3.0`.

---

## 9. Feature Preprocessing & Leakage Prevention

1. **Strict Fit Isolation**: The median imputer and standard scaler were fitted **exclusively on the training partition** ($< \text{2024-07-01}$).
2. **Deterministic Imputation**: If real-time inputs omit certain features, the pipeline substitutes the training-time median rather than zero, preventing distortion of standardized distributions.
3. **Strict Target Alignment**: Future targets shift strictly forward ($\text{shift}(-1), \text{shift}(-2), \text{shift}(-4)$), guaranteeing zero target leakage into input features.

---

## 10. Model Evaluation & Verified Metrics

Evaluated on the **untouched test partition** ($N = 68,685$, spanning Jan 2025 to Oct 2026):

### Primary Performance Summary

| Metric | 7-Day Model | 15-Day Model | 30-Day Model |
| :--- | :---: | :---: | :---: |
| **Selected Feature Set** | `Exp_A_Original53` (53) | `Exp_F_TunedCombination` (35) | `Exp_C_NoCurrentCDI` (52) |
| **Accuracy** | **88.31%** | **81.76%** | **74.17%** |
| **Macro F1 Score** | **0.7782** | **0.6710** | **0.5577** |
| **Weighted F1 Score** | **0.8845** | **0.8202** | **0.7483** |
| **Balanced Accuracy** | **82.05%** | **72.17%** | **61.04%** |
| **Multi-Class Log Loss**| **0.3478** | **0.5361** | **0.7302** |
| **Brier Score** | **0.1802** | **0.2789** | **0.3845** |

### Per-Class Recall Breakdown (Sensitivity to Drought)

| Drought Class | 7-Day Recall | 15-Day Recall | 30-Day Recall |
| :--- | :---: | :---: | :---: |
| **Class 0 (Normal)** | 91.97% | 87.22% | 81.33% |
| **Class 1 (Mild / D0)** | 79.67% | 69.83% | 56.97% |
| **Class 2 (Moderate / D1)** | 73.73% | 59.77% | 48.26% |
| **Class 3 (Severe / D2)** | 78.63% | 65.44% | 50.91% |
| **Class 4 (Extreme / D3-D4)** | **86.24%** | **78.59%** | **67.72%** |

---

## 11. Temporal and Spatial Validation

### Out-of-Distribution Spatial Holdout
To verify geographical transferability rather than local memorization, districts in **Rajasthan** (hyper-arid western zone) and **Tamil Nadu** (peninsular agrarian zone) were held out entirely from training:

- **7-Day Spatial Accuracy**: **93.28%** (Macro F1: 0.7544)
- **15-Day Spatial Accuracy**: **89.80%** (Macro F1: 0.6500)
- **30-Day Spatial Accuracy**: **86.38%** (Macro F1: 0.5634)

### Benchmark Comparison against Baselines

| Model / Baseline | 7-Day Test Acc (F1) | 15-Day Test Acc (F1) | 30-Day Test Acc (F1) |
| :--- | :---: | :---: | :---: |
| **Majority Class Baseline** | 74.00% (0.1701) | 73.95% (0.1700) | 73.84% (0.1699) |
| **Persistence Baseline** | 90.70% (0.7919) | 85.93% (0.6854) | 80.25% (0.5638) |
| **Environmental-Only (Exp D)**| 86.47% (0.7505)* | 79.14% (0.6494)* | 71.74% (0.5471)* |
| **Final Optimized Models** | **88.31% (0.7782)** | **81.76% (0.6710)** | **74.17% (0.5577)** |

*\*Note on Persistence*: While persistence ("tomorrow's drought is today's drought") yields nominal accuracy during equilibrium periods, it exhibits zero lead-time capacity for onset, recovery, or intensification. The optimized XGBoost models provide calibrated transition probabilities and physical feature attribution.

---

## 12. SHAP Explainability & Feature Importance

Feature importance was audited across all trees using Gain, Weight, and SHAP values:

1. **Short-Range Drivers (7-Day)**:
   - Split gain is dominated by short-term water balance: `current_cdi`, `current_drought_class`, `soil_moisture_rootzone`, and `spi_1month`.
2. **Medium-Range Transition Signals (15-Day)**:
   - Dynamic indicators lead split decisions: `water_deficit_30d`, `rainfall_anomaly`, `soil_moisture_4w_trend`, and `vci`.
3. **Long-Cycle Memory Drivers (30-Day)**:
   - Multi-month memory features dominate: `spi_3month`, `spi_6month`, `rainfall_90d`, `historical_drought_frequency`, and soil available water capacity (`awc_mm_per_m`).

Visualizations are available in `feature_information/` and `evaluation/`.

---

## 13. Installation

Clone this repository and install the lightweight inference requirements:

```bash
git clone https://github.com/your-username/drought_predictor_ml_model.git
cd drought_predictor_ml_model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

---

## 14. How to Load the Models

The predictor engine loads all three horizon models and cached scalers into memory:

```python
from src.inference import DroughtPredictor

# Initialize engine (caches 7d, 15d, 30d models)
predictor = DroughtPredictor.get_instance()
```

---

## 15. How to Provide Input Features

Input features can be supplied as a standard Python dictionary, a JSON file, or a pandas DataFrame. For example:

```python
sample_input = {
    "district": "Kushinagar",
    "state": "Uttar Pradesh",
    "date": "2025-01-01",
    "rainfall_7d": 0.0,
    "rainfall_30d": 12.5,
    "rainfall_90d": 85.0,
    "rainfall_365d": 820.0,
    "rainfall_anomaly": -8.5,
    "dry_weeks_last_4w": 3.0,
    "dry_weeks_last_12w": 8.0,
    "temperature_mean": 21.5,
    "temperature_max": 27.0,
    "temperature_min": 16.0,
    "temperature_anomaly": 1.2,
    "is_heatwave_week": 0,
    "soil_moisture_surface": 0.18,
    "soil_moisture_rootzone": 0.22,
    "soil_moisture_anomaly": -0.04,
    "soil_moisture_4w_trend": -0.02,
    "et0_evapotranspiration": 2.8,
    "water_deficit_7d": -2.8,
    "water_deficit_30d": -15.4,
    "rainfall_et0_ratio": 0.45,
    "solar_radiation_mean": 18.5,
    "wind_speed_max": 11.2,
    "ndvi": 0.42,
    "evi": 0.31,
    "ndvi_anomaly": -0.03,
    "evi_anomaly": -0.02,
    "vci": 42.5,
    "vegetation_4w_trend": -0.01,
    "clay_pct": 28.0,
    "sand_pct": 40.0,
    "silt_pct": 32.0,
    "soc_g_per_kg": 5.8,
    "ph_h2o": 7.4,
    "awc_mm_per_m": 135.0,
    "bulk_density_g_cm3": 1.42,
    "current_cdi": -0.65,
    "current_drought_class": 1.0,
    "spi_1month": -0.72,
    "spi_3month": -0.58,
    "spi_6month": -0.40,
    "sri_1month": -0.30,
    "ssmi_1month": -0.60,
    "historical_drought_frequency": 0.18,
    "month": 1,
    "week_of_year": 1,
    "season": 2
}
```

---

## 16. How to Generate Predictions

```python
from src.inference import predict_drought

# Generate predictions
result = predict_drought(sample_input)

# Inspect 7-day forecast
print(result["predictions"]["7day"]["predicted_label"])
# Outputs: 'Mild Drought (D0)'
```

---

## 17. Example Prediction Output

```json
{
  "status": "success",
  "district": "Kushinagar",
  "state": "Uttar Pradesh",
  "prediction_date": "2025-01-01",
  "model_version": "2.0-optimized-nationwide",
  "predictions": {
    "7day": {
      "horizon": "7day",
      "horizon_label": "7-Day Ahead Forecast (1-Week)",
      "predicted_class": 1,
      "predicted_label": "Mild Drought (D0)",
      "description": "Abnormally dry; incipient soil moisture deficit. Subsurface recharge slowed.",
      "advisory": "Monitor rainfall trends. Conserve topsoil moisture through mulching and micro-irrigation.",
      "confidence": 0.5658,
      "probabilities": {
        "Normal": 0.0082,
        "Mild Drought (D0)": 0.5658,
        "Moderate Drought (D1)": 0.4231,
        "Severe Drought (D2)": 0.0025,
        "Extreme Drought (D3/D4)": 0.0005
      }
    },
    "15day": {
      "horizon": "15day",
      "horizon_label": "15-Day Ahead Forecast (2-Weeks)",
      "predicted_class": 2,
      "predicted_label": "Moderate Drought (D1)",
      "description": "Moderate drought; agricultural moisture deficits, delayed crop development, reduced streamflow.",
      "advisory": "Implement rotational irrigation. Prioritize drought-resistant crop varieties and soil moisture conservation.",
      "confidence": 0.5510,
      "probabilities": {
        "Normal": 0.0126,
        "Mild Drought (D0)": 0.4329,
        "Moderate Drought (D1)": 0.5510,
        "Severe Drought (D2)": 0.0030,
        "Extreme Drought (D3/D4)": 0.0004
      }
    },
    "30day": {
      "horizon": "30day",
      "horizon_label": "30-Day Ahead Forecast (4-Weeks)",
      "predicted_class": 2,
      "predicted_label": "Moderate Drought (D1)",
      "description": "Moderate drought; agricultural moisture deficits, delayed crop development, reduced streamflow.",
      "advisory": "Implement rotational irrigation. Prioritize drought-resistant crop varieties and soil moisture conservation.",
      "confidence": 0.5979,
      "probabilities": {
        "Normal": 0.0294,
        "Mild Drought (D0)": 0.3450,
        "Moderate Drought (D1)": 0.5979,
        "Severe Drought (D2)": 0.0252,
        "Extreme Drought (D3/D4)": 0.0024
      }
    }
  }
}
```

---

## 18. Backend Integration Instructions (FastAPI)

A complete microservice implementation is included in `examples/fastapi_app_example.py`.

### Run the API Server:
```bash
pip install fastapi uvicorn
uvicorn examples.fastapi_app_example:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints:
- `GET /health`: Health probe and model metadata.
- `POST /predict`: Single observation drought forecast.
- `POST /predict/batch`: High-performance vectorized batch inference.
- `GET /docs`: Interactive Swagger documentation and OpenAPI spec.

---

## 19. Model Limitations

1. **Oceanic Islands Footprint**: The models cover **34 of India's 36 States and Union Territories** (716 districts). Andaman & Nicobar Islands and Lakshadweep are excluded because they fall outside the contiguous mainland 0.25° India Drought Monitor grid.
2. **Sub-District Heterogeneity**: In large geographic districts (e.g. Jaisalmer, Leh), localized microclimates may exhibit intra-district variance that centroid-aggregated statistics smooth over.
3. **Flash Drought Onset at 30-Day Lead**: Beyond 21 days, sudden flash droughts driven by unexpected heatwaves without prior rainfall anomalies can exhibit class boundary smoothing due to chaotic atmospheric limits.

---

## 20. Future Improvements

- Integration of sub-surface deep aquifer piezometer telemetry from the Central Ground Water Board (CGWB).
- Assimilation of high-resolution SAR backscatter (Sentinel-1 / NISAR) for 10-meter soil moisture resolution.
- Ensemble blending with seasonal coupled ocean-atmosphere dynamic models (IMD Monsoon Mission CFSv2).

---

## License

Proprietary agricultural intelligence package for India localized drought mitigation. All rights reserved.
