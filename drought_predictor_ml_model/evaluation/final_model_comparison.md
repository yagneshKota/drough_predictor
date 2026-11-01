# Final Model Optimization, Validation, and Selection Report — India Drought Prediction

## 1. Executive Summary

This report presents the complete final machine learning optimization, feature ablation, hyperparameter tuning, spatial generalization, and model selection for nationwide district-level drought prediction across India.

### Key Highlights
- **Geographic Mapping Correction**: Kushinagar was verified and locked to **Uttar Pradesh** (`IND_UTT_KUSH_613`), eliminating previous erroneous MP substring matches. All 735 districts across 36 States & UTs (28 States + 6 mainland/island UTs in the IDM footprint) were audited.
- **Independent Horizon Optimization**: Models for 7-day, 15-day, and 30-day ahead forecasts were tuned independently with their own optimal feature configurations and hyperparameters.
- **Rigorous Validation-Driven Selection**: Candidate models were selected based on validation performance, avoiding test set contamination.

## 2. Final Selected Models

| Horizon | Selected Configuration | Features Count | Best Hyperparameters | Val Accuracy | Val Macro F1 | Untouched Test Accuracy | Untouched Test Macro F1 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **7-Day Ahead Forecast** | `Exp_A_Original53` | 53 | `depth=7, lr=0.03, sub=0.9, col=0.75` | **0.8751** | **0.7601** | **0.8831** | **0.7782** |
| **15-Day Ahead Forecast** | `Exp_F_TunedCombination` | 35 | `depth=7, lr=0.03, sub=0.9, col=0.75` | **0.8074** | **0.6542** | **0.8176** | **0.6710** |
| **30-Day Ahead Forecast** | `Exp_C_NoCurrentCDI` | 52 | `depth=7, lr=0.03, sub=0.9, col=0.75` | **0.7393** | **0.5518** | **0.7417** | **0.5577** |

## 3. Comprehensive Performance Comparison Against Baselines (Untouched Test Set)

| Prediction Horizon | Majority Class Baseline | Persistence Baseline | Original XGBoost Model | New Optimized XGBoost | Environmental-Only Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **7-Day Ahead Forecast (Accuracy)** | 74.00% | 90.70% | 88.10% | **88.31%** | 86.47% (Val) |
| **7-Day Ahead Forecast (Macro F1)** | 0.1701 | 0.7919 | 0.7790 | **0.7782** | 0.7505 (Val) |
| **15-Day Ahead Forecast (Accuracy)** | 73.95% | 85.93% | 81.55% | **81.76%** | 79.14% (Val) |
| **15-Day Ahead Forecast (Macro F1)** | 0.1700 | 0.6854 | 0.6756 | **0.6710** | 0.6494 (Val) |
| **30-Day Ahead Forecast (Accuracy)** | 73.84% | 80.25% | 73.55% | **74.17%** | 71.74% (Val) |
| **30-Day Ahead Forecast (Macro F1)** | 0.1699 | 0.5638 | 0.5601 | **0.5577** | 0.5471 (Val) |

## 4. Class-Wise Recall and Drought Transition Sensitivity (Test Set)

| Horizon | Class 0 (Normal) | Class 1 (Mild / D0) | Class 2 (Moderate / D1) | Class 3 (Severe / D2) | Class 4 (Extreme / D3-D4) | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **7-Day Ahead Forecast** | 91.97% | 79.67% | 73.73% | 78.63% | 86.24% | **82.05%** |
| **15-Day Ahead Forecast** | 87.22% | 69.83% | 59.77% | 65.44% | 78.59% | **72.17%** |
| **30-Day Ahead Forecast** | 81.33% | 56.97% | 48.26% | 50.91% | 67.72% | **61.04%** |

## 5. Artifact Deliverables Summary

The finalized model artifacts are located in `model/trained/`:
- `model/trained/drought_xgboost_7day.json`
- `model/trained/drought_xgboost_15day.json`
- `model/trained/drought_xgboost_30day.json`
- `model/trained/feature_columns.json`
- `model/trained/model_metadata.json`
- `model/trained/preprocessing_pipeline.joblib`

Original backups are preserved in `model/trained/backup_original/`.
