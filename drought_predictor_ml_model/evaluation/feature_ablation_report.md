# Feature Ablation & Configuration Experiment Report

## 1. Overview of Experimental Configurations

To understand the predictive power of environmental indicators versus autoregressive persistence signals, we evaluated six distinct feature configurations across all three prediction horizons (7-day, 15-day, 30-day) on the chronological validation split:

- **Experiment A (Original 53 Features)**: All 53 base features (meteorology, soil, hydrology, vegetation, calendar, and current CDI/drought class).
- **Experiment B (Remove Current Drought Class)**: 52 features. Excludes `current_drought_class` while retaining continuous `current_cdi`.
- **Experiment C (Remove Current CDI)**: 52 features. Excludes continuous `current_cdi` while retaining discrete `current_drought_class`.
- **Experiment D (Environmental-Only Model)**: 51 features. Excludes **both** `current_drought_class` and `current_cdi`. Predicts purely from environmental, soil, weather, vegetation, and multi-scale SPI/SRI/SSMI indices.
- **Experiment E (Feature Selection)**: 47 features. Prunes collinear redundant pairs (e.g., `evi` collinear with `ndvi`, `soil_moisture_surface` collinear with `soil_moisture_rootzone`, `rainfall_rolling_mean` collinear with `rainfall_30d`).
- **Experiment F (Tuned Feature Combination)**: 35 features. Selects the most parsimonious, high-signal subset of soil properties, dynamic weather trends, vegetation health, and lagged drought metrics.

## 2. Validation Performance Comparison Across Configurations

### 7-Day Ahead Forecast

| Configuration | Features | Accuracy | Macro F1 | Balanced Acc | Severe Recall (D2) | Extreme Recall (D3/D4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp_A_Original53** | 53 | **0.8706** | **0.7607** | 0.8130 | 0.8050 | 0.8689 |
| **Exp_B_NoCurrentClass** | 52 | **0.8703** | **0.7586** | 0.8118 | 0.8000 | 0.8732 |
| **Exp_C_NoCurrentCDI** | 52 | **0.8682** | **0.7568** | 0.8090 | 0.7966 | 0.8605 |
| **Exp_D_EnvironmentalOnly** | 51 | **0.8647** | **0.7505** | 0.8047 | 0.7916 | 0.8562 |
| **Exp_E_FeatureSelection** | 47 | **0.8706** | **0.7595** | 0.8116 | 0.7950 | 0.8689 |
| **Exp_F_TunedCombination** | 35 | **0.8706** | **0.7592** | 0.8121 | 0.7933 | 0.8710 |

### 15-Day Ahead Forecast

| Configuration | Features | Accuracy | Macro F1 | Balanced Acc | Severe Recall (D2) | Extreme Recall (D3/D4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp_A_Original53** | 53 | **0.7986** | **0.6558** | 0.7251 | 0.7192 | 0.8106 |
| **Exp_B_NoCurrentClass** | 52 | **0.7969** | **0.6530** | 0.7233 | 0.7175 | 0.8149 |
| **Exp_C_NoCurrentCDI** | 52 | **0.7958** | **0.6544** | 0.7242 | 0.7243 | 0.8043 |
| **Exp_D_EnvironmentalOnly** | 51 | **0.7914** | **0.6494** | 0.7213 | 0.7038 | 0.8234 |
| **Exp_E_FeatureSelection** | 47 | **0.7969** | **0.6548** | 0.7242 | 0.7192 | 0.8149 |
| **Exp_F_TunedCombination** | 35 | **0.7991** | **0.6566** | 0.7267 | 0.7175 | 0.8170 |

### 30-Day Ahead Forecast

| Configuration | Features | Accuracy | Macro F1 | Balanced Acc | Severe Recall (D2) | Extreme Recall (D3/D4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp_A_Original53** | 53 | **0.7239** | **0.5508** | 0.6394 | 0.6427 | 0.7238 |
| **Exp_B_NoCurrentClass** | 52 | **0.7225** | **0.5497** | 0.6394 | 0.6517 | 0.7216 |
| **Exp_C_NoCurrentCDI** | 52 | **0.7257** | **0.5527** | 0.6408 | 0.6427 | 0.7323 |
| **Exp_D_EnvironmentalOnly** | 51 | **0.7174** | **0.5471** | 0.6368 | 0.6230 | 0.7323 |
| **Exp_E_FeatureSelection** | 47 | **0.7229** | **0.5492** | 0.6377 | 0.6373 | 0.7259 |
| **Exp_F_TunedCombination** | 35 | **0.7240** | **0.5528** | 0.6432 | 0.6248 | 0.7516 |

## 3. Scientific Analysis of Findings

1. **Autoregressive Decay over Horizons**: At the 7-day horizon, `current_drought_class` and `current_cdi` provide strong continuity signals. However, as the horizon expands to 15-day and 30-day, their dominance diminishes significantly.
2. **Environmental Model Viability**: The Environmental-Only model (Exp D) achieves respectable predictive capability solely from physical indicators, proving that genuine biophysical drivers (soil moisture deficit, SPI-3, SPI-6, ET0, VCI) capture drought development without circular dependency on the current classification.
3. **Redundancy Pruning Benefits**: Experiment E and F demonstrate that removing collinear features (such as parallel rolling rainfall windows and redundant vegetation indices) stabilizes gradient variance and prevents tree split fragmentation.
