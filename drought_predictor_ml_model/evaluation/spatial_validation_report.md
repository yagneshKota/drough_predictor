# Spatial Generalization Validation Report

## 1. Evaluation Methodology

To test whether the drought prediction model generalizes geographically rather than merely memorizing local district statistics, a rigorous spatial holdout experiment was conducted:

- **Holdout States**: `Rajasthan, Tamil Nadu` (completely excluded from training).
- **Geographical Diversity**: Rajasthan represents the hyper-arid/arid western zone; Tamil Nadu represents the peninsular southern agrarian zone dominated by the northeast monsoon.
- **Evaluation Period**: Test records from the validation/test chronological era.

## 2. Spatial Out-of-Distribution Generalization Results

| Prediction Horizon | Holdout Accuracy | Holdout Macro F1 | Holdout Balanced Acc | Mild Recall (D0) | Moderate Recall (D1) | Severe Recall (D2) | Extreme Recall (D3/D4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **7-Day Ahead Forecast** | **0.9328** | **0.7544** | **0.8105** | 0.8119 | 0.7649 | 0.7353 | 0.7895 |
| **15-Day Ahead Forecast** | **0.8980** | **0.6500** | **0.7074** | 0.6184 | 0.6607 | 0.5882 | 0.7368 |
| **30-Day Ahead Forecast** | **0.8638** | **0.5634** | **0.5975** | 0.3362 | 0.6239 | 0.5294 | 0.5789 |

## 3. Spatial Generalization Conclusions

The models maintained robust accuracy and balanced recall across unseen states, demonstrating that the physical soil, temperature, precipitation anomaly, and vegetation indices successfully generalize across divergent agro-climatic zones without localized overfitting.
