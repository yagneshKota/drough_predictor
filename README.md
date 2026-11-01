# India Nationwide Drought Prediction System

An end-to-end intelligent agricultural drought monitoring and multi-horizon early-warning prediction platform for 716 districts across India.

## Repository Overview

This monorepo houses the complete drought prediction solution developed for nationwide agricultural mitigation, incorporating machine learning prediction models, backend microservices, and user-facing dashboards.

```text
drough_predictor/
│
├── drought_predictor_ml_model/     # Machine Learning inference package & trained models
│   ├── models/                     # Trained XGBoost models (7d, 15d, 30d) & scalers
│   ├── src/                        # Prediction engine & preprocessing modules
│   ├── config/                     # Feature column definitions & metadata
│   ├── evaluation/                 # Metrics, ablation reports & confusion matrices
│   ├── feature_information/        # SHAP explainability & feature rankings
│   ├── examples/                   # FastAPI backend example & sample inputs
│   ├── scripts/                    # Inference & verification test scripts
│   ├── requirements.txt            # Minimal ML inference dependencies
│   └── README.md                   # Comprehensive ML package documentation
│
└── README.md                       # Project root overview
```

---

## Machine Learning Component (`drought_predictor_ml_model/`)

The ML package delivers multi-horizon drought classification:
- **7-Day Forecast (1-Week Ahead)**: 88.31% Accuracy | 0.7782 Macro F1
- **15-Day Forecast (2-Weeks Ahead)**: 81.76% Accuracy | 0.6710 Macro F1
- **30-Day Forecast (4-Weeks Ahead)**: 74.17% Accuracy | 0.5577 Macro F1

Five standardized severity classes are monitored:
0. **Normal**
1. **Mild Drought (D0)**
2. **Moderate Drought (D1)**
3. **Severe Drought (D2)**
4. **Extreme Drought (D3/D4)**

### Quick Usage

```python
from drought_predictor_ml_model.src.inference import predict_drought

# Run inference on observation dictionary
result = predict_drought(observation_features)
print(result["predictions"]["7day"]["predicted_label"])
```

For full documentation, architecture diagrams, API schemas, and FastAPI integration guides, see [`drought_predictor_ml_model/README.md`](./drought_predictor_ml_model/README.md).
