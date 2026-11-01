"""
India Nationwide Drought Predictor ML Package.
Provides multi-horizon (7-day, 15-day, 30-day) drought classification using optimized XGBoost models.
"""

from .constants import (
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
    CLASS_ADVISORIES,
    ALL_53_BASE_FEATURES,
    HORIZONS,
    HORIZON_LABELS
)
from .schemas import (
    HorizonPrediction,
    DroughtPredictionResponse,
    BatchPredictionResponse
)
from .preprocessor import DroughtPreprocessor
from .inference import DroughtPredictor, predict_drought

__version__ = "2.0.0"
__all__ = [
    "predict_drought",
    "DroughtPredictor",
    "DroughtPreprocessor",
    "DroughtPredictionResponse",
    "HorizonPrediction",
    "BatchPredictionResponse",
    "CLASS_NAMES",
    "CLASS_DESCRIPTIONS",
    "CLASS_ADVISORIES",
    "ALL_53_BASE_FEATURES",
    "HORIZONS",
    "HORIZON_LABELS"
]
