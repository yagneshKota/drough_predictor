"""
Inference Module for India Drought Prediction ML Package.
Provides high-level, production-ready inference methods for 7-day, 15-day,
and 30-day drought severity forecasts with probability distributions.
"""

from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from .constants import (
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
    CLASS_ADVISORIES,
    HORIZONS,
    HORIZON_LABELS
)
from .schemas import (
    HorizonPrediction,
    DroughtPredictionResponse,
    BatchPredictionResponse
)
from .preprocessor import DroughtPreprocessor

class DroughtPredictor:
    """
    Main prediction engine for nationwide district drought forecasting.
    Loads and caches the 7-day, 15-day, and 30-day XGBoost models in memory.
    """
    _instance: Optional['DroughtPredictor'] = None

    def __init__(self, models_dir: Union[str, Path] = None, config_dir: Union[str, Path] = None):
        pkg_root = Path(__file__).resolve().parent.parent
        self.models_dir = Path(models_dir) if models_dir else pkg_root / "models"
        self.config_dir = Path(config_dir) if config_dir else pkg_root / "config"
        
        self.preprocessor = DroughtPreprocessor(
            models_dir=self.models_dir,
            config_dir=self.config_dir
        )
        self.models: Dict[str, xgb.XGBClassifier] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load trained XGBoost models for 7day, 15day, and 30day horizons."""
        for h in HORIZONS:
            model_path = self.models_dir / f"drought_xgboost_{h}.json"
            if not model_path.exists():
                raise FileNotFoundError(f"Model artifact not found for horizon '{h}': {model_path}")
            
            clf = xgb.XGBClassifier()
            clf.load_model(str(model_path))
            self.models[h] = clf

    @classmethod
    def get_instance(cls) -> 'DroughtPredictor':
        """Singleton accessor to prevent reloading heavy models on every web request."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict_single(
        self,
        input_features: Dict[str, Any],
        allow_imputation: bool = True,
        as_dict: bool = True
    ) -> Union[Dict[str, Any], DroughtPredictionResponse]:
        """
        Generate predictions for a single district-week observation.
        
        Args:
            input_features: Dictionary of input environmental and meteorological features.
            allow_imputation: If False, raises ValueError when any of the 53 features are missing.
                              If True, applies training-set median imputation and reports imputed features.
            as_dict: If True, returns a standard Python dictionary; else a Pydantic object.
        """
        horizon_arrays, metadata, imputed_features = self.preprocessor.prepare_features(
            input_features, allow_imputation=allow_imputation
        )
        
        warnings = []
        if imputed_features:
            warnings.append(
                f"Imputation applied: {len(imputed_features)} features were missing and imputed using training-set medians: "
                f"{imputed_features[:5]}{'...' if len(imputed_features) > 5 else ''}"
            )
            
        predictions = {}
        for h in HORIZONS:
            X_h = horizon_arrays[h]
            clf = self.models[h]
            
            # Predict class and probabilities
            pred_class = int(clf.predict(X_h)[0])
            pred_probs = clf.predict_proba(X_h)[0]
            
            prob_dict = {
                CLASS_NAMES[c]: round(float(pred_probs[c]), 4)
                for c in range(5)
            }
            confidence = round(float(pred_probs[pred_class]), 4)
            
            hp = HorizonPrediction(
                horizon=h,
                horizon_label=HORIZON_LABELS[h],
                predicted_class=pred_class,
                predicted_label=CLASS_NAMES[pred_class],
                description=CLASS_DESCRIPTIONS[pred_class],
                advisory=CLASS_ADVISORIES[pred_class],
                confidence=confidence,
                probabilities=prob_dict
            )
            predictions[h] = hp
            
        response = DroughtPredictionResponse(
            status="success",
            district=metadata.get("district"),
            state=metadata.get("state"),
            prediction_date=metadata.get("prediction_date"),
            model_version="2.0-optimized-nationwide",
            imputation_applied=bool(imputed_features),
            imputed_features=imputed_features,
            warnings=warnings,
            predictions=predictions
        )
        
        return response.model_dump() if as_dict else response

    def predict_batch(
        self,
        input_data: Union[List[Dict[str, Any]], pd.DataFrame],
        allow_imputation: bool = True,
        as_dict: bool = True
    ) -> Union[Dict[str, Any], BatchPredictionResponse]:
        """
        Generate predictions for multiple district-week observations in vectorized batch.
        """
        horizon_arrays, _, imputed_features = self.preprocessor.prepare_features(
            input_data, allow_imputation=allow_imputation
        )
        
        if isinstance(input_data, pd.DataFrame):
            records = input_data.to_dict(orient="records")
        else:
            records = input_data
            
        n_samples = len(records)
        all_preds = {h: self.models[h].predict(horizon_arrays[h]) for h in HORIZONS}
        all_probs = {h: self.models[h].predict_proba(horizon_arrays[h]) for h in HORIZONS}
        
        warnings = []
        if imputed_features:
            warnings.append(
                f"Batch imputation applied: {len(imputed_features)} features were imputed using training medians."
            )
            
        results = []
        for i in range(n_samples):
            rec = records[i]
            d_name = rec.get("district")
            s_name = rec.get("state")
            p_date = str(rec.get("date")) if rec.get("date") else None
            
            preds_i = {}
            for h in HORIZONS:
                c_idx = int(all_preds[h][i])
                probs = all_probs[h][i]
                prob_dict = {
                    CLASS_NAMES[c]: round(float(probs[c]), 4)
                    for c in range(5)
                }
                preds_i[h] = HorizonPrediction(
                    horizon=h,
                    horizon_label=HORIZON_LABELS[h],
                    predicted_class=c_idx,
                    predicted_label=CLASS_NAMES[c_idx],
                    description=CLASS_DESCRIPTIONS[c_idx],
                    advisory=CLASS_ADVISORIES[c_idx],
                    confidence=round(float(probs[c_idx]), 4),
                    probabilities=prob_dict
                )
            results.append(DroughtPredictionResponse(
                status="success",
                district=d_name,
                state=s_name,
                prediction_date=p_date,
                model_version="2.0-optimized-nationwide",
                imputation_applied=bool(imputed_features),
                imputed_features=imputed_features,
                warnings=warnings,
                predictions=preds_i
            ))
            
        batch_resp = BatchPredictionResponse(
            status="success",
            total_records=n_samples,
            results=results
        )
        return batch_resp.model_dump() if as_dict else batch_resp

# Top-level functional API
def predict_drought(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    allow_imputation: bool = True,
    as_dict: bool = True
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Standard high-level prediction function for backend developers.
    
    Args:
        input_data: Dict (single observation), List[Dict] or DataFrame (batch).
        allow_imputation: If False, enforces strict validation requiring all 53 features.
                          If True, applies training-set median imputation and flags imputed fields.
        as_dict: If True, returns dict; else returns Pydantic response models.
    
    Usage:
        from src.inference import predict_drought
        result = predict_drought(input_features)
    """
    predictor = DroughtPredictor.get_instance()
    if isinstance(input_data, (list, pd.DataFrame)):
        return predictor.predict_batch(input_data, allow_imputation=allow_imputation, as_dict=as_dict)
    elif isinstance(input_data, dict):
        return predictor.predict_single(input_data, allow_imputation=allow_imputation, as_dict=as_dict)
    else:
        raise TypeError(f"Invalid input type: {type(input_data)}. Expected dict, list of dicts, or DataFrame.")
