"""
Preprocessor module for India Drought Prediction ML Package.
Handles input feature validation, schema enforcement, imputation, standard scaling,
and horizon-specific feature slicing.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from .constants import ALL_53_BASE_FEATURES

class DroughtPreprocessor:
    """
    Validates, aligns, imputes, and scales input feature vectors for the
    7-day, 15-day, and 30-day XGBoost drought prediction models.
    """
    def __init__(self, models_dir: Union[str, Path] = None, config_dir: Union[str, Path] = None):
        pkg_root = Path(__file__).resolve().parent.parent
        self.models_dir = Path(models_dir) if models_dir else pkg_root / "models"
        self.config_dir = Path(config_dir) if config_dir else pkg_root / "config"
        
        self.pipeline_path = self.models_dir / "preprocessing_pipeline.joblib"
        self.config_path = self.config_dir / "feature_columns.json"
        
        self._load_pipeline_and_config()

    def _load_pipeline_and_config(self) -> None:
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Preprocessing pipeline not found at: {self.pipeline_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Feature config not found at: {self.config_path}")
            
        self.pipeline = joblib.load(self.pipeline_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.all_53_features = self.config.get("all_53_base_features", ALL_53_BASE_FEATURES)
        self.horizon_features = {
            h: self.config["model_horizons"][h]["feature_list"]
            for h in ["7day", "15day", "30day"]
        }

    def _to_dataframe(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize input into a DataFrame and extract metadata."""
        metadata = {"district": None, "state": None, "prediction_date": None}
        
        if isinstance(input_data, dict):
            metadata["district"] = input_data.get("district")
            metadata["state"] = input_data.get("state")
            metadata["prediction_date"] = str(input_data.get("date")) if input_data.get("date") else None
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"Expected dict, list of dicts, or pandas DataFrame, got {type(input_data)}")
            
        return df, metadata

    def prepare_features(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Validate, impute, scale, and extract horizon-specific arrays.
        
        Returns:
            horizon_arrays: Dict mapping horizon ('7day', '15day', '30day') to scaled numpy array
            metadata: Dict with district, state, prediction_date (if single record)
        """
        df_raw, metadata = self._to_dataframe(input_data)
        
        # Ensure all 53 expected features exist in DataFrame; fill missing with NaN for imputer
        missing_cols = [c for c in self.all_53_features if c not in df_raw.columns]
        for col in missing_cols:
            df_raw[col] = np.nan
            
        # Reorder columns strictly to match training preprocessor schema
        df_53 = df_raw[self.all_53_features].astype(float)
        
        # Transform via pipeline (SimpleImputer + StandardScaler fitted on training split)
        scaled_matrix = self.pipeline.transform(df_53)
        scaled_df = pd.DataFrame(scaled_matrix, columns=self.all_53_features, index=df_53.index)
        
        # Slice exact features required by each horizon
        horizon_arrays = {
            h: scaled_df[self.horizon_features[h]].values
            for h in ["7day", "15day", "30day"]
        }
        
        return horizon_arrays, metadata
