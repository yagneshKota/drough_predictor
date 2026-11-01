"""
Pydantic Schemas for Drought Prediction Request & Response.
Provides typed, validated data containers for FastAPI and standalone inference.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class HorizonPrediction(BaseModel):
    """Prediction outputs for a single forecast horizon."""
    horizon: str = Field(..., description="Forecast horizon identifier (e.g. '7day', '15day', '30day')")
    horizon_label: str = Field(..., description="Human-readable forecast horizon label")
    predicted_class: int = Field(..., ge=0, le=4, description="Predicted drought severity class ID (0 to 4)")
    predicted_label: str = Field(..., description="Readable drought severity class name")
    description: str = Field(..., description="Official environmental condition description")
    advisory: str = Field(..., description="Recommended agricultural/water action advisory")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score (probability of predicted class)")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all 5 severity classes"
    )

class DroughtPredictionResponse(BaseModel):
    """Structured response container for all prediction horizons."""
    status: str = Field(default="success", description="Prediction status ('success' or 'error')")
    district: Optional[str] = Field(None, description="District name if provided")
    state: Optional[str] = Field(None, description="State name if provided")
    prediction_date: Optional[str] = Field(None, description="Observation or prediction date (YYYY-MM-DD)")
    model_version: str = Field(default="2.0-optimized-nationwide", description="Version of the deployed ML model")
    predictions: Dict[str, HorizonPrediction] = Field(
        ...,
        description="Predictions keyed by horizon ('7day', '15day', '30day')"
    )

class BatchPredictionResponse(BaseModel):
    """Container for batch inference requests."""
    status: str = Field(default="success")
    total_records: int = Field(..., ge=0, description="Number of observations processed")
    results: List[DroughtPredictionResponse] = Field(..., description="List of prediction responses")
