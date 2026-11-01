"""
FastAPI Backend Integration Example for India Drought Prediction.
Provides RESTful endpoints for single and batch predictions with strict feature validation.

To run:
    uvicorn examples.fastapi_app_example:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add package root to sys.path
pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import DroughtPredictor, predict_drought
from src.schemas import DroughtPredictionResponse, BatchPredictionResponse

app = FastAPI(
    title="India Nationwide Drought Prediction ML API",
    description="Production API delivering 7-day, 15-day, and 30-day drought severity forecasts across 716 Indian districts.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor on startup (singleton)
predictor = DroughtPredictor.get_instance()

class PredictionInputRequest(BaseModel):
    district: Optional[str] = Field("Kushinagar", description="District name")
    state: Optional[str] = Field("Uttar Pradesh", description="State name")
    date: Optional[str] = Field("2025-01-01", description="Observation date (YYYY-MM-DD)")
    allow_imputation: bool = Field(
        default=False,
        description="Set to True to allow training-median imputation for missing features. If False, strict validation requires all 53 features."
    )
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of input environmental and meteorological features (e.g. rainfall_7d, soil_moisture_rootzone, ndvi, current_cdi...)"
    )

class BatchInputRequest(BaseModel):
    allow_imputation: bool = Field(
        default=False,
        description="Set to True to allow training-median imputation for missing features."
    )
    records: List[Dict[str, Any]] = Field(..., description="List of district observations")

@app.get("/health", tags=["System"])
def health_check():
    """Health check probe endpoint for Kubernetes / Docker container monitoring."""
    return {
        "status": "healthy",
        "model_version": "2.0-optimized-nationwide",
        "horizons": ["7day", "15day", "30day"],
        "classes": [
            "Normal",
            "Mild Drought (D0)",
            "Moderate Drought (D1)",
            "Severe Drought (D2)",
            "Extreme Drought (D3/D4)"
        ]
    }

@app.post("/predict", response_model=DroughtPredictionResponse, tags=["Inference"])
def predict_single_observation(request: PredictionInputRequest):
    """
    Generate 7-day, 15-day, and 30-day drought predictions for a single district record.
    """
    try:
        payload = dict(request.features)
        if request.district:
            payload["district"] = request.district
        if request.state:
            payload["state"] = request.state
        if request.date:
            payload["date"] = request.date
            
        response = predictor.predict_single(
            payload,
            allow_imputation=request.allow_imputation,
            as_dict=False
        )
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(ve)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_observations(request: BatchInputRequest):
    """
    Generate multi-horizon drought predictions for a batch of district records.
    """
    try:
        response = predictor.predict_batch(
            request.records,
            allow_imputation=request.allow_imputation,
            as_dict=False
        )
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(ve)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch inference failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app_example:app", host="0.0.0.0", port=8000, reload=True)
