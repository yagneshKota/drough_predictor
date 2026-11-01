"""
Constants and Feature Definitions for India Drought Prediction ML Package.
"""

from typing import Dict, List

# Official 5-class drought severity mappings according to the India Drought Monitor (IDM)
CLASS_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Mild Drought (D0)",
    2: "Moderate Drought (D1)",
    3: "Severe Drought (D2)",
    4: "Extreme Drought (D3/D4)"
}

CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: "Near-normal or wetter moisture conditions. No significant soil or hydrological stress.",
    1: "Abnormally dry; incipient soil moisture deficit. Subsurface recharge slowed.",
    2: "Moderate drought; agricultural moisture deficits, delayed crop development, reduced streamflow.",
    3: "Severe drought; widespread crop damage, low reservoir levels, mandatory water conservation advisories.",
    4: "Extreme to exceptional drought; major water emergency, widespread crop failure, acute ecological crisis."
}

CLASS_ADVISORIES: Dict[int, str] = {
    0: "Standard irrigation scheduling. Optimal agricultural operations.",
    1: "Monitor rainfall trends. Conserve topsoil moisture through mulching and micro-irrigation.",
    2: "Implement rotational irrigation. Prioritize drought-resistant crop varieties and soil moisture conservation.",
    3: "Activate drought contingency protocols. Restrict non-essential water usage. Provide livestock relief.",
    4: "Declare emergency drought measures. Mobilize inter-basin water transfers and emergency water supplies."
}

# The complete master sequence of 53 engineered features expected by the preprocessing pipeline
ALL_53_BASE_FEATURES: List[str] = [
    "rainfall_7d",
    "rainfall_15d",
    "rainfall_30d",
    "rainfall_60d",
    "rainfall_90d",
    "rainfall_180d",
    "rainfall_365d",
    "rainfall_rolling_mean",
    "rainfall_rolling_std",
    "rainfall_variability",
    "rainfall_anomaly",
    "rainfall_percentage_deviation",
    "dry_weeks_last_4w",
    "dry_weeks_last_12w",
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "temperature_anomaly",
    "is_heatwave_week",
    "soil_moisture_surface",
    "soil_moisture_rootzone",
    "soil_moisture_anomaly",
    "soil_moisture_4w_trend",
    "et0_evapotranspiration",
    "water_deficit_7d",
    "water_deficit_30d",
    "rainfall_et0_ratio",
    "solar_radiation_mean",
    "wind_speed_max",
    "ndvi",
    "evi",
    "ndvi_anomaly",
    "evi_anomaly",
    "vci",
    "vegetation_4w_trend",
    "clay_pct",
    "sand_pct",
    "silt_pct",
    "soc_g_per_kg",
    "ph_h2o",
    "awc_mm_per_m",
    "bulk_density_g_cm3",
    "current_cdi",
    "current_drought_class",
    "spi_1month",
    "spi_3month",
    "spi_6month",
    "sri_1month",
    "ssmi_1month",
    "historical_drought_frequency",
    "month",
    "week_of_year",
    "season"
]

# Forecast horizons
HORIZONS: List[str] = ["7day", "15day", "30day"]

HORIZON_LABELS: Dict[str, str] = {
    "7day": "7-Day Ahead Forecast (1-Week)",
    "15day": "15-Day Ahead Forecast (2-Weeks)",
    "30day": "30-Day Ahead Forecast (4-Weeks)"
}
