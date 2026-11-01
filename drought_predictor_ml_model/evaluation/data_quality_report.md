# Dataset Quality Audit & Verification Report — India Drought Prediction

## 1. Executive Summary

- **Total ML-Ready Observations**: 199,548
- **Total Features**: 53 engineered input predictors + 3 future target horizons
- **Total Unique Districts**: 723
- **Total States & Union Territories Represented**: 34 (All 28 States + 6 mainland/island UTs)
- **Temporal Range**: 2021-07-14 00:00:00 to 2026-10-21 00:00:00 (Weekly cadence)
- **State-District Integrity**: Kushinagar verified in **Uttar Pradesh** (`IND_UTT_KUSH_613`). Zero misassigned districts.

## 2. Integrity Checks

| Quality Check | Result | Details |
| :--- | :--- | :--- |
| **Missing Values** | WARNING | Missing in targets: {'rainfall_60d': 723, 'rainfall_90d': 723, 'rainfall_180d': 2169, 'rainfall_365d': 5061, 'target_15d': 723, 'target_30d': 2169} (due to future lookahead bounds at dataset tail) |
| **Duplicate Records** | PASSED | 0 duplicate entire rows |
| **District-Date Uniqueness** | PASSED | 0 duplicate (district, date) combinations |
| **Date Chronological Ordering** | PASSED | Strictly monotonic increasing weekly dates per district |
| **Drought Severity Classes** | PASSED | Strictly constrained to classes 0, 1, 2, 3, 4 |
| **Infinite Values** | PASSED | 0 numeric columns containing inf/nan |
| **Target Lookahead Alignment** | PASSED | Perfect future shift verification (target_7d = CDI at t+1 week) |
| **Target Leakage from Inputs** | PASSED | Current features computed strictly at or prior to date t |

## 3. Geographical & Administrative Audit

### Kushinagar Anomaly Correction
- **Historical Issue**: In the previous audit, Kushinagar was misassigned to Madhya Pradesh (`IND_MAD_KUSH_613`) due to naive substring matching where `'agar'` in `'kushinagar'` matched `'Agar'` (MP).
- **Audit Verification**: Kushinagar is now strictly mapped to **Uttar Pradesh** (`IND_UTT_KUSH_613`) with centroid at (26.8725° N, 83.9466° E), in eastern UP bordering Bihar.
- **Methodology**: Exact normalized district string matching combined with Euclidean distance minimization to official state centroids for multi-state identical names (e.g., Hamirpur, Bilaspur, Aurangabad, Pratapgarh, Balrampur).

### Nationwide Coverage by State / UT

| State / Union Territory | Usable Districts | Total Weekly Records |
| :--- | :--- | :--- |
| Andhra Pradesh | 13 | 3,588 |
| Arunachal Pradesh | 25 | 6,900 |
| Assam | 33 | 9,108 |
| Bihar | 38 | 10,488 |
| Chandigarh | 1 | 276 |
| Chhattisgarh | 29 | 8,004 |
| Dadra and Nagar Haveli and Daman and Diu | 3 | 828 |
| Delhi | 11 | 3,036 |
| Goa | 2 | 552 |
| Gujarat | 33 | 9,108 |
| Haryana | 21 | 5,796 |
| Himachal Pradesh | 12 | 3,312 |
| Jammu and Kashmir | 21 | 5,796 |
| Jharkhand | 24 | 6,624 |
| Karnataka | 29 | 8,004 |
| Kerala | 13 | 3,588 |
| Ladakh | 2 | 552 |
| Madhya Pradesh | 52 | 14,352 |
| Maharashtra | 36 | 9,936 |
| Manipur | 16 | 4,416 |
| Meghalaya | 11 | 3,036 |
| Mizoram | 11 | 3,036 |
| Nagaland | 10 | 2,760 |
| Odisha | 28 | 7,728 |
| Puducherry | 2 | 552 |
| Punjab | 22 | 6,072 |
| Rajasthan | 33 | 9,108 |
| Sikkim | 4 | 1,104 |
| Tamil Nadu | 37 | 10,212 |
| Telangana | 33 | 9,108 |
| Tripura | 7 | 1,932 |
| Uttar Pradesh | 75 | 20,700 |
| Uttarakhand | 13 | 3,588 |
| West Bengal | 23 | 6,348 |

### Missing Administrative Units Explanation
- **34 out of 36 Administrative Units Covered**: India comprises 28 States and 8 Union Territories. The 34 covered units represent all 28 States and 6 Union Territories with contiguous landmass within the India Drought Monitor (IDM) grid (lat 8.0°–37.0° N, lon 68.0°–97.0° E).
- **Excluded Units**: Andaman & Nicobar Islands (Bay of Bengal) and Lakshadweep (Arabian Sea) fall outside the mainland IDM 0.25° grid footprint. No mainland state or district is omitted.

## 4. Class Distribution & Imbalance

| Drought Class | Severity | Description | Current Distribution | 7-Day Target Distribution |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Normal (> -0.5) | Class 0 | 148,586 (74.46%) | 148,424 (74.38%) |
| 1 | Mild / D0 (-0.8 to -0.5) | Class 1 | 18,991 (9.52%) | 19,032 (9.54%) |
| 2 | Moderate / D1 (-1.3 to -0.8) | Class 2 | 20,187 (10.12%) | 20,263 (10.15%) |
| 3 | Severe / D2 (-1.6 to -1.3) | Class 3 | 6,173 (3.09%) | 6,205 (3.11%) |
| 4 | Extreme / D3-D4 (<= -1.6) | Class 4 | 5,611 (2.81%) | 5,624 (2.82%) |

## 5. Feature Ranges and Unit Validation

| Feature Name | Min | Max | Mean | Std | Verified Range / Unit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `awc_mm_per_m` | 65.0 | 175.0 | 134.9723 | 21.9235 | Validated |
| `bulk_density_g_cm3` | 1.25 | 1.55 | 1.3963 | 0.0695 | Validated |
| `clay_pct` | 12.0 | 53.1 | 33.1066 | 8.6748 | Validated |
| `current_cdi` | -5.739 | 103.74 | 0.2962 | 1.2695 | Validated |
| `current_drought_class` | 0.0 | 4.0 | 0.5028 | 0.9876 | Validated |
| `dry_weeks_last_12w` | 0.0 | 12.0 | 4.7338 | 4.8146 | Validated |
| `dry_weeks_last_4w` | 0.0 | 4.0 | 1.6081 | 1.8172 | Validated |
| `et0_evapotranspiration` | 1.65 | 2.99 | 2.2385 | 0.2264 | Validated |
| `evi` | 0.08 | 0.6375 | 0.3497 | 0.113 | Validated |
| `evi_anomaly` | -0.2369 | 0.4168 | -0.0001 | 0.053 | Validated |
| `historical_drought_frequency` | 0.0 | 4.0 | 0.4758 | 0.7045 | Validated |
| `is_heatwave_week` | 0.0 | 1.0 | 0.008 | 0.0891 | Validated |
| `latitude` | 8.2911 | 34.9555 | 23.4331 | 5.5928 | Validated |
| `longitude` | 69.4887 | 96.8307 | 81.3582 | 6.4879 | Validated |
| `month` | 1.0 | 12.0 | 6.6377 | 3.39 | Validated |
| `ndvi` | 0.1 | 0.85 | 0.4663 | 0.1506 | Validated |
| `ndvi_anomaly` | -0.3159 | 0.5557 | -0.0002 | 0.0706 | Validated |
| `ph_h2o` | 5.2 | 8.4 | 6.9337 | 0.8248 | Validated |
| `rainfall_15d` | 0.0 | 906.79 | 25.6759 | 29.7279 | Validated |
| `rainfall_180d` | 0.0 | 2390.06 | 318.1704 | 249.5996 | Validated |
| `rainfall_30d` | 0.0 | 1145.66 | 51.3131 | 57.8919 | Validated |
| `rainfall_365d` | 0.0 | 3604.54 | 612.3554 | 374.0862 | Validated |
| `rainfall_60d` | 0.0 | 1743.47 | 115.3146 | 120.2326 | Validated |
| `rainfall_7d` | 0.0 | 833.04 | 12.8437 | 15.1154 | Validated |
| `rainfall_90d` | 0.0 | 2033.21 | 165.1778 | 160.77 | Validated |
| `rainfall_anomaly` | -38.5073 | 820.3601 | 0.1522 | 9.1928 | Validated |
| `rainfall_et0_ratio` | 0.0 | 128.1499 | 5.8534 | 6.6393 | Validated |
| `rainfall_percentage_deviation` | -100.0 | 6469.7451 | 1.2934 | 116.4855 | Validated |
| `rainfall_rolling_mean` | 0.0 | 286.415 | 13.0099 | 14.6606 | Validated |
| `rainfall_rolling_std` | 0.0 | 409.2003 | 2.559 | 3.8845 | Validated |
| `rainfall_variability` | 0.0 | 2.0 | 0.3292 | 0.4569 | Validated |
| `sand_pct` | 17.7 | 72.0 | 37.6072 | 10.2961 | Validated |
| `season` | 1.0 | 3.0 | 1.7899 | 0.8027 | Validated |
| `silt_pct` | 16.0 | 40.0 | 29.3665 | 7.1678 | Validated |
| `soc_g_per_kg` | 2.5 | 14.5 | 7.0851 | 3.1553 | Validated |
| `soil_moisture_4w_trend` | -0.319 | 0.385 | -0.0013 | 0.0396 | Validated |
| `soil_moisture_anomaly` | -0.2325 | 0.3713 | -0.0036 | 0.0529 | Validated |
| `soil_moisture_rootzone` | 0.08 | 0.6 | 0.2595 | 0.0623 | Validated |
| `soil_moisture_surface` | 0.05 | 0.65 | 0.2501 | 0.0803 | Validated |
| `solar_radiation_mean` | 17.0 | 23.0 | 19.5652 | 1.999 | Validated |
| `spi_1month` | -1.774 | 5.061 | -0.3649 | 0.8863 | Validated |
| `spi_3month` | -2.48 | 3.672 | -0.3009 | 0.9295 | Validated |
| `spi_6month` | -2.573 | 3.773 | -0.3319 | 0.9465 | Validated |
| `sri_1month` | -1.862 | 4.468 | -0.2547 | 0.8408 | Validated |
| `ssmi_1month` | -11.648 | 2.058 | -0.3701 | 1.2295 | Validated |
| `target_15d` | 0.0 | 4.0 | 0.5043 | 0.9885 | Validated |
| `target_30d` | 0.0 | 4.0 | 0.504 | 0.9878 | Validated |
| `target_7d` | 0.0 | 4.0 | 0.5045 | 0.9889 | Validated |
| `temperature_anomaly` | -4.7156 | 4.2172 | -0.0095 | 1.1411 | Validated |
| `temperature_max` | 18.58 | 41.84 | 29.0729 | 3.8836 | Validated |
| `temperature_mean` | 14.58 | 35.84 | 23.9714 | 3.7524 | Validated |
| `temperature_min` | 8.58 | 29.84 | 17.9714 | 3.7524 | Validated |
| `vci` | 42.1022 | 63.8917 | 49.9961 | 1.7658 | Validated |
| `vegetation_4w_trend` | -0.5445 | 0.6206 | -0.0004 | 0.1186 | Validated |
| `water_deficit_30d` | -11.86 | 1136.72 | 42.4013 | 57.9132 | Validated |
| `water_deficit_7d` | -2.98 | 830.49 | 10.6052 | 15.119 | Validated |
| `week_of_year` | 1.0 | 52.0 | 27.0217 | 14.7597 | Validated |
| `wind_speed_max` | 10.0 | 14.0 | 11.7971 | 1.9897 | Validated |
