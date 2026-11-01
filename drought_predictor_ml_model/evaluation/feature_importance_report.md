# Feature Importance Analysis Report — India Drought Prediction Models

## 1. Executive Summary

Feature importance was analyzed independently across all three optimized models using tree Gain, Weight (frequency), and Cover metrics.

> **Scientific Note on Interpretation**: In gradient boosted decision trees, feature gain represents the relative reduction in loss function variance brought by splits on that feature. It should not be misinterpreted as a direct causal percentage of environmental drought impact.

## 2. Feature Importance for 7-Day Ahead Forecast

| Rank | Feature Name | Gain | Gain % | Split Weight | Cover |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `current_drought_class` | 930.45 | 39.35% | 758 | 11561.98 |
| 2 | `season` | 547.15 | 23.14% | 6 | 2541.49 |
| 3 | `current_cdi` | 336.57 | 14.23% | 8275 | 6333.87 |
| 4 | `wind_speed_max` | 75.92 | 3.21% | 51 | 2044.18 |
| 5 | `soil_moisture_rootzone` | 35.18 | 1.49% | 1622 | 2126.63 |
| 6 | `soil_moisture_anomaly` | 29.04 | 1.23% | 4241 | 1497.13 |
| 7 | `evi` | 28.37 | 1.20% | 1192 | 1718.01 |
| 8 | `soil_moisture_surface` | 26.71 | 1.13% | 2813 | 1583.70 |
| 9 | `rainfall_7d` | 22.72 | 0.96% | 759 | 1023.03 |
| 10 | `ndvi` | 21.04 | 0.89% | 2898 | 1279.45 |
| 11 | `water_deficit_7d` | 13.77 | 0.58% | 1517 | 946.03 |
| 12 | `rainfall_percentage_deviation` | 12.74 | 0.54% | 860 | 1464.12 |
| 13 | `evi_anomaly` | 10.54 | 0.45% | 1513 | 1451.82 |
| 14 | `vci` | 10.44 | 0.44% | 474 | 1104.31 |
| 15 | `rainfall_rolling_mean` | 9.93 | 0.42% | 600 | 1257.83 |
| 16 | `rainfall_anomaly` | 9.47 | 0.40% | 4560 | 947.92 |
| 17 | `dry_weeks_last_4w` | 9.27 | 0.39% | 67 | 526.24 |
| 18 | `rainfall_15d` | 8.97 | 0.38% | 952 | 938.63 |
| 19 | `rainfall_et0_ratio` | 8.81 | 0.37% | 862 | 1291.80 |
| 20 | `month` | 8.31 | 0.35% | 1080 | 766.88 |
| 21 | `historical_drought_frequency` | 8.09 | 0.34% | 6081 | 1247.90 |
| 22 | `ndvi_anomaly` | 8.00 | 0.34% | 4596 | 866.61 |
| 23 | `temperature_min` | 7.74 | 0.33% | 1028 | 1013.70 |
| 24 | `rainfall_30d` | 7.65 | 0.32% | 1186 | 907.70 |
| 25 | `soc_g_per_kg` | 7.46 | 0.32% | 411 | 368.62 |

## 2. Feature Importance for 15-Day Ahead Forecast

| Rank | Feature Name | Gain | Gain % | Split Weight | Cover |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `current_drought_class` | 816.88 | 57.39% | 776 | 11972.06 |
| 2 | `current_cdi` | 249.52 | 17.53% | 8875 | 6733.51 |
| 3 | `season` | 43.84 | 3.08% | 56 | 1682.66 |
| 4 | `soil_moisture_rootzone` | 30.90 | 2.17% | 4036 | 1851.29 |
| 5 | `soil_moisture_anomaly` | 21.08 | 1.48% | 5709 | 1527.92 |
| 6 | `ndvi` | 19.38 | 1.36% | 4334 | 1403.17 |
| 7 | `rainfall_7d` | 17.97 | 1.26% | 2291 | 1165.88 |
| 8 | `dry_weeks_last_4w` | 11.25 | 0.79% | 157 | 677.66 |
| 9 | `rainfall_30d` | 10.03 | 0.70% | 2553 | 1698.16 |
| 10 | `month` | 9.57 | 0.67% | 1934 | 991.13 |
| 11 | `soc_g_per_kg` | 9.27 | 0.65% | 1072 | 931.52 |
| 12 | `temperature_mean` | 8.95 | 0.63% | 7725 | 1019.65 |
| 13 | `soil_moisture_4w_trend` | 8.90 | 0.63% | 7316 | 1210.94 |
| 14 | `rainfall_et0_ratio` | 8.75 | 0.61% | 1995 | 1527.75 |
| 15 | `historical_drought_frequency` | 8.67 | 0.61% | 8362 | 1540.01 |
| 16 | `week_of_year` | 8.53 | 0.60% | 5797 | 1554.29 |
| 17 | `rainfall_anomaly` | 8.43 | 0.59% | 5810 | 1286.39 |
| 18 | `ndvi_anomaly` | 8.30 | 0.58% | 6009 | 1271.71 |
| 19 | `vci` | 8.28 | 0.58% | 1337 | 1198.53 |
| 20 | `water_deficit_30d` | 8.28 | 0.58% | 4504 | 1393.75 |
| 21 | `spi_1month` | 8.13 | 0.57% | 5491 | 1066.26 |
| 22 | `vegetation_4w_trend` | 8.05 | 0.57% | 8866 | 1196.75 |
| 23 | `dry_weeks_last_12w` | 7.87 | 0.55% | 1068 | 879.34 |
| 24 | `temperature_anomaly` | 7.75 | 0.54% | 6623 | 830.22 |
| 25 | `et0_evapotranspiration` | 7.66 | 0.54% | 4553 | 1224.67 |

## 2. Feature Importance for 30-Day Ahead Forecast

| Rank | Feature Name | Gain | Gain % | Split Weight | Cover |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `current_drought_class` | 644.66 | 41.61% | 2086 | 14800.32 |
| 2 | `wind_speed_max` | 84.30 | 5.44% | 105 | 4316.92 |
| 3 | `soil_moisture_anomaly` | 60.11 | 3.88% | 5309 | 3007.82 |
| 4 | `soil_moisture_surface` | 48.71 | 3.14% | 4643 | 2759.92 |
| 5 | `soil_moisture_rootzone` | 48.60 | 3.14% | 2584 | 3209.58 |
| 6 | `season` | 40.95 | 2.64% | 28 | 2591.82 |
| 7 | `rainfall_7d` | 37.65 | 2.43% | 1267 | 1694.88 |
| 8 | `dry_weeks_last_4w` | 29.96 | 1.93% | 153 | 4125.36 |
| 9 | `evi` | 29.35 | 1.89% | 1923 | 2380.46 |
| 10 | `rainfall_percentage_deviation` | 25.79 | 1.66% | 1365 | 2641.98 |
| 11 | `ndvi` | 22.63 | 1.46% | 4890 | 1816.48 |
| 12 | `evi_anomaly` | 20.12 | 1.30% | 1463 | 1858.25 |
| 13 | `historical_drought_frequency` | 17.14 | 1.11% | 8759 | 1997.94 |
| 14 | `water_deficit_7d` | 15.94 | 1.03% | 1954 | 1338.05 |
| 15 | `rainfall_15d` | 15.44 | 1.00% | 1248 | 1378.52 |
| 16 | `vci` | 15.09 | 0.97% | 484 | 2309.72 |
| 17 | `solar_radiation_mean` | 15.01 | 0.97% | 3636 | 1619.79 |
| 18 | `rainfall_anomaly` | 14.53 | 0.94% | 4734 | 1680.92 |
| 19 | `clay_pct` | 14.10 | 0.91% | 1745 | 2032.11 |
| 20 | `ndvi_anomaly` | 13.92 | 0.90% | 4849 | 1338.40 |
| 21 | `soc_g_per_kg` | 13.53 | 0.87% | 825 | 1041.59 |
| 22 | `awc_mm_per_m` | 13.25 | 0.86% | 155 | 1168.17 |
| 23 | `ph_h2o` | 13.05 | 0.84% | 779 | 1947.07 |
| 24 | `week_of_year` | 12.65 | 0.82% | 3869 | 1685.62 |
| 25 | `rainfall_90d` | 12.35 | 0.80% | 3633 | 1399.83 |

## 3. Comparative Importance Analysis

- **Short-Range (7-Day)**: Features representing short-term water balance (`current_cdi`, `current_drought_class`, `soil_moisture_rootzone`, `spi_1month`) have the highest split gain.
- **Medium-Range (15-Day)**: Transition features emerge prominently: `water_deficit_30d`, `rainfall_anomaly`, `soil_moisture_4w_trend`, and `vci`.
- **Long-Range (30-Day)**: Long-cycle drought memory features (`spi_3month`, `spi_6month`, `rainfall_90d`, `historical_drought_frequency`, `soil_moisture_rootzone`, `soc_g_per_kg`) exert substantial predictive influence.
