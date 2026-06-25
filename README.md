# Urban Air Pollution Prediction (PM2.5) — Zindi Challenge

A one-week team project (2 people) predicting daily PM2.5 concentration
from weather and satellite data, as part of the [Zindi Urban Air Pollution Challenge](https://zindi.africa/competitions/zindiweekendz-learning-urban-air-pollution-challenge).

## Problem Statement

The objective is to predict daily PM2.5 concentration for each location, using:
- **Ground-based sensors** — target variable (train only)
- **GFS weather data** — humidity, temperature, wind
- **Sentinel-5P satellite data** — 8 atmospheric pollutant products (NO2, O3, CO, HCHO, CLOUD, AER_AI, SO2, CH4)

**Key structural finding:** Train (340 locations) and Test (179 locations) are **completely
disjoint** in `Place_ID` (verified via set intersection, overlap = 0), while covering the
**identical date range** (2020-01-02 to 2020-04-04). This means the task is pure **spatial
extrapolation to unseen locations** — not a time-series forecasting problem. This single fact
drove every downstream modeling decision in this project.

## Methodology

### 1. Validation Strategy

A naive row-wise `train_test_split` would leave rows from the same location in both train and
validation, letting the model implicitly memorize location-specific baselines — producing an
overly optimistic CV score that would not transfer to the leaderboard (where test locations are
never seen at all).

**Solution:** `GroupShuffleSplit` / `GroupKFold`, grouped by `Place_ID`, ensures each location is
entirely in train *or* validation, never both. This split was exported once and shared across
both subgroups to keep CV scores comparable.

### 2. Exploratory Data Analysis & Feature Engineering

**Satellite data (Sentinel-5P):**
- Missingness analysis revealed that NaNs are largely **pass-based** — within most pollutant
  product groups, missingness across columns is correlated near 1.0, meaning a single satellite
  pass either succeeds or fails for the whole product.
- Two exceptions, both explained by the underlying retrieval algorithms:
  - **NO2 tropospheric column** has its own independent failure mode, since separating the
    tropospheric from the stratospheric column requires a downstream chemical transport model
    (TM5-MP) assimilation step, separate from the initial DOAS retrieval.
  - **CLOUD product** splits into two correlation clusters because it bundles two independent
    algorithms: OCRA (cloud fraction, UV/VIS-based) and ROCINN (cloud height/optical depth,
    O2 A-band-based).
- Missing-indicator flags added for high/moderate-missing pollutants (CH4 ~81%, NO2/CO/HCHO/SO2
  ~18–28%); low-missing groups (CLOUD, O3, AER_AI, ~1–4%) handled with simple imputation only.
- **Physically impossible zero-values** in CH4 columns identified as satellite fill-values
  (atmospheric CH4 is never truly absent) and converted to NaN before flagging.
- Redundant columns (cross-correlation > 0.9) removed within product groups (e.g. CLOUD's
  height/pressure quartet, NO2/HCHO slant-vs-tropospheric pairs), using target correlation as a
  tie-breaker only when the gap was large enough to be meaningful, and physical reasoning
  otherwise.
- Sensor viewing-geometry metadata (`*_angle` columns) excluded — not atmospheric measurements.

**Weather data (GFS):**
- Wind components (`u`, `v`) decomposed into `wind_speed` and circularly-encoded `wind_direction`
  (`sin`/`cos`) to avoid the false discontinuity at the 0°/360° wrap-around.
- Highly correlated `specific_humidity` dropped in favor of `precipitable_water_entire_atmosphere`.

**Notable finding — spurious correlation caught and removed:**
`sensor_altitude` columns initially ranked among the top features by importance (moderate target
correlation, ~-0.31). Hypothesized initially as a possible terrain-elevation effect (plausible in
principle — elevation affects pollutant accumulation via inversion layers). Verification showed
the values (~824–844 km) match Sentinel-5P's known orbital altitude, not terrain elevation
(max ~8.8 km globally) — confirming this is satellite viewing-geometry metadata, not a ground
property. Most likely an indirect proxy for geographic position rather than a causal driver of
PM2.5. **Removed** to avoid building a non-generalizable shortcut into the model.

### 3. Modeling

| Model | Setup | RMSE |
|---|---|---|
| Dummy (mean) | No features | 42.0 |
| XGBoost Regressor (untuned) | Single split | 31.8 |
| XGBoost Regressor (tuned, `RandomizedSearchCV`) | GroupKFold(5), 80% train | 33.4 (CV mean) |
| XGBoost Regressor (tuned, incl. Sensor_altitude) | Held-out 20% validation | 30.09 |
| XGBoost Regressor (tuned, excl. Sensor_altitude) | Held-out 20% validation | **30.78** |
| Random Forest Regressor | GroupKFold(5), comparison model | *(see notebook)* |

*Note: RMSE values shift slightly between iterations as the feature set and validation strategy
were refined (e.g. spurious feature removal, switch from single split to GroupKFold) — later,
more conservative numbers reflect a more trustworthy estimate, not a regression in performance.*

## Key Learnings

- **Spatial vs. temporal generalization is not always obvious from the file structure** — it has
  to be explicitly verified (disjoint location check, date range check) before choosing a CV
  strategy.
- **High feature importance does not imply a causal or generalizable relationship** — metadata
  fields can act as indirect proxies for things the model was never meant to learn from directly.
- **Missingness patterns can be diagnostic, not just a nuisance** — correlated missingness
  revealed the underlying satellite retrieval pipeline structure.
- **Domain reasoning and statistical evidence should both inform feature decisions** — when
  correlation differences are within noise, defer to physical/mechanistic reasoning.

## Repository Structure

```
├── 01_eda.ipynb              # Data exploration, missingness analysis, feature engineering
├── 02_model_xgboost.ipynb    # XGBoost pipeline, hyperparameter tuning, feature importance
├── 03_model_randomforest.ipynb  # Random Forest comparison model
├── data/                     # Not tracked — see data/README.md for download instructions
└── README.md
```

## Team

4-person team project, split into 2 subgroups working in parallel on weather and satellite data
streams, with shared validation split and daily knowledge-sharing standups.