# Project 9 - Favorita Store Sales Forecasting

**Track:** Data Scientist / MLE - Multivariate Time-Series
**Difficulty:** 7/10
**Status:** Phase 1 - scaffolded code-only

## Goal

Forecast daily unit sales for ~1,800 store-family series across 54 Corporacion Favorita supermarkets in Ecuador. The series cover 33 product families (BEVERAGES, BREAD/BAKERY, DAIRY, GROCERY I, MEATS, PRODUCE, etc.) over 4+ years (2013-01-01 to 2017-08-31). The horizon for the Kaggle test set is 16 days (2017-08-16 to 2017-08-31). The objective is a calibrated point and quantile forecast that respects holiday spikes, oil-price macroeconomic regime, and store-level promotional state.

## Why this matters (DACH supply chain context)

Demand-planning teams at DACH grocery and pharmacy chains (Edeka, Rewe, Lidl, dm, Rossmann) run the same forecasting problem at SKU-store-day granularity. The standard production stack is gradient-boosted trees with hand-engineered lag and rolling features for the point forecast, plus a parametric prediction interval for safety stock. The Favorita dataset is the most realistic public proxy for that stack: hierarchical (country -> store -> family), exogenously driven (oil, holidays, promotions), and noisy enough that naive seasonal baselines are competitive on a non-trivial slice.

## Target

`sales` (continuous, non-negative; mostly count-like with float precision because of weight-sold items). Out-of-sample evaluation: Root Mean Squared Logarithmic Error (RMSLE) per the Kaggle competition, plus per-family weighted RMSLE for the WRMSSE-style ranking used in the M5 follow-up literature.

## Hierarchy

- 54 stores, indexed by `store_nbr`, located across 22 cities and 16 states in Ecuador.
- 33 product families (`family`).
- 1,782 active series at the store-family level (54 * 33 with some empty combinations).
- Daily granularity. The training file is ~125 MB across roughly 3 million rows.

## Exogenous drivers

- `holidays_events.csv` - national, regional, and local holidays plus events; type column distinguishes Holiday, Transfer, Bridge, Work Day.
- `oil.csv` - daily WTI oil price; Ecuador's economy is oil-linked and the brief calls this out as a known macro driver.
- `transactions.csv` - daily store transaction count (proxy for footfall).
- `stores.csv` - store metadata (city, state, type, cluster).

## Deliverables (Liora MLE Phase 1 format)

- `brief.md` (this file)
- `data/README.md` - Kaggle CLI command, file-by-file schema, post-download steps
- `notebooks/01_EDA.ipynb` - raw, not executed
- `reports/references.md` - 26 verified academic references
- `src/model_baseline.py` - per-store-family Prophet / SARIMA with holiday and oil regressors
- `src/model_advanced.py` - LightGBM with lag features (1, 7, 14, 28), rolling means, quantile regression for prediction intervals
- `manuscripts/manuscript.md` - IMRaD, ~4500 words
- `deliverables/presentation.html` - self-contained, 10 slide-style sections
- `checkpoint.json` - status JSON

## Out of scope for Phase 1

- Hyperparameter optimisation (Optuna, hyperopt).
- Hierarchical reconciliation (MinT, OLS, BU/TD).
- Probabilistic deep learning (DeepAR, NBEATSx, Temporal Fusion Transformer).
- Earthquake-week (April 2016) anomaly handling beyond a binary flag.

These are documented in the manuscript Discussion as explicit follow-up directions.
