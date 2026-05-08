# Improvements - Project 09 (Favorita Store Sales Forecasting)

## Summary

The scaffold is methodologically coherent (Prophet baseline + panel LightGBM with quantile heads, RMSLE metric, log1p target, lag/rolling/calendar/holiday/promo/oil features). The biggest gaps are: a single time-based holdout instead of rolling-origin CV, no hierarchical reconciliation, store-locale resolution missing on holiday flags, no intermittent-demand head for sparse families, and no requirements pinning or seed plumbing through LightGBM. The single highest-leverage change is to replace per-store-family Prophet with a per-family panel LightGBM-DART model and add MinT reconciliation.

---

## Top recommendation (single highest-leverage change)

**Replace the per-series Prophet baseline with a per-family panel LightGBM (DART) plus MinT trace-minimisation reconciliation across the country -> state -> city -> store -> family hierarchy.** The current Prophet baseline fits 1,782 separate models, ignores cross-series signal, applies every holiday to every store regardless of locale, and is the slowest part of the pipeline (Prophet at ~5-15 s per series implies 2-7 hours of CPU). M5 winners and Abolghasemi 2024 both demonstrate that even a single per-family LightGBM with cross-series features beats per-series statistical fits at this granularity, and MinT reconciliation reliably adds 2-5% accuracy gain at the family/store totals at a few minutes of additional compute. Concrete next steps:
1. Drop Prophet as the baseline; promote it to a sanity-check on three high-volume families only.
2. Train one LightGBM per family (33 models) on the panel of stores using `boosting_type='dart'` for variance reduction.
3. Implement MinT via `hierarchicalforecast` (Nixtla) using the OLS+shrinkage residual covariance estimator from Wickramasuriya 2019.
4. Report RMSLE at every level of the hierarchy (national, state, city, store, store-family) with reconciliation-on vs reconciliation-off ablation.

---

## Weakness 1: Single time-based holdout instead of rolling-origin CV  [HIGH]

The validation protocol uses one 16-day window (2017-07-31 to 2017-08-15) which sits inside the high-Christmas-build-up period and may not generalise to mid-year months. Bergmeir 2018 is cited in the manuscript but the scripts do not implement rolling-origin CV.

**Improvement.** Add a `tools/rolling_origin_cv.py` helper that fits the LightGBM panel at four cut-offs (2016-12-31, 2017-03-31, 2017-05-31, 2017-07-31) with the same 16-day forecast horizon at each. Report mean and standard error of RMSLE across the four origins. Total cost: 4x current LightGBM fit (~30-60 minutes on a single CPU).

---

## Weakness 2: Holiday flags ignore store locale  [HIGH]

`build_holiday_flags` in `model_advanced.py` collapses all holidays to per-date booleans (national/regional/local) without joining `locale_name` to `stores.city` or `stores.state`. A regional holiday in Azuay province currently fires for every store in the country, diluting signal and adding label noise on training rows.

**Improvement.** Replace the date-only flag table with a `(date, store_nbr) -> {is_natl, is_regional_for_this_store, is_local_for_this_store, days_to_next_natl, days_since_last_natl}` join keyed on `stores.state` and `stores.city`. Add 3 leading and 3 trailing day flags around each holiday (Christmas/New Year cause 7-10 day pre-spike pattern in grocery). This is a half-day Phase 2 task with 3-7% RMSLE improvement on regional and local holiday-affected slices.

---

## Weakness 3: No intermittent-demand head for sparse families  [HIGH]

Families like BABY CARE, BOOKS, HARDWARE, PET SUPPLIES, MAGAZINES have >50% zero-sales rows. Training LightGBM regression on `log1p(sales)` for these is misspecified - the model learns to predict near-zero everywhere and sacrifices the rare positive observations. The manuscript Discussion notes this as deferred but the code does not even segment.

**Improvement.** Add a Croston/TSB head (Croston 1972, Teunter 2011) for any family where p(sales==0) > 0.5 across the panel. Use `statsforecast.CrostonClassic` or `TSB` from Nixtla's statsforecast, route low-volume families to it via a family-list switch in `model_advanced.py`, and combine with the LightGBM head via simple weighted averaging based on per-family validation RMSLE. Expected gain: 5-15% RMSLE reduction on the 6-8 intermittent families.

---

## Weakness 4: No hyperparameter tuning, no Optuna, no calibration on quantile heads  [MEDIUM]

LightGBM hyperparameters are fixed at `num_leaves=127, learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=100`. The brief lists Optuna as out of scope for Phase 1, but even a 30-trial Optuna sweep with TPE on a 10% subsample would close most of the leaderboard gap. Quantile heads at tau=0.1/0.5/0.9 are not calibrated post-hoc; on Kaggle datasets they typically over-cover at q90 by 3-5 percentage points.

**Improvement.** Add `tools/lgb_optuna.py` running 30-50 trials on a stratified 10% panel subsample, optimising on the rolling-origin RMSLE. Then apply isotonic regression calibration to the quantile heads on a small held-out calibration fold (e.g. last 4 weeks of 2017-06) and report empirical coverage at q10/q50/q90 before vs after calibration in `deliverables/calibration_report.md`.

---

## Weakness 5: Reproducibility gaps - missing requirements, no seeds in LightGBM, data not pinned  [MEDIUM]

There is no `requirements.txt` in the project root. LightGBM's `train()` call sets no `seed`/`random_state` parameter (the manuscript claims "Random seeds are fixed at 42 throughout" but the only seed is in the `train_df.sample(random_state=42)` line). The Kaggle CSV is downloaded fresh each time; there is no md5/sha256 pin to detect schema drift.

**Improvement.** Add `requirements.txt` with pinned versions (`lightgbm==4.3.0, prophet==1.1.5, pandas==2.2.0, numpy==1.26.0, statsmodels==0.14.1, hierarchicalforecast==0.4.0`). Pass `seed=42, deterministic=True, force_col_wise=True` into LightGBM `params`. Compute and commit a `data/checksums.txt` with sha256 of each Kaggle CSV after first download. One-hour task.

---

## Weakness 6: Single-machine pandas pipeline at 3M rows is slow and not memory-aware  [MEDIUM]

`build_master_table` does a `pd.concat([train, test])` then per-group lag generation on the full 3M-row panel using pandas `groupby().shift()`. This works but takes 3-5 minutes and peaks above 4 GB RAM. For DACH-scale deployment (Lidl alone has 30,000+ SKU-store pairs across 12,000+ stores) this approach does not scale.

**Improvement.** Migrate the feature-engineering layer to `polars` (10-30x faster groupby-shift, half the RAM) or to `duckdb` SQL for declarative window expressions. Keep LightGBM as-is. Document the migration in `src/feature_engineering_polars.py` and benchmark vs the pandas version. Expected: feature-build time drops from ~4 minutes to 15-30 seconds.

---

## Weakness 7: Earthquake handling is a single 30-day binary flag  [LOW]

The `is_post_quake` flag spans 16 April to 15 May 2016 uniformly across all 54 stores. In reality, coastal stores (Manabi, Esmeraldas) had multi-week supply disruption, while Quito/Guayaquil inland stores saw spikes from disaster-relief shipments. Several Kaggle gold-medal solutions used per-store recovery curves.

**Improvement.** Replace the binary flag with `quake_distance_to_epicenter_km` (Pedernales lat/lon = -0.07, -80.05; geocode each store via `stores.city` -> lat/lon) and `days_since_quake` clipped to [0, 60]. Add an interaction `quake_distance * days_since_quake` for the recovery-curve effect. ~2-hour task; 1-2% RMSLE gain on April-May 2016 validation slices but minimal benefit for the August 2017 test window.

---

## Weakness 8: Presentation lacks the "what would change for a DACH client" slide  [LOW]

The 10-section HTML deck closes with a generic Conclusion. For Liora's DACH-grocery audience, the deck is missing the explicit slide that maps each Favorita signal to the corresponding DACH analogue: `holidays_events.csv` -> ferienkalender.de + Bundesland Schulferien API; `oil.csv` -> DWD weather (temperature/precipitation are stronger DACH demand drivers than oil); `transactions.csv` -> Kassenbon counts from the POS feed; `cluster` -> Edeka/Rewe assortment cluster ID.

**Improvement.** Add slide 11 "Porting to a DACH grocery chain" with a four-row mapping table (Favorita field -> DACH equivalent -> data-source pointer -> code-change line count). Cite the Lidl DSV operational-research talks (Wirtschaftsinformatik 2023, ORBEL 2024) and the EHI Retail Institute 2024 demand-forecasting benchmark. 1-hour task; directly increases the manuscript's commercial credibility.

---

## Priority summary

| # | Weakness | Priority | Effort | Expected RMSLE delta |
|---|---|---|---|---|
| Top | Per-family LightGBM-DART + MinT reconciliation | HIGH | 1-2 days | 3-8% overall, 5-10% at higher levels |
| 1 | Rolling-origin CV | HIGH | 4 hours | reporting only, not RMSLE |
| 2 | Locale-aware holiday flags | HIGH | 4 hours | 3-7% on regional/local slices |
| 3 | Croston/TSB head for intermittent families | HIGH | 6 hours | 5-15% on 6-8 sparse families |
| 4 | Optuna sweep + quantile calibration | MEDIUM | 1 day | 2-5% point + better coverage |
| 5 | Pinned requirements + LightGBM seed + data checksums | MEDIUM | 1 hour | reproducibility only |
| 6 | Polars/DuckDB feature pipeline | MEDIUM | 1 day | runtime 10x, no RMSLE delta |
| 7 | Quake distance + days-since instead of binary | LOW | 2 hours | 1-2% on 2016 slices |
| 8 | DACH-mapping slide in presentation | LOW | 1 hour | commercial credibility |
