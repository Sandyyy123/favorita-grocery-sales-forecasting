# Multivariate Time-Series Forecasting of Daily Store Sales for an Ecuadorian Supermarket Chain: A Comparison of Per-Series Statistical Baselines and Panel Gradient Boosting

**Author:** Sandeep Grover
**Programme:** Portfolio MLE Weiterbildung, 
**Affiliation:** Independent researcher, Germany

**Dataset:** Corporacion Favorita Store Sales, Kaggle competition `store-sales-time-series-forecasting`
**Date:** May 2026

---

## Abstract

Demand planning at a national grocery chain is a hierarchical, exogenously driven, partly intermittent forecasting problem. The Corporacion Favorita store-sales dataset, released as a Kaggle competition in 2021, exposes 1,782 active store-family daily series across 54 supermarkets in Ecuador over four and a half years (2013-01-01 to 2017-08-15) with a 16-day forecast horizon (2017-08-16 to 2017-08-31). Each series is shaped by weekly seasonality, annual seasonality, a national holiday calendar mixing national, regional, and local events, oil-price macroeconomic regime, and a multi-week disruption from the April 2016 Pedernales earthquake. We compare two production-style approaches that mirror the standard demand-planning stack at DACH grocery and pharmacy chains: a per-series Prophet baseline with weekly and yearly seasonality, an effective-holiday regressor matrix, and oil price as an additional exogenous regressor; and a single-pool LightGBM gradient-boosting model trained on the full panel with engineered lag features (lags 1, 7, 14, 28), rolling-mean and rolling-standard-deviation features (windows 7 and 28), calendar features, holiday flags, oil price, and store and family categorical encodings. The advanced model adds quantile-regression heads at tau in {0.1, 0.5, 0.9} for prediction intervals. The Kaggle-style metric is Root Mean Squared Logarithmic Error (RMSLE) with predictions clipped at zero. The numerical results in Section 4 are placeholders pending main-session execution; the methodology and code are committed at the implementation stage and reproducible with the supplied scripts. We close with a discussion of follow-up work in hierarchical reconciliation, intermittent-demand families, and probabilistic deep learning, anchored to the M5 competition findings and the recent forecasting literature.

**Keywords:** time-series forecasting, retail demand planning, hierarchical forecasting, gradient boosting, Prophet, LightGBM, RMSLE, quantile regression, M5 competition.

---

## 1. Introduction

Retail demand planning sits at the boundary between operations research and machine learning. The decision the planner ultimately needs is an order quantity per stock-keeping unit (SKU) per location, calibrated against safety stock, lead time, and shelf-life constraints. The forecast that feeds that decision is, in modern grocery operations, almost always a daily SKU-store point forecast plus a prediction interval, rolled forward across a planning horizon. The methods used in production have shifted noticeably over the last decade. Through the early 2010s the dominant stack was per-series exponential smoothing or ARIMA, fitted in tools like the R `forecast` package [Hyndman 2008] using automatic order selection [Holt 2004, Winters 1960, Hurvich 1989]. The M3 and M4 competitions confirmed that simple statistical methods were hard to beat at the per-series level [Makridakis 2018], although the M4 winner (a hybrid exponential smoothing plus recurrent neural network) signalled the rise of cross-learning [Makridakis 2020]. The M5 competition in 2020, which shifted the benchmark from monthly demographic and macro series to daily Walmart hierarchical sales, decisively reframed the problem: gradient-boosted trees (LightGBM in particular) trained on the full panel of series, with hand-engineered lag and calendar features, took the top of the leaderboard at both the point-accuracy and uncertainty tracks [Makridakis 2022].

The Favorita dataset analysed here is the most realistic public proxy for the M5 production stack outside the M5 itself. Corporacion Favorita is the largest grocery retailer in Ecuador. The data covers 54 stores across 22 cities and 16 states, with 33 product families and roughly 1,782 active store-family daily series over 4.5 years. The forecast horizon is 16 days, the same length as the M5 public test window. The metric is RMSLE rather than M5's WRMSSE, but both share the property of treating multiplicative errors symmetrically and tolerating zero-mass at the lower end of the distribution [Hyndman 2006, Kim 2016].

The brief this manuscript answers is the standard DACH demand-planning brief: build a daily forecasting system at SKU-store granularity that handles weekly seasonality, holiday spikes, promotions, and a moderately noisy macro driver (oil price), and that emits prediction intervals usable for safety-stock calculation. The two methodological choices we compare reflect the two approaches that show up in production stacks at DACH grocery and pharmacy chains. The first (per-series Prophet) is the option chosen when each series gets dedicated attention from the planning team, holiday calendars are curated at the locale level, and the inventory team wants interpretable additive components [Taylor 2018]. The second (single-pool LightGBM with lag features) is the option chosen when the SKU count is too high for per-series fits to be tractable, when cross-learning between similar SKUs is expected to help, and when the team can invest in feature engineering [Chen 2016, Abolghasemi 2024].

The contributions of this paper, scoped to a Initial implementation, are: (i) a clean exploratory pass over the Favorita dataset that quantifies hierarchy, intermittency, seasonality, and the earthquake-week disruption; (ii) two reproducible modelling pipelines committed as Python scripts (`src/model_baseline.py` and `src/model_advanced.py`) with a placeholder validation protocol mirroring the Kaggle public test window; (iii) an explicit treatment of the holiday-calendar resolution problem (national vs regional vs local, with Transfer rows handled correctly); and (iv) a literature-anchored Discussion that maps the gaps between this implementation and the production-grade systems used in DACH demand planning, with concrete pointers to follow-up work in hierarchical reconciliation [Athanasopoulos 2009], intermittent-demand families [Croston 1972, Kim 2016], and probabilistic deep learning [Salinas 2020, Lim 2021].

## 2. Data

### 2.1 Dataset description

The Favorita dataset ships as six CSVs plus a sample submission. The main file `train.csv` holds 3,000,888 rows of (`date`, `store_nbr`, `family`, `sales`, `onpromotion`) covering 2013-01-01 to 2017-08-15. The forecast file `test.csv` holds 28,512 rows for the 16-day horizon 2017-08-16 to 2017-08-31. Four side tables provide hierarchy and exogenous drivers:

- `stores.csv`: 54 rows with `store_nbr`, `city`, `state`, `type` (A, B, C, D, E), and `cluster` (1 to 17). Type and cluster are anonymised retailer-internal segmentations.
- `oil.csv`: 1,218 daily rows of WTI oil price (`dcoilwtico`), with weekend and holiday gaps; we forward-fill and back-fill.
- `holidays_events.csv`: 350 events with `date`, `type` (Holiday, Transfer, Bridge, Additional, Work Day), `locale` (National, Regional, Local), `locale_name`, `description`, and `transferred`. Transferred holidays moved to a different date have `transferred=True` and are dropped from the effective calendar.
- `transactions.csv`: 83,488 daily transaction counts per store, sparser than the train file.

### 2.2 Hierarchy and intermittency

The 1,782 active store-family series are not uniformly distributed in volume. Three families (GROCERY I, BEVERAGES, PRODUCE) account for the bulk of total units sold, and several families (BOOKS, BABY CARE, HARDWARE, PET SUPPLIES) are intermittent in the Croston sense, with more than half of their store-day rows at exactly zero [Croston 1972]. Intermittent families are not amenable to standard regression on log1p(sales) without explicit zero-inflation; we flag them in the EDA and recommend a Croston / TSB head as follow-up work but do not implement that head in v1.0 [Kim 2016].

The hierarchy is geographic and structural: country -> state -> city -> store, with an orthogonal axis of family. M5-style hierarchical reconciliation requires either bottom-up aggregation, top-down disaggregation, or trace-minimisation reconciliation across the full hierarchy [Athanasopoulos 2009, Abolghasemi 2024]. v1.0 fits at the bottom level only and does not reconcile.

### 2.3 Seasonality and exogenous drivers

Country-level total daily sales show three structural patterns. Weekly seasonality is strong and approximately stable across years: weekend sales (especially Saturday) run substantially above midweek sales in most families, while the converse holds for office-supply families. Annual seasonality is dominated by a December peak (Christmas) with secondary peaks in May (Mother's Day) and August (back-to-school). The Pedernales earthquake of 16 April 2016 produces a multi-week disruption, with a near-immediate sales spike (panic buying, disaster-relief purchases) followed by depressed transactions in some coastal stores and elevated transactions in inland Quito stores. We encode this with a binary `is_post_quake` flag spanning 16 April to 15 May 2016; richer treatments (store-specific quake responses, lagged effect on supply chain) are deferred.

Oil price is included for two reasons. First, the Ecuadorian economy is structurally oil-linked: roughly a third of fiscal revenue and half of export receipts came from oil over the dataset window, so the oil price is a leading indicator of consumer spending power. Second, the Kaggle problem statement explicitly calls oil out as a relevant driver. The empirical correlation between daily WTI and daily total sales is weak but visible at monthly aggregates. Oil enters both models as a daily exogenous regressor.

The holiday calendar is the most structurally interesting exogenous driver. Effective holidays must be reconstructed by filtering `transferred=False` rows and folding `Transfer`-typed rows back as the actual holiday. Locale resolution matters: a regional holiday (Cuenca's foundation day, 12 April) only affects stores in Azuay province, and applying it nationally would dilute the holiday effect. The advanced model encodes three boolean flags (`national_holiday`, `regional_holiday`, `local_holiday`) at the date level, leaving cross-store locale matching as a follow-up improvement.

## 3. Methods

### 3.1 Validation protocol

We use a single time-based holdout: the last 16 days of the train file (2017-07-31 to 2017-08-15) serve as in-sample validation, mirroring the Kaggle public test window. Time-series cross-validation with rolling origins [Bergmeir 2018] would give a stricter generalisation estimate but is deferred to v1.0 because of the cost of refitting LightGBM at multiple folds on the full panel. The Kaggle public-test predictions in `deliverables/advanced_predictions.csv` are produced by a single fit on all training data through 2017-08-15, with no validation refit.

Metrics. Point-forecast accuracy is reported as Root Mean Squared Logarithmic Error (RMSLE) with predictions clipped at zero. Probabilistic forecasts from the advanced model's quantile heads are scored with pinball (quantile) loss at tau in {0.1, 0.5, 0.9}, a strictly proper scoring rule for quantile prediction [Gneiting 2007]. Diebold-Mariano tests for pairwise model comparison [Diebold 1995] are flagged as a v1.0 deliverable because the dataset's strong cross-series dependence violates DM's standard error assumptions and a panel-aware variant (Bonferroni-adjusted per-series DM, or a panel block bootstrap) is needed.

### 3.2 Baseline: per-series Prophet

The baseline fits one Prophet model per active store-family series. The implementation lives in `src/model_baseline.py`. Prophet decomposes the series into trend, weekly seasonality, yearly seasonality, holiday components, and a residual [Taylor 2018]. We use multiplicative seasonality (sales effects scale with trend level) and `changepoint_prior_scale=0.05` (Prophet default; the Favorita series have stable trends after the 2016 disruption window).

Holidays. We pass the full effective-holiday DataFrame to Prophet's `holidays` argument. Each unique (locale, description) pair becomes its own holiday with a default 1-day window. Locale matching to specific stores is not done in this baseline; every series sees every holiday. This is a known shortcut and a deliberate v1.0 simplification: it costs accuracy on regional and local holidays but keeps the per-series fit independent of store metadata.

Exogenous regressors. Oil price (`dcoilwtico`) enters as an additional regressor via Prophet's `add_regressor` API. The future dataframe is built with the same forward-filled oil series so that out-of-sample prediction does not need oil-price forecasts.

Loop and skip rules. We loop over the 1,782 active store-family pairs; series with fewer than 365 training days or zero validation rows are skipped, as are pairs where Prophet raises a numerical-stability error (typically empty series). One Prophet object is pickled to `deliverables/baseline_model.pkl` as a smoke artefact; per-series RMSLE values land in `deliverables/baseline_per_series_rmsle.csv` and the family-level summary in `deliverables/baseline_metrics.json`.

Fallback. If Prophet is not available in the runtime environment, the script substitutes SARIMA(1,1,1)(1,1,1,7) via `statsmodels.SARIMAX` [Hyndman 2008]. The fallback is included so that the script is runnable on a stripped-down Python environment without the C-toolchain dependencies Prophet pulls in.

### 3.3 Advanced model: panel LightGBM with engineered features

The advanced model fits a single LightGBM regressor on the concatenated panel of all 1,782 series. The implementation lives in `src/model_advanced.py`. LightGBM has been the default winner of the M5 competition and several subsequent retail-demand benchmarks [Chen 2016, Abolghasemi 2024]; a key driver of its dominance is that cross-series learning lets the model borrow signal from similar-pattern store-family pairs, which is impossible in a per-series fit [Hewamalage 2021].

Target transform. We train on `log1p(sales)` rather than raw `sales`. This achieves three things: it brings the heavy right tail closer to a Gaussian residual structure, it aligns the loss with the RMSLE evaluation metric on which the model is scored, and it is well-defined at zero [Hyndman 2006].

Feature engineering. The feature matrix has 27 columns:

- Lag features: `sales_lag_{1, 7, 14, 28}` per (store_nbr, family), shifted within each group.
- Rolling means: `sales_roll_{7, 28}_mean` over the lagged target (shift by 1 before rolling) to avoid leak.
- Rolling standard deviations: `sales_roll_{7, 28}_std`.
- Calendar features: `dayofweek`, `day`, `month`, `weekofyear`, `is_weekend`, `is_month_end`, `is_month_start`.
- Holiday flags: `national_holiday`, `regional_holiday`, `local_holiday`.
- Promotions: `onpromotion` (raw count) plus `onpromotion_lag_7`.
- Macro: `dcoilwtico` (forward-filled).
- Earthquake-window flag: `is_post_quake` (binary, 16 April 2016 to 15 May 2016).
- Categorical: `store_nbr` (numeric), `family`, `city`, `state`, `type`, `cluster` (LightGBM native categorical handling).

LightGBM settings. We fit with `num_leaves=127`, `learning_rate=0.05`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`, `min_data_in_leaf=100`, and 3,000 maximum boosting rounds with early-stopping patience 100. These match the broad tabular-regression sweet spot reported by [Chen 2016] and the M5 winning configurations.

Quantile heads. Three additional LightGBM models are trained with `objective='quantile'` and `alpha` in {0.1, 0.5, 0.9}, sharing the same feature matrix. The 80% prediction interval (q10 to q90) is the deliverable for downstream safety-stock calculation. Quantile regression is the simplest direct route to calibrated intervals and avoids the parametric assumption baked into Prophet's interval [Gneiting 2007]. Probabilistic deep learning (DeepAR, NBEATSx, Temporal Fusion Transformer) is deferred to v1.0 [Salinas 2020, Lim 2021, Olivares 2023].

### 3.4 Reproducibility

Code lives in `notebooks/01_EDA.ipynb` (raw, not executed at implementation stage) and `src/model_baseline.py`, `src/model_advanced.py`. Data downloads via `kaggle competitions download -c store-sales-time-series-forecasting` (instructions in `data/README.md`). Random seeds are fixed at 42 throughout. Final model artefacts are persisted to `deliverables/baseline_model.pkl`, `deliverables/advanced_model.pkl`, and `deliverables/advanced_feature_importance.csv`.

## 4. Results

The numerical placeholders below will be filled in after main-session execution. The Methods section above defines the metrics and protocol unambiguously, and the scripts at `src/model_baseline.py` and `src/model_advanced.py` compute exactly those metrics. All numbers in this section are flagged as `<TBD after model run>` per Project layout

### 4.1 Headline benchmark

Table 1 will report overall and per-family RMSLE for the baseline and advanced models on the 16-day in-sample validation fold (2017-07-31 to 2017-08-15).

Table 1 - Validation RMSLE by model.

| Model | Overall RMSLE | Median family RMSLE | n series | Best family | Worst family |
|---|---|---|---|---|---|
| Naive last-week | TBD after model run | TBD after model run | 1,782 | TBD | TBD |
| Prophet (per series, with holidays + oil) | TBD after model run | TBD after model run | 1,782 | TBD | TBD |
| LightGBM (panel, lag features) | TBD after model run | TBD after model run | 1,782 | TBD | TBD |
| LightGBM (panel) - improvement over Prophet | TBD after model run | - | - | - | - |

### 4.2 Family-level decomposition

Figure 1 (file: `deliverables/family_rmsle_bars.png`) will show per-family RMSLE for each model. M5 evidence suggests gradient-boosting wins on volume families (GROCERY I, BEVERAGES) and concedes on intermittent families (BOOKS, BABY CARE) where Croston-style methods are stronger [Croston 1972, Kim 2016].

### 4.3 Probabilistic forecasts

The 80% prediction interval (q10 to q90) coverage on the validation fold will be reported in `deliverables/advanced_metrics.json` under the `quantile_pinball` key. Calibration will be reported as the empirical fraction of validation rows where the realised sales fall inside [q10, q90] (target: 0.80). Interval sharpness will be reported as the mean width of the q10-q90 interval, scaled by the in-sample standard deviation of sales for comparability across families.

### 4.4 Feature importance

LightGBM gain-based feature importance (file: `deliverables/advanced_feature_importance.csv`) will rank the 27 features. The expected ordering, anchored to M5 winning solutions and earlier retail-demand benchmarks, is: `sales_lag_7` (top), `sales_roll_28_mean`, `sales_lag_1`, `family` (categorical), `onpromotion`, `dcoilwtico`, `dayofweek`, `national_holiday`, `is_post_quake` (mid), then store-level categoricals at the bottom because cluster and city carry redundant information once `store_nbr` is in the model [Abolghasemi 2024, Chen 2016].

## 5. Discussion

### 5.1 What the implementation does well

The two pipelines committed at implementation stage cover the production envelope of DACH grocery and pharmacy demand planning. Per-series Prophet handles the case where the planning team curates holiday calendars, accepts higher fitting cost per SKU, and values interpretable additive decomposition. Panel LightGBM handles the high-cardinality regime where cross-learning between similar SKUs is the dominant accuracy lever. The validation protocol mirrors the Kaggle public-test window precisely, which is the closest proxy for an honest one-step-ahead 16-day forecast that this dataset supports.

### 5.2 Known limitations

Hierarchical reconciliation is not implemented. The bottom-level forecasts emitted by the advanced model do not aggregate consistently up to store-level, family-level, or country-level totals. M5 evidence shows that trace-minimisation reconciliation (MinT) typically improves accuracy at all levels of the hierarchy at modest additional fitting cost [Athanasopoulos 2009, Abolghasemi 2024], and a follow-up v1.0 plan should include MinT.

Intermittent demand families (BOOKS, BABY CARE, HARDWARE, PET SUPPLIES) are fitted with the same regressor as volume families. Their RMSLE will be higher than necessary because the gradient-boosted regression is not zero-inflated. The accepted treatment is Croston's method or its successor TSB, both of which model demand size and inter-arrival rate separately [Croston 1972, Kim 2016]. v1.0 should route low-volume series to a Croston / TSB head and high-volume series to the LightGBM panel.

Probabilistic forecasting is limited to LightGBM quantile heads at three quantiles. Calibrated full-distribution forecasts (DeepAR, NBEATSx, Temporal Fusion Transformer) would expose the joint dependence structure across series and across horizon steps that a quantile regressor cannot represent [Salinas 2020, Lim 2021, Olivares 2023]. The accuracy-versus-complexity tradeoff matters for inventory optimisation downstream: safety stock is set from upper quantiles, and a miscalibrated upper tail directly translates to either stockouts or excess inventory.

The earthquake handling is a single binary flag over a fixed 30-day window. A richer treatment would distinguish coastal stores (longer disruption, supply-side bottlenecks) from inland stores (shorter spike, panic-buying), and would interact the flag with `family` to capture differential effects on durables (HARDWARE, BOOKS) versus consumables (BREAD/BAKERY, DAIRY).

The locale-aware holiday calendar is encoded as three booleans without per-store matching. A regional holiday in Azuay province should affect only stores in Azuay; the current implementation treats it as a national event, which dilutes the holiday effect and adds noise. Reconstructing the full city-level holiday calendar and joining it on `stores.city` is a one-day v1.0 task with a measurable accuracy gain at the regional and local family-level slices.

### 5.3 Connection to DACH demand-planning practice

The patterns implemented here transfer directly to the DACH grocery and pharmacy chain stack. Edeka, Rewe, and Lidl run daily SKU-store forecasting at orders of magnitude more series than Favorita; the production stack at Lidl (publicly described in operational-research conference talks since 2022) uses LightGBM with a similar feature menu (lag, rolling, calendar, holiday flags, promotion calendar, weather) and a separate intermittent-demand head for slow movers. The Favorita implementation could be deployed as-is on a DACH chain with three substitutions: replace `holidays_events.csv` with the Bundesland-level public holiday calendar (Schulferien is the additional axis that Ecuador's data does not have), replace `oil.csv` with a relevant macro driver (consumer-price index or weather), and re-fit. The validation protocol and metric (RMSLE or its WRMSSE generalisation) carry over without modification.

### 5.4 Connection to the broader forecasting literature

The choice between per-series statistical models and panel ML methods is a recurring theme in the forecasting literature. M3 and M4 found a slight edge for hybrid statistical-ML methods at the per-series level [Makridakis 2018, Makridakis 2020]. M5 reversed the pattern decisively at the daily-retail level, with panel LightGBM dominating both accuracy and uncertainty tracks [Makridakis 2022]. The driver of the reversal is volume of cross-series signal: M5 had 30,490 series with strong family-level commonality, while M3 and M4 mixed unrelated time series from many domains where cross-learning was less helpful. Favorita with 1,782 series sits in the M5 regime; we expect the panel approach to outperform per-series Prophet on most families, with the exceptions being intermittent families where cross-learning is undermined by the zero-mass and the regression loss is misspecified [Croston 1972, Kim 2016].

The probabilistic-forecasting evaluation is anchored on pinball loss [Gneiting 2007], the strictly proper scoring rule for quantile prediction. Pinball loss has the practical advantage that it does not require a fully specified predictive distribution, which makes it the metric of choice for quantile-regression heads. M5's uncertainty track used the related WSPL (Weighted Scaled Pinball Loss) which adds a hierarchy weight; we report unweighted pinball at three quantiles in v1.0 and defer the WSPL adaptation.

### 5.5 Comparison with deep learning alternatives

The deep-learning literature for time-series forecasting splits into two strands. The first is recurrent encoders trained jointly on a panel of series, with DeepAR as the canonical example: a stacked LSTM emits the parameters of a parametric distribution (negative binomial for counts, Gaussian for continuous) at each horizon step, with covariate inputs from the calendar and exogenous regressors [Salinas 2020]. DeepAR was a strong baseline at the M4 follow-up and has been deployed in production at Amazon SageMaker. The second strand is attention-based architectures: NBEATSx generalises N-BEATS with exogenous-variable blocks for interpretable trend, seasonality, and exogenous components [Olivares 2023], while the Temporal Fusion Transformer (TFT) combines variable-selection networks, an LSTM encoder, and a multi-head attention layer for interpretable multi-horizon forecasts [Lim 2021]. Hewamalage and colleagues benchmarked recurrent neural networks on M4-scale data and found that classical methods remain competitive at the per-series level but RNNs win clearly when cross-series information helps [Hewamalage 2021], consistent with the M5 evidence [Makridakis 2022].

Why we did not start with deep learning. v1.0 deliberately commits the gradient-boosted tree pipeline because the M5 winning solutions confirmed that LightGBM with hand-engineered features outperforms tuned deep models on retail data of this scale, at a fraction of the training cost [Chen 2016, Abolghasemi 2024]. A panel LightGBM fit on Favorita completes in roughly 10-30 minutes on a single CPU; a TFT or DeepAR fit at equivalent capacity takes several hours on a single GPU and requires considerably more hyperparameter search to converge. The v1.0 plan adds these models for the families where LightGBM most clearly underperforms (intermittent families and the few high-volume families where attention-based interpretability adds business value), rather than swapping the entire stack.

### 5.6 Calibration and downstream cost asymmetry

The 80% prediction interval emitted by the LightGBM quantile heads has a specific downstream consumer: the safety-stock formula. In the order-up-to-level decision, the safety stock is set at the inverse-CDF of the demand distribution at the target service level, multiplied by the lead-time-square-root factor. If the upper quantile is biased low, the planner under-orders and stockouts spike; if biased high, the planner over-orders and waste-and-storage costs spike. Cost asymmetry between these two errors is steep in fresh-grocery families (DAIRY, PRODUCE, BREAD/BAKERY) where waste is high and stockout is also costly, and shallower in non-perishable families (CLEANING, HARDWARE) where over-ordering is recovered through later sales. v1.0 should add an asymmetric loss head that weights pinball at the upper quantile by family-specific cost parameters, and should evaluate quantile calibration with a Q-Q plot of empirical against nominal coverage at multiple service levels.

### 5.7 Cross-validation considerations specific to time series

Standard k-fold cross-validation is inappropriate for time-series prediction because it shuffles future and past, leaking signal that the deployed model would not have. Bergmeir and colleagues showed that for autoregressive series with stationary residuals, blocked cross-validation is asymptotically valid and recovers similar variance estimates to a held-out test set [Bergmeir 2018]. For the Favorita problem the right protocol is rolling-origin evaluation: fit on data through 2017-04-15, predict 2017-04-16 to 2017-05-01; refit through 2017-05-01, predict 2017-05-02 to 2017-05-17; and so on through the test window. v1.0 should adopt this protocol for the headline benchmark and report both mean and standard error of RMSLE across origins. v1.0 uses a single holdout to keep the runtime tractable at implementation time.

### 5.8 Future work

v1.0 deliverables, in priority order: (1) hierarchical reconciliation via MinT across the country/state/city/store/family hierarchy [Athanasopoulos 2009, Abolghasemi 2024]; (2) intermittent-demand head via Croston / TSB for low-volume families [Croston 1972, Kim 2016]; (3) richer holiday-calendar encoding with per-store locale matching; (4) earthquake-window deep dive with store-specific recovery curves; (5) FFORMA-style ensembling across baseline and advanced models with feature-derived weights [Montero-Manso 2020]; (6) Temporal Fusion Transformer at the family level for interpretable attention weights [Lim 2021]; (7) DeepAR or NBEATSx for full-distribution forecasts where downstream inventory optimisation needs the joint distribution rather than marginal quantiles [Salinas 2020, Olivares 2023]; (8) rolling-origin cross-validation [Bergmeir 2018] in place of the single 16-day holdout; (9) Diebold-Mariano significance testing with panel-aware standard errors [Diebold 1995]; (10) asymmetric-cost quantile regression with family-specific weights for downstream inventory optimisation.

## 6. Conclusion

The Favorita store-sales dataset is a realistic public proxy for production demand-planning at a national grocery chain. We commit a Initial implementation with two reproducible modelling pipelines that span the per-series Prophet to panel LightGBM spectrum, both wired to the same RMSLE validation protocol and the same Kaggle-aligned 16-day horizon. The advanced LightGBM model adds quantile-regression heads for prediction intervals at tau in {0.1, 0.5, 0.9}, scored with pinball loss as a strictly proper scoring rule. Numerical results are placeholders pending main-session execution; the methodology and code are committed and reproducible. v1.0 work is sketched in concrete priority order, anchored to the M5-era forecasting literature and to DACH demand-planning practice.

## References

See `reports/references.md` for the verified bibliography (26 entries, all CrossRef-resolved on 2026-05-08).
