# Validation Report - Project 09 Favorita Store Sales

## Compact summary

**Overall: PASS-WITH-WARNINGS**

All eleven validator checks executed against the scaffold artefacts. Notebook JSON parses, both Python scripts have valid syntax, manuscript follows full IMRaD structure with all required sections, presentation HTML is fully self-contained (zero external href/src), zero em-dashes across all artefacts, zero AI-tell strings, all 18 distinct inline citations map to entries in `reports/references.md`, all 5 CrossRef-sampled DOIs return HTTP 200 with matching titles, and the checkpoint schema contains all four required fields plus extras. The single warning is the manuscript word count: 4,271 words sits inside the 4,000-5,000 target band and is therefore PASS, not WARN. The genuine WARN is task #11 schema-extra: checkpoint includes `phase`, `needs_main_session_execution`, and `blockers` beyond the required four, which is additive and does not violate the schema. No FAILs.

## Findings

- [PASS] Task 1 - Notebook validity: `python3 -c "import json; json.load(open('notebooks/01_EDA.ipynb'))"` returned cleanly. Notebook JSON parses.
- [PASS] Task 2a - Python syntax (baseline): `ast.parse('src/model_baseline.py')` succeeded. No syntax errors.
- [PASS] Task 2b - Python syntax (advanced): `ast.parse('src/model_advanced.py')` succeeded. No syntax errors.
- [PASS] Task 3 - Manuscript word count: `wc -w` reports 4,271 words. Target band 4,000-5,000. Inside target.
- [PASS] Task 4 - Self-contained HTML: `grep -E 'href="http|src="http' deliverables/presentation.html` returned 0 hits. Presentation has no external resources.
- [PASS] Task 5 - IMRaD completeness: Title, Abstract, Introduction (1), Data (2), Methods (3), Results (4), Discussion (5), Conclusion (6), References all present with proper numeric headings.
- [PASS] Task 6 - Method drift: Methods section names Prophet (multiplicative seasonality, holidays regressor, oil regressor, changepoint_prior_scale=0.05), SARIMA(1,1,1)(1,1,1,7) fallback, log1p target transform, panel LightGBM with `num_leaves=127, learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=100, 3000 rounds`, lag features {1,7,14,28}, rolling means/std at {7,28}, calendar features, holiday flags (national/regional/local), oil price, onpromotion + lag_7, earthquake-window flag, native categoricals, and quantile heads at tau in {0.1,0.5,0.9}. Every named method is present in `model_baseline.py` or `model_advanced.py`. No drift.
- [PASS] Task 7 - Citation drift: 18 unique inline citations extracted from manuscript (Athanasopoulos 2009, Bergmeir 2018, Chen 2016, Croston 1972, Diebold 1995, Gneiting 2007, Hewamalage 2021, Hyndman 2006, Hyndman 2008, Kim 2016, Lim 2021, Makridakis 2018, Makridakis 2020, Makridakis 2022, Montero-Manso 2020, Olivares 2023, Salinas 2020, Taylor 2018). All 18 map to entries in `reports/references.md` (which has 26 numbered entries). No orphans.
- [PASS] Task 8 - CrossRef live verification, 5 sampled DOIs:
  - 10.1016/j.ijforecast.2019.04.014 -> 200 -> "The M4 Competition: 100,000 time series and 61 forecasting methods"
  - 10.1145/2939672.2939785 -> 200 -> "XGBoost"
  - 10.1016/j.ijforecast.2019.07.001 -> 200 -> "DeepAR: Probabilistic forecasting with autoregressive recurrent networks"
  - 10.1080/00031305.2017.1380080 -> 200 -> "Forecasting at Scale"
  - 10.1057/jors.1972.50 -> 200 -> "Forecasting and Stock Control for Intermittent Demands"
  All five return HTTP 200 with title-match against the references.md text.
- [PASS] Task 9 - Em-dash scan: 0 hits across brief.md, notebook, references.md, both src files, manuscript, and presentation HTML. Total = 0.
- [PASS] Task 10 - AI-tell scan: `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude' .` returned no matches. Zero hits.
- [PASS] Task 11 - Checkpoint schema: keys present = `['project_number', 'title', 'methodology', 'phase', 'status', 'needs_main_session_execution', 'blockers']`. All four required fields (project_number, title, methodology, status) are present. `phase` substitutes for an explicit status-detail field; `status` is an object containing executable substates. Extras are additive.
- [WARN] Project type note: Project #09 is scaffold-only (#1-#8 are executed). Per QA rules, no saved-model artefact check applies. Deliverables folder contains only `presentation.html`; this is expected for a scaffold-stage project.

## Sources

- `/root/AI/liora_projects/09_favorita_sales/brief.md`
- `/root/AI/liora_projects/09_favorita_sales/notebooks/01_EDA.ipynb`
- `/root/AI/liora_projects/09_favorita_sales/src/model_baseline.py`
- `/root/AI/liora_projects/09_favorita_sales/src/model_advanced.py`
- `/root/AI/liora_projects/09_favorita_sales/manuscripts/manuscript.md`
- `/root/AI/liora_projects/09_favorita_sales/reports/references.md`
- `/root/AI/liora_projects/09_favorita_sales/deliverables/presentation.html`
- `/root/AI/liora_projects/09_favorita_sales/checkpoint.json`
- CrossRef API: `https://api.crossref.org/works/{doi}` (live, 2026-05-08)
