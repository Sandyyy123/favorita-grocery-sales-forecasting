# Additional References - Favorita Store Sales Forecasting (Literature Scout)

Independent CrossRef + Europe PMC search, 2024-2026 publications. Each entry resolved live against the chosen API on 2026-05-08. Volume / issue / pages intentionally omitted per project rule.

## State-of-the-art callout (gaps in current `reports/references.md`)

Current `reports/references.md` covers M4/M5 (Makridakis et al.), DeepAR (Salinas 2020), Temporal Fusion Transformer (Lim 2021), NBEATSx (Olivares 2023), Prophet (Taylor 2018), MinT-style hierarchical work (Athanasopoulos 2009), and Abolghasemi 2024 on hierarchical promotions. The following SOTA gaps should be cited:

1. **Time-series foundation / pre-trained models** - TimesFM, Chronos, Lag-Llama. Zero-shot and few-shot baselines have become a 2024-26 standard reference point for retail demand work and are absent from the current bibliography.
2. **Channel-independent and inverted Transformers (PatchTST, iTransformer)** - the two architectures that displaced vanilla TFT as the long-horizon baseline in 2023-24, also absent.
3. **Conformal prediction for time-series intervals** - the production substitute for parametric quantile regression on RMSLE-style targets; the manuscript's quantile-LightGBM intervals should be benchmarked against it.
4. **M6 competition results** - the 2024-25 follow-up to M5 introduced calibrated probabilistic ranking; relevant to the "per-family weighted RMSLE" objective.
5. **2024-26 hierarchical reconciliation advances** - non-negative reconciliation (Girolimetto 2025), shape-preserving MinT (Gonzalez-Sierra 2026), and forecast-combination-based reconciliation (Di Fonzo 2024) extend the 2009 Athanasopoulos baseline currently cited.

---

## Foundation and pre-trained time-series models (2024-2026)

1. Llanes-Guilarte DS, Herrera-Semenets V, Bustio-Martinez L. Forecasting Electrical Demand with Zero-Shot Lag Llama and TimesFM V2. Lecture Notes in Computer Science. 2026. DOI: 10.1007/978-3-032-11358-0_27
2. Vishwas BVK, Macharla SR. TimesFM: Time Series Forecasting Using Decoder-Only Foundation Model. Time Series Forecasting Using Generative AI. 2025. DOI: 10.1007/979-8-8688-1276-7_8
3. Pertsev Y, Korotka L. Innovative Approach to Time Series Forecasting: From Traditional Methods to the Cutting-Edge Model TimesFM. Information Technologies in Metallurgy and Machine Building. 2024. DOI: 10.34185/1991-7848.itmm.2024.01.084
4. Li H, Zhu Z, Chen X. Remaining Useful Life Prediction of Lithium-Ion Batteries Using Lag-Llama Model with Auto-Correlation Analysis. IEEE International Conference on Systems, Man, and Cybernetics. 2024. DOI: 10.1109/smc54092.2024.10831068
5. Liu Y, Wang X, Cao Z. A framework using large time series model for early warning of infectious diseases. Infectious Disease Modelling. 2026. DOI: 10.1016/j.idm.2025.08.006

## Architectures: PatchTST, iTransformer, hybrid Transformers (2024-2026)

6. Fan H. Enhancing Long-Term Time Series Forecasting via Hybrid DLinear-PatchTST Ensemble Framework. Applied and Computational Engineering. 2025. DOI: 10.54254/2755-2721/2025.kl22290
7. Lu K, Huo M, Li Y. CT-PatchTST: Channel-Time Patch Time-Series Transformer for Long-Term Renewable Energy Forecasting. International Conference on Computer and Information Processing Technology. 2025. DOI: 10.1109/iscipt67144.2025.11265471
8. Kamal M, Shiddiqi AM, Nurhayati E. ACyLeR: An Enhanced iTransformer for Long-Term Time-Series Forecasting Using Adaptive Cycling Learning Rate. International Conference on Innovation and Intelligence for Informatics, Computing, and Technologies. 2024. DOI: 10.1109/3ict64318.2024.10824647
9. Huang X, Tang J, Shen Y. Long time series of ocean wave prediction based on PatchTST model. Ocean Engineering. 2024. DOI: 10.1016/j.oceaneng.2024.117572
10. Shao Z, Wang Z, Yao X. ST-MambaSync: Complement the power of Mamba and Transformer fusion for less computational cost in spatial-temporal traffic forecasting. Information Fusion. 2025. DOI: 10.1016/j.inffus.2024.102872
11. Wang Q, Nicodemas KA. Hierarchical Attention Transformer for Multivariate Time Series Forecasting. Computers, Materials and Continua. 2026. DOI: 10.32604/cmc.2026.074305

## Hierarchical reconciliation (2024-2026)

12. Gonzalez-Sierra M, Velez JI, Arango-Manrique A. Shape-Preserving Minimum Trace (SP-MinT): A Regularized Forecast Reconciliation Method for Hierarchical Time Series. Research Square. 2026. DOI: 10.21203/rs.3.rs-9161917/v1
13. Di Fonzo T, Girolimetto D. Forecast combination-based forecast reconciliation: Insights and extensions. International Journal of Forecasting. 2024. DOI: 10.1016/j.ijforecast.2022.07.001
14. Girolimetto D. Non-Negative Forecast Reconciliation: Optimal Methods and Operational Solutions. Forecasting. 2025. DOI: 10.3390/forecast7040064
15. Quinn CO, Corliss GF, Povinelli RJ. Cross-Temporal Hierarchical Forecast Reconciliation of Natural Gas Demand. Energies. 2024. DOI: 10.3390/en17133077
16. Yan X, Zhang H, Miao Q. A novel sales forecast framework based on separate feature extraction and reconciliation under hierarchical constraint. Computers and Industrial Engineering. 2025. DOI: 10.1016/j.cie.2025.110875
17. Zhang B, Panagiotelis A, Li H. Constructing hierarchical time series through clustering. International Journal of Forecasting. 2025. DOI: 10.1016/j.ijforecast.2024.10.002
18. Mutinda JK, Yong L. Deep Learning Applications in Hierarchical Time Series Forecasting. Computational Economics. 2025. DOI: 10.1007/s10614-025-11030-y
19. Mitchell R, Monokroussos G, Nikzad A, Wang W. Hierarchical Demand Forecasting in Retail: A View from the Trenches. SSRN. 2024. DOI: 10.2139/ssrn.5030756

## Probabilistic forecasting and conformal prediction intervals (2024-2026)

20. Wang X, Hyndman R. conformalForecast: Conformal Prediction Methods for Multistep-Ahead Time Series Forecasting. CRAN. 2025. DOI: 10.32614/cran.package.conformalforecast
21. Barber R, Candes E, Xie R. Boosted Conformal Prediction Intervals. NeurIPS. 2024. DOI: 10.52202/079017-2296
22. Chu BXY, Chen CWS. Robust Prediction Intervals for Time Series Forecasting: A Bootstrap and Bayesian Approach. Journal of Forecasting. 2026. DOI: 10.1002/for.70126
23. Hajibabaee P, Pourkamali-Anaraki F, Hariri-Ardebili MA. Adaptive Conformal Prediction Intervals Using Data-Dependent Weights. IEEE Access. 2024. DOI: 10.1109/access.2024.3387858
24. Sousa J, Henriques R. RL-ConFormer: Reinforcement Learning-Enhanced Conformal Prediction with Transformer-Based Hybrid Forecasting for Multi-Horizon Time Series. SSRN. 2026. DOI: 10.2139/ssrn.6263102
25. Chevalier D, Cote MP. From point to probabilistic gradient boosting for claim frequency and severity prediction. European Actuarial Journal. 2025. DOI: 10.1007/s13385-025-00428-5

## Retail demand forecasting (LightGBM, ensembles, holiday and promo effects)

26. Mishra LN, Senapati B, Nayak SK. Explainable Retail Demand Forecasting: A Hybrid LightGBM-SHAP Framework for Inventory Optimization. IEEE Access. 2026. DOI: 10.1109/access.2026.3675540
27. Hewage HC, Perera HN, Bandara K. Enhancing Demand Forecasting in Retail: A Comprehensive Analysis of Sales Promotional Effects on the Entire Demand Life Cycle. Journal of Forecasting. 2025. DOI: 10.1002/for.70039
28. Ye L, Xie N, Boylan JE, Shang Z. Forecasting seasonal demand for retail: A Fourier time-varying grey model. International Journal of Forecasting. 2024. DOI: 10.1016/j.ijforecast.2023.12.006
29. Kim JH, Cho NW. Hybrid Clustering for Retail Demand Forecasting: Combining Rule-Based and Machine Learning Methods. Forecasting. 2026. DOI: 10.3390/forecast8030037
30. Chintapanti A, Maiti S. Optimizing Retail Inventory and Sales Through Advanced Time Series Forecasting. IEEE Access. 2025. DOI: 10.1109/access.2025.3605229
31. Bar K. Counterfactual Demand Forecasting for Retail Promotions Using Bayesian Causal Forests. SSRN. 2025. DOI: 10.2139/ssrn.5981034
32. Jin T. Optimizing Retail Sales Forecasting Through a PSO-Enhanced Ensemble Model Integrating LightGBM, XGBoost, and Deep Neural Networks. International Conference on Digital Society and Intelligent Systems. 2024. DOI: 10.1109/dsins64146.2024.10992072

## M5 / M6 follow-up evidence (competition learnings since 2024)

33. Theodorou E, Spiliotis E, Assimakopoulos V. Forecast accuracy and inventory performance: Insights on their relationship from the M5 competition data. European Journal of Operational Research. 2025. DOI: 10.1016/j.ejor.2024.12.033
34. Makridakis S, Spiliotis E, Hollyman R. The M6 forecasting competition: Bridging the gap between forecasting and investment decisions. International Journal of Forecasting. 2025. DOI: 10.1016/j.ijforecast.2024.11.002
35. Kaltsounis A, Theodorou E, Spiliotis E. Unraveling the effect of engagement and consistency in the results of the M6 forecasting competition. International Journal of Forecasting. 2025. DOI: 10.1016/j.ijforecast.2025.04.002

---

Notes on omissions: candidate hits whose DOI did not resolve via CrossRef (e.g. several SocArXiv preprints and newer 2026 SSRN drafts) were dropped rather than padded. Items 11 and 22 are 2026-dated but were already indexed and resolvable on 2026-05-08.
