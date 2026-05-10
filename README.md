![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Time-series](https://img.shields.io/badge/task-forecasting-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

# Favorita Grocery Store Sales Forecasting

Multivariate time-series forecasting of daily unit sales across 54 Ecuadorian stores and 33 product families.

---

## Task

**Multivariate Time-series Forecasting**

---

## Architecture

```
Sales History + Promotions + Oil Prices → Lag Features → LightGBM RMSLE → Walk-forward Validation
```

---

## Key Features

- Daily unit sales forecast for 1,782 store-family combinations
- Lag and rolling window features (7, 14, 28 day)
- Oil price, holiday, and promotion external regressors
- Chronological walk-forward validation (no future leakage)
- RMSLE evaluation matching Kaggle competition metric

---

## Dataset

[Store Sales — Time Series Forecasting (Kaggle)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/favorita-grocery-sales-forecasting.git
cd favorita-grocery-sales-forecasting
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`LightGBM · statsmodels · pandas · scikit-learn`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
