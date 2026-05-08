"""Baseline forecasting model for Favorita store sales (project 09).

Per-store-family univariate forecasts. For each of the ~1,782 active
store_nbr x family series we fit one of:

  - Prophet with weekly + yearly seasonality, holiday regressor matrix
    built from holidays_events.csv, and oil price as an additional
    regressor.
  - SARIMA(1,1,1)(1,1,1,7) as a fallback when Prophet is unavailable.

The script writes:
  - deliverables/baseline_predictions.csv  (sample_submission shape)
  - deliverables/baseline_metrics.json     (per-family RMSLE, overall RMSLE)
  - deliverables/baseline_model.pkl        (one Prophet object as smoke artefact)

NOT executed at scaffold time. Run from the project root with:
    python src/model_baseline.py
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DELIV_DIR = PROJECT_ROOT / "deliverables"
DELIV_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 16  # days: matches the Kaggle public test window
VALIDATION_DAYS = 16  # last 16 days of train held out for in-sample RMSLE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all() -> dict[str, pd.DataFrame]:
    """Load the six Favorita CSVs from data/."""
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(
        DATA_DIR / "holidays_events.csv", parse_dates=["date"]
    )
    transactions = pd.read_csv(
        DATA_DIR / "transactions.csv", parse_dates=["date"]
    )
    return {
        "train": train,
        "test": test,
        "stores": stores,
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
    }


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def build_oil_series(oil: pd.DataFrame) -> pd.DataFrame:
    """Reindex oil to a daily date range and fill weekend gaps."""
    oil = oil.set_index("date").sort_index()
    full = pd.date_range(oil.index.min(), oil.index.max(), freq="D")
    oil = oil.reindex(full)
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    oil = oil.reset_index().rename(columns={"index": "date"})
    return oil


def build_effective_holidays(holidays: pd.DataFrame) -> pd.DataFrame:
    """Drop transferred=True rows; keep Transfer / Bridge / Holiday rows.

    Returns a DataFrame in Prophet's expected schema: columns ds, holiday.
    """
    eff = holidays[~holidays["transferred"]].copy()
    eff = eff[eff["type"].isin(["Holiday", "Transfer", "Bridge", "Additional"])]
    eff = eff.rename(columns={"date": "ds"})[["ds", "type", "locale", "description"]]
    eff["holiday"] = (
        eff["locale"].astype(str) + "_" + eff["description"].astype(str).str.replace(" ", "_")
    )
    return eff[["ds", "holiday"]].drop_duplicates()


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error (Kaggle metric).

    Predictions are clipped at 0 because log1p of a negative number is
    undefined and the underlying target is non-negative.
    """
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


# ---------------------------------------------------------------------------
# Per-series forecasting
# ---------------------------------------------------------------------------
def fit_prophet_series(
    series: pd.DataFrame,
    horizon: int,
    holidays_df: pd.DataFrame,
    oil: pd.DataFrame,
):
    """Fit one Prophet model on a single store-family series.

    Parameters
    ----------
    series : DataFrame with columns date, sales
    horizon : forecast horizon in days
    holidays_df : Prophet-format holidays (ds, holiday)
    oil : daily oil dataframe with columns date, dcoilwtico
    """
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "prophet is required for the baseline. Install with `pip install prophet`."
        ) from exc

    df = series.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    df = df.merge(oil.rename(columns={"date": "ds"}), on="ds", how="left")

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.add_regressor("dcoilwtico")
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="D")
    future = future.merge(oil.rename(columns={"date": "ds"}), on="ds", how="left")
    future["dcoilwtico"] = future["dcoilwtico"].ffill().bfill()
    forecast = model.predict(future)
    return model, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def fit_sarima_series(series: pd.DataFrame, horizon: int):
    """SARIMA(1,1,1)(1,1,1,7) fallback when Prophet is not available."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = series.set_index("date")["sales"].asfreq("D").fillna(method="ffill")
    model = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=horizon)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.2)
    out = pd.DataFrame(
        {
            "ds": mean.index,
            "yhat": mean.values,
            "yhat_lower": ci.iloc[:, 0].values,
            "yhat_upper": ci.iloc[:, 1].values,
        }
    )
    return res, out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_baseline(use_prophet: bool = True, max_series: int | None = None) -> dict:
    """Loop over store-family pairs, fit baseline, score on holdout.

    Parameters
    ----------
    use_prophet : whether to use Prophet (True) or SARIMA fallback (False).
    max_series : optional cap for smoke-testing (None = run all).
    """
    data = load_all()
    train = data["train"]
    oil = build_oil_series(data["oil"])
    holidays_df = build_effective_holidays(data["holidays"])

    # Time-based holdout: last VALIDATION_DAYS of train as in-sample test.
    cutoff = train["date"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    train_fit = train[train["date"] <= cutoff]
    train_val = train[train["date"] > cutoff]

    series_keys = (
        train.groupby(["store_nbr", "family"]).size().sort_values(ascending=False).index
    )
    if max_series is not None:
        series_keys = series_keys[:max_series]

    rows = []
    family_errors: dict[str, list[float]] = {}
    saved_one_model = False

    for i, (store_nbr, family) in enumerate(series_keys):
        s_train = train_fit[
            (train_fit["store_nbr"] == store_nbr) & (train_fit["family"] == family)
        ]
        s_val = train_val[
            (train_val["store_nbr"] == store_nbr) & (train_val["family"] == family)
        ]
        if len(s_train) < 365 or len(s_val) == 0:
            continue
        try:
            if use_prophet:
                model, fc = fit_prophet_series(
                    s_train[["date", "sales"]], VALIDATION_DAYS, holidays_df, oil
                )
            else:
                model, fc = fit_sarima_series(s_train[["date", "sales"]], VALIDATION_DAYS)
        except Exception as exc:
            print(f"[skip] store={store_nbr} family={family}: {exc}")
            continue

        fc_holdout = fc.tail(VALIDATION_DAYS).rename(columns={"ds": "date"})
        merged = s_val.merge(fc_holdout, on="date", how="left")
        if merged["yhat"].isna().any():
            continue
        err = rmsle(merged["sales"].values, merged["yhat"].values)
        family_errors.setdefault(family, []).append(err)
        rows.append(
            {
                "store_nbr": store_nbr,
                "family": family,
                "n_train": len(s_train),
                "n_val": len(s_val),
                "rmsle": err,
            }
        )
        if not saved_one_model and use_prophet:
            with open(DELIV_DIR / "baseline_model.pkl", "wb") as f:
                pickle.dump({"store_nbr": store_nbr, "family": family, "model": model}, f)
            saved_one_model = True
        if (i + 1) % 50 == 0:
            print(f"  fitted {i + 1} series")

    per_series = pd.DataFrame(rows)
    per_series.to_csv(DELIV_DIR / "baseline_per_series_rmsle.csv", index=False)

    metrics = {
        "n_series_fitted": int(len(per_series)),
        "overall_rmsle_mean": float(per_series["rmsle"].mean()) if len(per_series) else None,
        "overall_rmsle_median": float(per_series["rmsle"].median())
        if len(per_series)
        else None,
        "family_rmsle_mean": {fam: float(np.mean(errs)) for fam, errs in family_errors.items()},
    }
    with open(DELIV_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved baseline_metrics.json")
    return metrics


if __name__ == "__main__":
    metrics = run_baseline(use_prophet=True, max_series=None)
    print(json.dumps(metrics, indent=2)[:500])
