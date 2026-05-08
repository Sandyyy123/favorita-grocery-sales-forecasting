"""Advanced forecasting model for Favorita store sales (project 09).

Single LightGBM regressor trained on the full panel of 1,782 store-family
series. Features:

  - Lag features: sales_lag_{1, 7, 14, 28} (per store-family).
  - Rolling means: sales_roll_{7, 28}_mean (lagged by 1 day to avoid leak).
  - Rolling std: sales_roll_28_std.
  - Calendar: dayofweek, day, month, weekofyear, is_weekend, is_month_end.
  - Holiday flags: national_holiday, regional_holiday, local_holiday.
  - Oil price (forward-filled).
  - Promotions: onpromotion (raw + same-store-family lag_7).
  - Categorical: store_nbr, family, store_type, store_cluster, city, state.
  - Earthquake-week flag (16 Apr 2016 to 15 May 2016).

Two heads are trained:
  - Point forecast: LightGBM with regression objective on log1p(sales).
  - Quantile forecasts: LightGBM with quantile objective at tau in {0.1, 0.5, 0.9}
    for prediction intervals.

Outputs:
  - deliverables/advanced_predictions.csv  (sample_submission shape with q10/q50/q90)
  - deliverables/advanced_metrics.json     (RMSLE, pinball loss per quantile)
  - deliverables/advanced_model.pkl        (LightGBM Booster, point forecast)
  - deliverables/advanced_feature_importance.csv

NOT executed at scaffold time. Run from the project root with:
    python src/model_advanced.py
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DELIV_DIR = PROJECT_ROOT / "deliverables"
DELIV_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 16
VALIDATION_DAYS = 16
LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 28]
QUANTILES = [0.1, 0.5, 0.9]
EARTHQUAKE_START = pd.Timestamp("2016-04-16")
EARTHQUAKE_END = pd.Timestamp("2016-05-15")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all() -> dict[str, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(
        DATA_DIR / "holidays_events.csv", parse_dates=["date"]
    )
    return {
        "train": train,
        "test": test,
        "stores": stores,
        "oil": oil,
        "holidays": holidays,
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_oil_series(oil: pd.DataFrame) -> pd.DataFrame:
    oil = oil.set_index("date").sort_index()
    full = pd.date_range(oil.index.min(), oil.index.max(), freq="D")
    oil = oil.reindex(full)
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    return oil.reset_index().rename(columns={"index": "date"})


def build_holiday_flags(holidays: pd.DataFrame) -> pd.DataFrame:
    """Emit per-date holiday booleans at national, regional, local level."""
    eff = holidays[~holidays["transferred"]].copy()
    eff = eff[eff["type"].isin(["Holiday", "Transfer", "Bridge", "Additional"])]
    flags = pd.DataFrame({"date": eff["date"].unique()})
    flags["national_holiday"] = flags["date"].isin(
        eff.loc[eff["locale"] == "National", "date"]
    ).astype(int)
    flags["regional_holiday"] = flags["date"].isin(
        eff.loc[eff["locale"] == "Regional", "date"]
    ).astype(int)
    flags["local_holiday"] = flags["date"].isin(
        eff.loc[eff["locale"] == "Local", "date"]
    ).astype(int)
    return flags


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_post_quake"] = (
        (df["date"] >= EARTHQUAKE_START) & (df["date"] <= EARTHQUAKE_END)
    ).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features per store-family group.

    Sorted by date inside each group; rolling stats are shifted by 1 day to
    avoid using same-day target.
    """
    df = df.sort_values(["store_nbr", "family", "date"]).copy()
    grp = df.groupby(["store_nbr", "family"], observed=True)["sales"]
    for lag in LAGS:
        df[f"sales_lag_{lag}"] = grp.shift(lag)
    for w in ROLL_WINDOWS:
        df[f"sales_roll_{w}_mean"] = grp.shift(1).rolling(w, min_periods=1).mean()
        df[f"sales_roll_{w}_std"] = grp.shift(1).rolling(w, min_periods=2).std()
    grp_promo = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"]
    df["onpromotion_lag_7"] = grp_promo.shift(7)
    return df


def build_master_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    train = data["train"].copy()
    test = data["test"].copy()
    test["sales"] = np.nan

    full = pd.concat([train, test], ignore_index=True, sort=False)
    full = full.merge(data["stores"], on="store_nbr", how="left")
    full = full.merge(build_oil_series(data["oil"]), on="date", how="left")
    full = full.merge(build_holiday_flags(data["holidays"]), on="date", how="left")
    full[["national_holiday", "regional_holiday", "local_holiday"]] = (
        full[["national_holiday", "regional_holiday", "local_holiday"]].fillna(0).astype(int)
    )
    full["dcoilwtico"] = full["dcoilwtico"].ffill().bfill()

    full = add_calendar_features(full)
    full = add_lag_features(full)

    for col in ["family", "city", "state", "type"]:
        full[col] = full[col].astype("category")

    return full


FEATURE_COLS = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
    "onpromotion",
    "onpromotion_lag_7",
    "dcoilwtico",
    "national_holiday",
    "regional_holiday",
    "local_holiday",
    "dayofweek",
    "day",
    "month",
    "weekofyear",
    "is_weekend",
    "is_month_end",
    "is_month_start",
    "is_post_quake",
] + [f"sales_lag_{l}" for l in LAGS] + [
    f"sales_roll_{w}_mean" for w in ROLL_WINDOWS
] + [f"sales_roll_{w}_std" for w in ROLL_WINDOWS]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    objective: str = "regression",
    alpha: float | None = None,
):
    """Train a single LightGBM model. Uses regression on log1p(target)
    for the point head, and quantile objective for the probabilistic heads."""
    import lightgbm as lgb

    params = {
        "objective": objective,
        "metric": "rmse" if objective == "regression" else "quantile",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 100,
        "verbose": -1,
    }
    if objective == "quantile" and alpha is not None:
        params["alpha"] = alpha

    target_train = np.log1p(y_train.values) if objective == "regression" else y_train.values
    target_val = np.log1p(y_val.values) if objective == "regression" else y_val.values

    cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]
    dtrain = lgb.Dataset(X_train, target_train, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_val, target_val, categorical_feature=cat_cols, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def run_advanced(max_rows: int | None = None) -> dict:
    data = load_all()
    full = build_master_table(data)

    cutoff_val = full["date"].max() - pd.Timedelta(days=VALIDATION_DAYS + HORIZON)
    cutoff_train_end = full["date"].max() - pd.Timedelta(days=HORIZON)

    is_train = (full["sales"].notna()) & (full["date"] <= cutoff_val)
    is_val = (full["sales"].notna()) & (full["date"] > cutoff_val) & (full["date"] <= cutoff_train_end)
    is_test = full["sales"].isna()

    if max_rows is not None:
        train_df = full[is_train].sample(min(max_rows, is_train.sum()), random_state=42)
    else:
        train_df = full[is_train]
    val_df = full[is_val]
    test_df = full[is_test]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["sales"].clip(lower=0)
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["sales"].clip(lower=0)
    X_test = test_df[FEATURE_COLS]

    print(f"Train rows: {len(X_train):,}  Val rows: {len(X_val):,}  Test rows: {len(X_test):,}")

    # ------------------------------------------------------------------
    # Point forecast head
    # ------------------------------------------------------------------
    print("Training point-forecast LightGBM (log1p target) ...")
    point_model = train_lightgbm(X_train, y_train, X_val, y_val, objective="regression")
    val_pred_log = point_model.predict(X_val, num_iteration=point_model.best_iteration)
    val_pred = np.clip(np.expm1(val_pred_log), 0, None)
    val_rmsle = rmsle(y_val.values, val_pred)
    print(f"Validation RMSLE (point): {val_rmsle:.4f}")

    test_pred_log = point_model.predict(X_test, num_iteration=point_model.best_iteration)
    test_pred = np.clip(np.expm1(test_pred_log), 0, None)

    importance = pd.DataFrame(
        {
            "feature": point_model.feature_name(),
            "gain": point_model.feature_importance(importance_type="gain"),
            "split": point_model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(DELIV_DIR / "advanced_feature_importance.csv", index=False)

    with open(DELIV_DIR / "advanced_model.pkl", "wb") as f:
        pickle.dump(point_model, f)

    # ------------------------------------------------------------------
    # Quantile heads for prediction intervals
    # ------------------------------------------------------------------
    quantile_metrics = {}
    test_quantiles = {}
    for tau in QUANTILES:
        print(f"Training quantile LightGBM (tau={tau}) ...")
        q_model = train_lightgbm(
            X_train, y_train, X_val, y_val, objective="quantile", alpha=tau
        )
        q_val = np.clip(q_model.predict(X_val, num_iteration=q_model.best_iteration), 0, None)
        q_test = np.clip(q_model.predict(X_test, num_iteration=q_model.best_iteration), 0, None)
        pl = pinball_loss(y_val.values, q_val, tau)
        quantile_metrics[f"pinball_{int(tau*100):02d}"] = pl
        test_quantiles[f"q{int(tau*100):02d}"] = q_test
        print(f"  pinball loss tau={tau}: {pl:.4f}")

    # ------------------------------------------------------------------
    # Save predictions in submission shape
    # ------------------------------------------------------------------
    submission = test_df[["id", "date", "store_nbr", "family"]].copy()
    submission["sales"] = test_pred
    for k, v in test_quantiles.items():
        submission[k] = v
    submission.to_csv(DELIV_DIR / "advanced_predictions.csv", index=False)

    metrics = {
        "n_train_rows": int(len(X_train)),
        "n_val_rows": int(len(X_val)),
        "n_test_rows": int(len(X_test)),
        "validation_rmsle_point": val_rmsle,
        "quantile_pinball": quantile_metrics,
        "best_iteration_point": int(point_model.best_iteration or 0),
        "feature_count": len(FEATURE_COLS),
    }
    with open(DELIV_DIR / "advanced_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved advanced_metrics.json")
    return metrics


if __name__ == "__main__":
    metrics = run_advanced(max_rows=None)
    print(json.dumps(metrics, indent=2))
