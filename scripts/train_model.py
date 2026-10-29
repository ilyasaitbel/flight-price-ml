"""
Train the SkyPrice flight-price regressor and regenerate the deployable artifacts.

Usage
-----
    python scripts/train_model.py

Outputs
-------
    artifacts/flight_price_pipeline.joblib   fitted sklearn Pipeline (model + preprocessing)
    artifacts/flight_price_model.meta.json   feature contract, options and evaluation metrics

The pipeline bundles preprocessing and the estimator in a single object, so the
Streamlit app only has to build a DataFrame with the exact feature names and
call ``pipeline.predict(df)``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Clean_Dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "flight_price_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "flight_price_model.meta.json"

# =========================================================
# FEATURE CONTRACT
#
# The raw dataset also carries a "flight" column (1561 distinct flight
# numbers). It is deliberately excluded: it is a high-cardinality identifier
# that would explode the one-hot encoding without generalising to unseen
# flights.
# =========================================================

TARGET = "price"

CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
]

NUMERICAL_FEATURES = [
    "duration",
    "days_left",
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = 30
MIN_SAMPLES_LEAF = 2
N_BOOTSTRAP = 200


def bootstrap_mae_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 200, seed: int = RANDOM_STATE):
    """Percentile bootstrap confidence interval for the mean absolute error."""
    rng = np.random.default_rng(seed)
    errors = np.abs(y_true - y_pred)
    n = len(errors)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(errors, size=n, replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    print("=" * 60)
    print("SkyPrice - training flight price regressor")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD
    # ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {DATA_PATH.name}")

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    # ---------------------------------------------------------
    # 2. SPLIT
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train {len(X_train):,} | Test {len(X_test):,}")

    # ---------------------------------------------------------
    # 3. PIPELINE
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )

    regressor = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        bootstrap=True,
        max_samples=None,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    # ---------------------------------------------------------
    # 4. TRAIN
    # ---------------------------------------------------------
    print(f"Training RandomForestRegressor (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}) ...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")

    # ---------------------------------------------------------
    # 5. EVALUATE
    # ---------------------------------------------------------
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mae_lo, mae_hi = bootstrap_mae_ci(y_test.values, y_pred)

    print("-" * 60)
    print(f"MAE   {mae:,.2f}   CI95 [{mae_lo:,.2f}, {mae_hi:,.2f}]")
    print(f"RMSE  {rmse:,.2f}")
    print(f"R2    {r2:.6f}")
    print("-" * 60)

    # ---------------------------------------------------------
    # 6. FEATURE IMPORTANCE
    # ---------------------------------------------------------
    encoded = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["regressor"].feature_importances_
    order = np.argsort(importances)[::-1]
    print("Top feature contributions:")
    for idx in order[:12]:
        print(f"   {encoded[idx]:<28} {importances[idx] * 100:6.2f}%")

    # ---------------------------------------------------------
    # 7. PERSIST
    # ---------------------------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved model  -> {MODEL_PATH.relative_to(BASE_DIR)} ({size_mb:.1f} MB)")

    metadata = {
        "model_name": "Random Forest",
        "target": TARGET,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAE_CI_95_lower": mae_lo,
            "MAE_CI_95_upper": mae_hi,
        },
        "categorical_options": {
            col: sorted(df[col].astype(str).unique().tolist())
            for col in CATEGORICAL_FEATURES
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=4, ensure_ascii=False)
        fh.write("\n")

    print(f"Saved metadata -> {METADATA_PATH.relative_to(BASE_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
