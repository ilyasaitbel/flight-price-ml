"""Measure the model-size / accuracy trade-off across forest configurations.

Usage
-----
    python scripts/model_size_benchmark.py

Each configuration is fitted, scored on the test split and pickled to a
temporary file so the on-disk size of the artefact is known. The shipped
configuration is the largest one that still fits comfortably inside a
GitHub commit and a hosted Streamlit app.
"""

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

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "Clean_Dataset.csv"

CAT = ["airline", "source_city", "departure_time", "stops", "arrival_time",
       "destination_city", "class"]
NUM = ["duration", "days_left"]
FEATURES = CAT + NUM

df = pd.read_csv(DATA)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
X, y = df[FEATURES], df["price"].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
     ("num", "passthrough", NUM)],
    remainder="drop",
)

CONFIGS = [
    # (n_estimators, max_depth, min_samples_leaf, max_features, compress)
    (200, 30, 2, 1.0, 3),   # the original configuration, ~1 GB
    (120, 20, 4, 1.0, 3),   # shipped
    (80, 16, 8, 1.0, 3),
    (60, 14, 12, 1.0, 3),
    (40, 12, 20, 1.0, 3),
]

print(f"{'trees':>6} {'depth':>6} {'leaf':>5} | {'MAE':>9} {'RMSE':>9} {'R2':>8} | {'MB':>7} {'fit s':>6}")
print("-" * 80)

best = None
for n_est, depth, leaf, feat, comp in CONFIGS:
    pipe = Pipeline([
        ("preprocessor", pre),
        ("regressor", RandomForestRegressor(
            n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
            max_features=feat, n_jobs=-1, random_state=42)),
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_s = time.time() - t0

    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    out = BASE / "artifacts" / f"_size_test_{n_est}_{depth}_{leaf}.joblib"
    joblib.dump(pipe, out, compress=comp)
    mb = out.stat().st_size / 1024 / 1024
    out.unlink()

    print(f"{n_est:>6} {depth:>6} {leaf:>5} | {mae:>9,.2f} {rmse:>9,.2f} {r2:>8.5f} | {mb:>7.1f} {fit_s:>6.0f}")

    # Prefer anything under 60 MB; among those take the highest R2.
    if mb < 80 and (best is None or r2 > best[0]):
        best = (r2, (n_est, depth, leaf, feat, comp), mb, mae, rmse)

print()
if best:
    r2, cfg, mb, mae, rmse = best
    print(f"BEST under 80 MB: {cfg} -> {mb:.1f} MB, MAE {mae:,.2f}, RMSE {rmse:,.2f}, R2 {r2:.5f}")
else:
    print("no config under 80 MB")
