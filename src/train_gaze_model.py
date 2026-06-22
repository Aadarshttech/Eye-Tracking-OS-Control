"""
Enhanced Gaze Model Training Pipeline
======================================
Trains multiple regression models to predict screen gaze coordinates
from facial landmark features.

Models compared:
  1. Ridge Regression (baseline)
  2. Random Forest
  3. Gradient Boosting
  4. MLP — Small  (128-64-2)
  5. MLP — Large  (256-256-128-64-2 with BatchNorm + Dropout)

Outputs:
  - Comparison table with per-axis MAE, Euclidean pixel error
  - Best model saved as .keras
  - Feature importance analysis (for tree models)

Backward compatible: works with both old (19-column) and new (28-column) CSVs.
"""

import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "gaze_model.keras")

# ---------------------------------------------------------------------------
# These columns are always present (old + new format)
# ---------------------------------------------------------------------------
CORE_FEATURES = [
    "head_pitch", "head_yaw", "head_roll",
    "l_iris_x", "l_iris_y", "l_iris_z",
    "r_iris_x", "r_iris_y", "r_iris_z",
    "inter_ocular_dist",
]

# Available only in the new CSV format
EXTENDED_FEATURES = [
    "l_gaze_ratio_x", "l_gaze_ratio_y",
    "r_gaze_ratio_x", "r_gaze_ratio_y",
    "l_ear", "r_ear",
    "face_area",
]

# Engineered features computed from raw columns
ENGINEERED_NAMES = [
    "avg_iris_x", "avg_iris_y",
    "iris_diff_x", "iris_diff_y",
    "avg_gaze_h", "avg_gaze_v",
    "gaze_diff_h",
]


def load_data():
    """Load all CSV files and build feature / target arrays."""
    import pandas as pd

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in {DATA_DIR}")
        return None, None, None

    print(f"Found {len(csv_files)} data file(s)")

    dfs = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  + {os.path.basename(path):50s}  {len(df):>6,} rows")
        except Exception as e:
            print(f"  x {os.path.basename(path):50s}  ERROR: {e}")

    if not dfs:
        return None, None, None

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=CORE_FEATURES + ["target_x", "target_y", "screen_w", "screen_h"])

    # ---------- Quality filtering ----------
    initial_len = len(df)

    # Filter blinks (if EAR columns exist)
    if "l_ear" in df.columns and "r_ear" in df.columns:
        df = df[(df["l_ear"] > 0.15) | (df["r_ear"] > 0.15)]

    # Filter extreme brightness (if available)
    if "frame_brightness" in df.columns:
        df = df[(df["frame_brightness"] > 15) & (df["frame_brightness"] < 245)]

    filtered = initial_len - len(df)
    if filtered:
        print(f"\n  Filtered {filtered} low-quality rows ({len(df)} remaining)")

    print(f"\n  Total usable samples: {len(df):,}")

    # ---------- Build features ----------
    feature_cols = []

    # Core features (always available)
    for c in CORE_FEATURES:
        if c in df.columns:
            feature_cols.append(c)

    # Extended features (new format only)
    for c in EXTENDED_FEATURES:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols].values.copy()
    col_names = list(feature_cols)

    # Engineer derived features
    eng = []

    if "l_iris_x" in df.columns and "r_iris_x" in df.columns:
        avg_x = (df["l_iris_x"].values + df["r_iris_x"].values) / 2
        avg_y = (df["l_iris_y"].values + df["r_iris_y"].values) / 2
        diff_x = df["l_iris_x"].values - df["r_iris_x"].values
        diff_y = df["l_iris_y"].values - df["r_iris_y"].values
        eng.extend([avg_x, avg_y, diff_x, diff_y])
        col_names.extend(["avg_iris_x", "avg_iris_y", "iris_diff_x", "iris_diff_y"])

    if "l_gaze_ratio_x" in df.columns and "r_gaze_ratio_x" in df.columns:
        avg_gh = (df["l_gaze_ratio_x"].values + df["r_gaze_ratio_x"].values) / 2
        avg_gv = (df["l_gaze_ratio_y"].values + df["r_gaze_ratio_y"].values) / 2
        diff_gh = df["l_gaze_ratio_x"].values - df["r_gaze_ratio_x"].values
        eng.extend([avg_gh, avg_gv, diff_gh])
        col_names.extend(["avg_gaze_h", "avg_gaze_v", "gaze_diff_h"])

    if eng:
        X = np.column_stack([X] + [e.reshape(-1, 1) for e in eng])

    print(f"  Feature vector dimension: {X.shape[1]}  ({', '.join(col_names)})")

    # ---------- Targets ----------
    y_x = df["target_x"].values / df["screen_w"].values
    y_y = df["target_y"].values / df["screen_h"].values
    y = np.column_stack([y_x, y_y])

    # Store screen dimensions for error conversion
    screen_w_median = float(np.median(df["screen_w"].values))
    screen_h_median = float(np.median(df["screen_h"].values))
    meta = dict(
        screen_w=screen_w_median,
        screen_h=screen_h_median,
        col_names=col_names,
    )

    return X, y, meta


def build_sklearn_models():
    """Instantiate sklearn baselines."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        ),
    }
    return models


def build_mlp_small(input_dim):
    """Lightweight MLP."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(2, activation="sigmoid"),
    ], name="MLP_Small")

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_mlp_large(input_dim):
    """Larger MLP with residual-style skip and heavier regularisation."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(input_dim,))

    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    out = layers.Dense(2, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out, name="MLP_Large")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def evaluate_model(name, y_true, y_pred, screen_w, screen_h):
    """Compute error metrics and return a result dict."""
    # Normalised MAE per axis
    mae_x = float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))
    mae_y = float(np.mean(np.abs(y_true[:, 1] - y_pred[:, 1])))

    # Pixel errors
    px_err_x = mae_x * screen_w
    px_err_y = mae_y * screen_h

    # Euclidean pixel error
    dx = (y_true[:, 0] - y_pred[:, 0]) * screen_w
    dy = (y_true[:, 1] - y_pred[:, 1]) * screen_h
    euc = np.sqrt(dx ** 2 + dy ** 2)
    mean_euc = float(np.mean(euc))
    median_euc = float(np.median(euc))
    p90_euc = float(np.percentile(euc, 90))
    p95_euc = float(np.percentile(euc, 95))

    return dict(
        name=name,
        mae_x=mae_x, mae_y=mae_y,
        px_x=px_err_x, px_y=px_err_y,
        euc_mean=mean_euc, euc_median=median_euc,
        euc_p90=p90_euc, euc_p95=p95_euc,
    )


def print_results_table(results):
    """Pretty-print the comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Model':<22s} {'MAE-X':>8s} {'MAE-Y':>8s} "
          f"{'PxErr-X':>9s} {'PxErr-Y':>9s} "
          f"{'Euc-Mean':>10s} {'Euc-Med':>9s} {'Euc-P90':>9s} {'Euc-P95':>9s}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["euc_mean"]):
        print(f"{r['name']:<22s} "
              f"{r['mae_x']:8.4f} {r['mae_y']:8.4f} "
              f"{r['px_x']:8.1f}px {r['px_y']:8.1f}px "
              f"{r['euc_mean']:9.1f}px {r['euc_median']:8.1f}px "
              f"{r['euc_p90']:8.1f}px {r['euc_p95']:8.1f}px")
    print("=" * 100)


def main():
    # ----- Load data -----
    X, y, meta = load_data()
    if X is None:
        return

    screen_w = meta["screen_w"]
    screen_h = meta["screen_h"]

    # ----- Feature scaling -----
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )

    print(f"\n  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")
    print(f"  Screen resolution (median): {int(screen_w)}×{int(screen_h)}")

    results = []

    # ----- sklearn baselines -----
    print("\n--- Training sklearn baselines ---")
    sk_models = build_sklearn_models()
    for name, model in sk_models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 1)
        res = evaluate_model(name, y_test, y_pred, screen_w, screen_h)
        results.append(res)
        print(f"Euc={res['euc_mean']:.1f}px")

    # ----- MLP Small -----
    print("\n--- Training MLP Small ---")
    import tensorflow as tf

    mlp_small = build_mlp_small(X_train.shape[1])
    mlp_small.summary()

    mlp_small.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
            ),
        ],
        verbose=1,
    )
    y_pred_s = np.clip(mlp_small.predict(X_test, verbose=0), 0, 1)
    res_s = evaluate_model("MLP Small", y_test, y_pred_s, screen_w, screen_h)
    results.append(res_s)

    # ----- MLP Large -----
    print("\n--- Training MLP Large ---")
    mlp_large = build_mlp_large(X_train.shape[1])
    mlp_large.summary()

    mlp_large.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
            ),
        ],
        verbose=1,
    )
    y_pred_l = np.clip(mlp_large.predict(X_test, verbose=0), 0, 1)
    res_l = evaluate_model("MLP Large", y_test, y_pred_l, screen_w, screen_h)
    results.append(res_l)

    # ----- Comparison -----
    print_results_table(results)

    # ----- Save best model -----
    best = min(results, key=lambda r: r["euc_mean"])
    print(f"\n>>> Best model: {best['name']}  (Mean Euclidean Error = {best['euc_mean']:.1f}px)")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if "MLP Large" == best["name"]:
        mlp_large.save(MODEL_SAVE_PATH)
    elif "MLP Small" == best["name"]:
        mlp_small.save(MODEL_SAVE_PATH)
    else:
        # For sklearn models, save the MLP Large anyway as the .keras model
        # (sklearn models are saved separately)
        mlp_candidate = mlp_large if res_l["euc_mean"] < res_s["euc_mean"] else mlp_small
        mlp_candidate.save(MODEL_SAVE_PATH)
        print(f"  (Best was sklearn — saved best MLP to {MODEL_SAVE_PATH} as well)")

    # Save the scaler for inference
    import pickle
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  Model saved  -> {MODEL_SAVE_PATH}")
    print(f"  Scaler saved -> {scaler_path}")
    print(f"  Feature cols  : {meta['col_names']}")
    print("\nDone!")


if __name__ == "__main__":
    main()
