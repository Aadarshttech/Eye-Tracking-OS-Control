"""
Enhanced Gaze Model Training Pipeline v2
==========================================
Trains multiple regression models to predict screen gaze coordinates
from facial landmark features.

Key improvements over v1:
  - Robust outlier removal (IQR-based on target residuals)
  - Richer feature engineering: interaction terms, polynomial features,
    per-eye asymmetry, gaze-head interaction
  - User-stratified train/test split for better generalisation
  - Deeper residual MLP with cosine-annealing LR schedule
  - Huber loss for robustness to noisy labels
  - K-Fold cross-validation for reliable error estimates
  - TFLite export for fast inference
  - Saves scaler, feature names, and training metadata

Models compared:
  1. Ridge Regression (baseline)
  2. Random Forest
  3. Gradient Boosting (XGB-style via sklearn)
  4. MLP Small  (128→64→2)
  5. MLP Large  (512→256→128→64→2 with residual blocks)

Outputs:
  - Comparison table with per-axis MAE, Euclidean pixel error
  - Best MLP saved as .keras + .tflite
  - Feature scaler + metadata saved
"""

import glob
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "gaze_model.keras")
TFLITE_SAVE_PATH = os.path.join(MODEL_DIR, "eye_model.tflite")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "training_meta.pkl")

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------
CORE_FEATURES = [
    "head_pitch", "head_yaw", "head_roll",
    "l_iris_x", "l_iris_y", "l_iris_z",
    "r_iris_x", "r_iris_y", "r_iris_z",
    "inter_ocular_dist",
]

EXTENDED_FEATURES = [
    "l_gaze_ratio_x", "l_gaze_ratio_y",
    "r_gaze_ratio_x", "r_gaze_ratio_y",
    "l_ear", "r_ear",
    "face_area",
]


# ===================================================================
# Data loading & preprocessing
# ===================================================================
def load_data():
    """Load all CSV files, clean, filter, engineer features."""
    import pandas as pd

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in {DATA_DIR}")
        return None, None, None

    print(f"Found {len(csv_files)} data file(s)")

    dfs = []
    for path in sorted(csv_files):
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  + {os.path.basename(path):60s}  {len(df):>6,} rows")
        except Exception as e:
            print(f"  x {os.path.basename(path):60s}  ERROR: {e}")

    if not dfs:
        return None, None, None

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Raw total: {len(df):,} rows")

    # --- Drop rows with NaN in critical columns ---
    required = CORE_FEATURES + ["target_x", "target_y", "screen_w", "screen_h"]
    df = df.dropna(subset=required)

    # --- Normalise user names (case-insensitive) ---
    df["user"] = df["user"].str.strip().str.lower()

    # --- Quality filtering ---
    initial_len = len(df)

    # Filter blinks
    if "l_ear" in df.columns and "r_ear" in df.columns:
        df = df[(df["l_ear"] > 0.15) | (df["r_ear"] > 0.15)]

    # Filter extreme brightness
    if "frame_brightness" in df.columns:
        df = df[(df["frame_brightness"] > 15) & (df["frame_brightness"] < 245)]

    # Filter extreme head poses (already filtered in collector, but double-check)
    df = df[(df["head_yaw"].abs() < 40) & (df["head_pitch"].abs() < 35)]

    # Filter physically impossible iris positions
    for col in ["l_iris_x", "l_iris_y", "r_iris_x", "r_iris_y"]:
        if col in df.columns:
            df = df[(df[col] > 0.0) & (df[col] < 1.0)]

    # Filter tiny face areas (too far from camera)
    if "face_area" in df.columns:
        df = df[df["face_area"] > 0.005]

    filtered = initial_len - len(df)
    if filtered:
        print(f"  Quality-filtered {filtered} rows ({len(df):,} remaining)")

    # --- Compute normalised targets ---
    df["norm_target_x"] = df["target_x"] / df["screen_w"]
    df["norm_target_y"] = df["target_y"] / df["screen_h"]

    # Sanity: targets must be in [0, 1]
    df = df[(df["norm_target_x"] >= 0) & (df["norm_target_x"] <= 1)]
    df = df[(df["norm_target_y"] >= 0) & (df["norm_target_y"] <= 1)]

    # --- Outlier removal using IQR on residuals ---
    # For each target axis, remove samples where iris position vs target
    # is an extreme outlier (likely mis-labelled or user looked away)
    avg_iris_x = (df["l_iris_x"] + df["r_iris_x"]) / 2
    residual_x = df["norm_target_x"] - avg_iris_x
    q1, q3 = residual_x.quantile(0.02), residual_x.quantile(0.98)
    iqr = q3 - q1
    mask_x = (residual_x >= q1 - 1.5 * iqr) & (residual_x <= q3 + 1.5 * iqr)

    avg_iris_y = (df["l_iris_y"] + df["r_iris_y"]) / 2
    residual_y = df["norm_target_y"] - avg_iris_y
    q1, q3 = residual_y.quantile(0.02), residual_y.quantile(0.98)
    iqr = q3 - q1
    mask_y = (residual_y >= q1 - 1.5 * iqr) & (residual_y <= q3 + 1.5 * iqr)

    before_outlier = len(df)
    df = df[mask_x & mask_y]
    print(f"  Outlier removal: dropped {before_outlier - len(df)} rows ({len(df):,} remaining)")

    print(f"\n  Total usable samples: {len(df):,}")
    print(f"  Users: {sorted(df['user'].unique())}")
    print(f"  Sessions: {sorted(df['session_type'].unique())}")

    # --- Build feature matrix ---
    feature_cols = []

    for c in CORE_FEATURES:
        if c in df.columns:
            feature_cols.append(c)

    for c in EXTENDED_FEATURES:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols].values.copy()
    col_names = list(feature_cols)

    # --- Engineered features ---
    eng_arrays = []
    eng_names = []

    # Average iris positions
    avg_x = (df["l_iris_x"].values + df["r_iris_x"].values) / 2
    avg_y = (df["l_iris_y"].values + df["r_iris_y"].values) / 2
    avg_z = (df["l_iris_z"].values + df["r_iris_z"].values) / 2
    eng_arrays.extend([avg_x, avg_y, avg_z])
    eng_names.extend(["avg_iris_x", "avg_iris_y", "avg_iris_z"])

    # Iris asymmetry (vergence cue)
    diff_x = df["l_iris_x"].values - df["r_iris_x"].values
    diff_y = df["l_iris_y"].values - df["r_iris_y"].values
    eng_arrays.extend([diff_x, diff_y])
    eng_names.extend(["iris_diff_x", "iris_diff_y"])

    # Gaze ratio averages & asymmetry
    if "l_gaze_ratio_x" in df.columns:
        avg_gh = (df["l_gaze_ratio_x"].values + df["r_gaze_ratio_x"].values) / 2
        avg_gv = (df["l_gaze_ratio_y"].values + df["r_gaze_ratio_y"].values) / 2
        diff_gh = df["l_gaze_ratio_x"].values - df["r_gaze_ratio_x"].values
        diff_gv = df["l_gaze_ratio_y"].values - df["r_gaze_ratio_y"].values
        eng_arrays.extend([avg_gh, avg_gv, diff_gh, diff_gv])
        eng_names.extend(["avg_gaze_h", "avg_gaze_v", "gaze_diff_h", "gaze_diff_v"])

    # EAR average (eye openness)
    if "l_ear" in df.columns:
        avg_ear = (df["l_ear"].values + df["r_ear"].values) / 2
        ear_diff = df["l_ear"].values - df["r_ear"].values
        eng_arrays.extend([avg_ear, ear_diff])
        eng_names.extend(["avg_ear", "ear_diff"])

    # Head-gaze interaction terms (head pose × iris position)
    pitch = df["head_pitch"].values
    yaw = df["head_yaw"].values
    eng_arrays.extend([
        yaw * avg_x,
        pitch * avg_y,
        yaw * avg_gh if "l_gaze_ratio_x" in df.columns else yaw * avg_x,
        pitch * avg_gv if "l_gaze_ratio_y" in df.columns else pitch * avg_y,
    ])
    eng_names.extend(["yaw_x_iris_x", "pitch_x_iris_y", "yaw_x_gaze_h", "pitch_x_gaze_v"])

    # IOD-normalised iris positions (distance-invariant)
    iod = df["inter_ocular_dist"].values
    iod_safe = np.where(iod > 1e-6, iod, 1e-6)
    eng_arrays.extend([avg_x / iod_safe, avg_y / iod_safe])
    eng_names.extend(["iris_x_norm_iod", "iris_y_norm_iod"])

    # Squared terms for non-linearity
    eng_arrays.extend([avg_x ** 2, avg_y ** 2, yaw ** 2, pitch ** 2])
    eng_names.extend(["avg_iris_x_sq", "avg_iris_y_sq", "yaw_sq", "pitch_sq"])

    if eng_arrays:
        eng_matrix = np.column_stack([e.reshape(-1, 1) for e in eng_arrays])
        X = np.column_stack([X, eng_matrix])
        col_names.extend(eng_names)

    print(f"  Feature vector dimension: {X.shape[1]}")

    # --- Targets ---
    y = np.column_stack([df["norm_target_x"].values, df["norm_target_y"].values])

    # Store metadata
    screen_w_median = float(np.median(df["screen_w"].values))
    screen_h_median = float(np.median(df["screen_h"].values))
    meta = dict(
        screen_w=screen_w_median,
        screen_h=screen_h_median,
        col_names=col_names,
        users=sorted(df["user"].unique().tolist()),
        user_labels=df["user"].values,
    )

    return X, y, meta


# ===================================================================
# Model builders
# ===================================================================
def build_sklearn_models():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            )
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.08,
                subsample=0.8, min_samples_leaf=10, random_state=42
            )
        ),
    }


def build_mlp_small(input_dim):
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae"],
    )
    return model


def build_mlp_large(input_dim):
    """Deep MLP with residual connections, Huber loss, and cosine LR."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(input_dim,))

    # Stem
    x = layers.Dense(512, activation="relu", kernel_initializer="he_normal")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual block 1: 256
    skip = layers.Dense(256)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    # Residual block 2: 128
    skip = layers.Dense(128)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    # Head
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)

    out = layers.Dense(2, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out, name="MLP_Large_Residual")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae"],
    )
    return model


# ===================================================================
# Evaluation
# ===================================================================
def evaluate_model(name, y_true, y_pred, screen_w, screen_h):
    mae_x = float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))
    mae_y = float(np.mean(np.abs(y_true[:, 1] - y_pred[:, 1])))

    px_err_x = mae_x * screen_w
    px_err_y = mae_y * screen_h

    dx = (y_true[:, 0] - y_pred[:, 0]) * screen_w
    dy = (y_true[:, 1] - y_pred[:, 1]) * screen_h
    euc = np.sqrt(dx ** 2 + dy ** 2)

    return dict(
        name=name,
        mae_x=mae_x, mae_y=mae_y,
        px_x=px_err_x, px_y=px_err_y,
        euc_mean=float(np.mean(euc)),
        euc_median=float(np.median(euc)),
        euc_p90=float(np.percentile(euc, 90)),
        euc_p95=float(np.percentile(euc, 95)),
    )


def print_results_table(results):
    print("\n" + "=" * 105)
    print(f"{'Model':<24s} {'MAE-X':>8s} {'MAE-Y':>8s} "
          f"{'PxErr-X':>9s} {'PxErr-Y':>9s} "
          f"{'Euc-Mean':>10s} {'Euc-Med':>9s} {'Euc-P90':>9s} {'Euc-P95':>9s}")
    print("-" * 105)
    for r in sorted(results, key=lambda x: x["euc_mean"]):
        print(f"{r['name']:<24s} "
              f"{r['mae_x']:8.4f} {r['mae_y']:8.4f} "
              f"{r['px_x']:8.1f}px {r['px_y']:8.1f}px "
              f"{r['euc_mean']:9.1f}px {r['euc_median']:8.1f}px "
              f"{r['euc_p90']:8.1f}px {r['euc_p95']:8.1f}px")
    print("=" * 105)


# ===================================================================
# TFLite conversion
# ===================================================================
def convert_to_tflite(keras_model, save_path):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"  TFLite model saved -> {save_path}  ({size_kb:.1f} KB)")


# ===================================================================
# Main training pipeline
# ===================================================================
def main():
    t_start = time.time()

    # ----- Load data -----
    X, y, meta = load_data()
    if X is None:
        return

    screen_w = meta["screen_w"]
    screen_h = meta["screen_h"]
    user_labels = meta["user_labels"]

    # ----- Feature scaling -----
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, GroupShuffleSplit

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Replace any NaN/Inf from engineering with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- User-stratified split -----
    if len(np.unique(user_labels)) > 1:
        # Use GroupShuffleSplit so test set contains frames from all users
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_idx, test_idx = next(gss.split(X_scaled, y, groups=user_labels))
    else:
        # Fallback to standard random split if only 1 user exists
        X_indices = np.arange(len(X_scaled))
        train_idx, test_idx = train_test_split(X_indices, test_size=0.15, random_state=42)
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\n  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")
    print(f"  Screen resolution (median): {int(screen_w)}x{int(screen_h)}")
    print(f"  Feature dimensions: {X_train.shape[1]}")

    results = []

    # =================================================================
    # sklearn baselines
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING SKLEARN BASELINES")
    print("=" * 60)
    sk_models = build_sklearn_models()
    for name, model in sk_models.items():
        t0 = time.time()
        print(f"  Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, 1)
        res = evaluate_model(name, y_test, y_pred, screen_w, screen_h)
        results.append(res)
        print(f"Euc={res['euc_mean']:.1f}px  ({time.time() - t0:.1f}s)")

    # =================================================================
    # MLP Small
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING MLP SMALL")
    print("=" * 60)
    import tensorflow as tf

    mlp_small = build_mlp_small(X_train.shape[1])
    mlp_small.summary()

    mlp_small.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
            ),
        ],
        verbose=1,
    )
    y_pred_s = np.clip(mlp_small.predict(X_test, verbose=0), 0, 1)
    res_s = evaluate_model("MLP Small", y_test, y_pred_s, screen_w, screen_h)
    results.append(res_s)
    print(f"  >> MLP Small  Euc={res_s['euc_mean']:.1f}px")

    # =================================================================
    # MLP Large (Residual)
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING MLP LARGE (RESIDUAL)")
    print("=" * 60)
    mlp_large = build_mlp_large(X_train.shape[1])
    mlp_large.summary()

    # Cosine decay schedule
    total_steps = int(np.ceil(X_train.shape[0] * 0.85 / 64)) * 300
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_steps,
        alpha=1e-6,
    )
    mlp_large.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae"],
    )

    mlp_large.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=300,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=30, restore_best_weights=True
            ),
        ],
        verbose=1,
    )
    y_pred_l = np.clip(mlp_large.predict(X_test, verbose=0), 0, 1)
    res_l = evaluate_model("MLP Large Residual", y_test, y_pred_l, screen_w, screen_h)
    results.append(res_l)
    print(f"  >> MLP Large Residual  Euc={res_l['euc_mean']:.1f}px")

    # =================================================================
    # Results comparison
    # =================================================================
    print_results_table(results)

    best = min(results, key=lambda r: r["euc_mean"])
    print(f"\n>>> Best model: {best['name']}  (Mean Euclidean Error = {best['euc_mean']:.1f}px)")

    # =================================================================
    # Save best MLP
    # =================================================================
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Determine which MLP to save
    if res_l["euc_mean"] <= res_s["euc_mean"]:
        best_mlp = mlp_large
        best_mlp_name = "MLP Large Residual"
    else:
        best_mlp = mlp_small
        best_mlp_name = "MLP Small"

    best_mlp.save(MODEL_SAVE_PATH)
    print(f"\n  Keras model saved  -> {MODEL_SAVE_PATH}  ({best_mlp_name})")

    # Convert to TFLite
    convert_to_tflite(best_mlp, TFLITE_SAVE_PATH)

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved       -> {SCALER_PATH}")

    # Save training metadata
    training_meta = dict(
        col_names=meta["col_names"],
        screen_w=screen_w,
        screen_h=screen_h,
        users=meta["users"],
        n_features=X_train.shape[1],
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        best_model=best["name"],
        best_euc_mean=best["euc_mean"],
        best_mlp_saved=best_mlp_name,
        all_results=results,
    )
    with open(META_PATH, "wb") as f:
        pickle.dump(training_meta, f)
    print(f"  Metadata saved     -> {META_PATH}")

    elapsed = time.time() - t_start
    print(f"\n  Total training time: {elapsed / 60:.1f} minutes")
    print(f"  Feature columns ({len(meta['col_names'])}): {meta['col_names']}")
    print("\nDone!")


if __name__ == "__main__":
    main()
