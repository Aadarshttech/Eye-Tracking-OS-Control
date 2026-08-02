"""
Enhanced Gaze Model Training Pipeline v3
==========================================
Extends v2 with additional model architectures and techniques
for improved accuracy.

New models added:
  1. Conv1D Network (treats features as 1D signal)
  2. Attention MLP (learnable feature weighting)
  3. XGBoost (if available, fallback to sklearn GB)
  4. Separate X/Y MLPs (independent axis prediction)
  5. Ensemble (weighted average of top models)
  6. MLP Large v2 (tuned hyperparams: wider stem, lower dropout)

Also adds:
  - Gaussian noise data augmentation during training
  - Hyperparameter-tuned MLP variant

Outputs:
  - Full comparison table with all models
  - Best single model saved as .keras + .tflite
  - Ensemble predictions saved for comparison
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

# Also save individual model variants
MODEL_SAVE_DIR = os.path.join(MODEL_DIR, "variants")

# ---------------------------------------------------------------------------
# Feature column definitions (same as v2)
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
# Data loading & preprocessing (same as v2)
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

    # Filter extreme head poses
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

    # Head-gaze interaction terms
    pitch = df["head_pitch"].values
    yaw = df["head_yaw"].values
    eng_arrays.extend([
        yaw * avg_x,
        pitch * avg_y,
        yaw * avg_gh if "l_gaze_ratio_x" in df.columns else yaw * avg_x,
        pitch * avg_gv if "l_gaze_ratio_y" in df.columns else pitch * avg_y,
    ])
    eng_names.extend(["yaw_x_iris_x", "pitch_x_iris_y", "yaw_x_gaze_h", "pitch_x_gaze_v"])

    # IOD-normalised iris positions
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
# Data augmentation
# ===================================================================
def augment_with_noise(X_train, y_train, noise_std=0.02, n_copies=2):
    """Add Gaussian noise copies to training data for regularisation."""
    aug_X = [X_train]
    aug_y = [y_train]
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_std, X_train.shape)
        aug_X.append(X_train + noise)
        aug_y.append(y_train)  # targets stay the same
    return np.vstack(aug_X), np.vstack(aug_y)


# ===================================================================
# Model builders — ORIGINAL (from v2)
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
    """Deep MLP with residual connections (original v2)."""
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
# NEW Model builders — v3 additions
# ===================================================================
def build_mlp_large_v2(input_dim):
    """Tuned MLP: wider stem (768), lower dropout, smaller Huber delta."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(input_dim,))

    # Wider stem
    x = layers.Dense(768, activation="relu", kernel_initializer="he_normal")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    # Residual block 1: 384
    skip = layers.Dense(384)(x)
    x = layers.Dense(384, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(384, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    # Residual block 2: 192
    skip = layers.Dense(192)(x)
    x = layers.Dense(192, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(192, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    # Residual block 3: 96
    skip = layers.Dense(96)(x)
    x = layers.Dense(96, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(96, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)

    # Head
    x = layers.Dense(48, activation="relu", kernel_initializer="he_normal")(x)
    out = layers.Dense(2, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out, name="MLP_Large_v2_Tuned")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.Huber(delta=0.01),
        metrics=["mae"],
    )
    return model


def build_conv1d_model(input_dim):
    """1D Convolutional model — treats feature vector as a 1D signal."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(2, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out, name="Conv1D_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae"],
    )
    return model


def build_attention_mlp(input_dim):
    """MLP with learnable feature attention mechanism."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(input_dim,))

    # Attention gate — learns which features matter most per sample
    attn_weights = layers.Dense(input_dim, activation="softmax", name="attention_gate")(inp)
    attended = layers.Multiply()([inp, attn_weights])

    # Concatenate original + attended for richer representation
    x = layers.Concatenate()([inp, attended])

    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    out = layers.Dense(2, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out, name="Attention_MLP")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae"],
    )
    return model


def build_separate_xy_models(input_dim):
    """Two independent models — one for X, one for Y."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    def _single_axis_model(name):
        inp = layers.Input(shape=(input_dim,))
        x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m = models.Model(inputs=inp, outputs=out, name=name)
        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss=tf.keras.losses.Huber(delta=0.03),
            metrics=["mae"],
        )
        return m

    return _single_axis_model("MLP_X_Only"), _single_axis_model("MLP_Y_Only")


def build_xgboost_models():
    """Try real XGBoost; fall back to sklearn GradientBoosting if not installed."""
    from sklearn.multioutput import MultiOutputRegressor

    try:
        import xgboost as xgb
        print("  Using real XGBoost library")
        return {
            "XGBoost": MultiOutputRegressor(
                xgb.XGBRegressor(
                    n_estimators=500, max_depth=10, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0,
                    n_jobs=-1, random_state=42, verbosity=0,
                )
            )
        }
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        print("  XGBoost not installed, using sklearn GradientBoosting (tuned)")
        return {
            "GradientBoosting Tuned": MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=500, max_depth=10, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=5, random_state=42,
                )
            )
        }


# ===================================================================
# Evaluation (same as v2)
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
    print("\n" + "=" * 110)
    print(f"{'Model':<28s} {'MAE-X':>8s} {'MAE-Y':>8s} "
          f"{'PxErr-X':>9s} {'PxErr-Y':>9s} "
          f"{'Euc-Mean':>10s} {'Euc-Med':>9s} {'Euc-P90':>9s} {'Euc-P95':>9s}")
    print("-" * 110)
    for r in sorted(results, key=lambda x: x["euc_mean"]):
        print(f"{r['name']:<28s} "
              f"{r['mae_x']:8.4f} {r['mae_y']:8.4f} "
              f"{r['px_x']:8.1f}px {r['px_y']:8.1f}px "
              f"{r['euc_mean']:9.1f}px {r['euc_median']:8.1f}px "
              f"{r['euc_p90']:8.1f}px {r['euc_p95']:8.1f}px")
    print("=" * 110)


# ===================================================================
# TFLite conversion (same as v2)
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
# Keras training helper
# ===================================================================
def train_keras_model(model, X_train, y_train, epochs=300, patience=30, batch_size=64, verbose=1, reduce_lr=False):
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]
    if reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
            )
        )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


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
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- User-stratified split -----
    if len(np.unique(user_labels)) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_idx, test_idx = next(gss.split(X_scaled, y, groups=user_labels))
    else:
        X_indices = np.arange(len(X_scaled))
        train_idx, test_idx = train_test_split(X_indices, test_size=0.15, random_state=42)

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\n  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")
    print(f"  Screen resolution (median): {int(screen_w)}x{int(screen_h)}")
    print(f"  Feature dimensions: {X_train.shape[1]}")

    # ----- Data augmentation -----
    print("\n  Augmenting training data with Gaussian noise...")
    X_train_aug, y_train_aug = augment_with_noise(X_train, y_train, noise_std=0.02, n_copies=2)
    print(f"  Augmented train size: {X_train_aug.shape[0]:,} (original: {X_train.shape[0]:,})")

    results = []
    keras_models = {}  # Store keras models for ensemble later

    # =================================================================
    # 1. sklearn baselines
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING SKLEARN BASELINES")
    print("=" * 60)
    sk_models = build_sklearn_models()
    sk_predictions = {}
    for name, model in sk_models.items():
        t0 = time.time()
        print(f"  Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)  # No augmentation for sklearn baselines
        y_pred = np.clip(model.predict(X_test), 0, 1)
        sk_predictions[name] = y_pred
        res = evaluate_model(name, y_test, y_pred, screen_w, screen_h)
        results.append(res)
        print(f"Euc={res['euc_mean']:.1f}px  ({time.time() - t0:.1f}s)")

    # =================================================================
    # 2. XGBoost / Tuned GradientBoosting
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING XGBOOST / TUNED GRADIENT BOOSTING")
    print("=" * 60)
    xgb_models = build_xgboost_models()
    for name, model in xgb_models.items():
        t0 = time.time()
        print(f"  Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, 1)
        sk_predictions[name] = y_pred
        res = evaluate_model(name, y_test, y_pred, screen_w, screen_h)
        results.append(res)
        print(f"Euc={res['euc_mean']:.1f}px  ({time.time() - t0:.1f}s)")

    # =================================================================
    # 3. MLP Small (original)
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING MLP SMALL")
    print("=" * 60)
    import tensorflow as tf

    mlp_small = build_mlp_small(X_train.shape[1])
    train_keras_model(mlp_small, X_train_aug, y_train_aug, epochs=200, patience=20, reduce_lr=True)
    y_pred_s = np.clip(mlp_small.predict(X_test, verbose=0), 0, 1)
    res_s = evaluate_model("MLP Small", y_test, y_pred_s, screen_w, screen_h)
    results.append(res_s)
    keras_models["MLP Small"] = (mlp_small, y_pred_s)
    print(f"  >> MLP Small  Euc={res_s['euc_mean']:.1f}px")

    # =================================================================
    # 4. MLP Large Residual (original v2)
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING MLP LARGE RESIDUAL (v1)")
    print("=" * 60)
    mlp_large = build_mlp_large(X_train.shape[1])

    total_steps = int(np.ceil(X_train_aug.shape[0] * 0.85 / 64)) * 300
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=total_steps, alpha=1e-6,
    )
    mlp_large.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(delta=0.05), metrics=["mae"],
    )

    train_keras_model(mlp_large, X_train_aug, y_train_aug, epochs=300, patience=30)
    y_pred_l = np.clip(mlp_large.predict(X_test, verbose=0), 0, 1)
    res_l = evaluate_model("MLP Large Residual", y_test, y_pred_l, screen_w, screen_h)
    results.append(res_l)
    keras_models["MLP Large Residual"] = (mlp_large, y_pred_l)
    print(f"  >> MLP Large Residual  Euc={res_l['euc_mean']:.1f}px")

    # =================================================================
    # 5. MLP Large v2 (tuned hyperparams)
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING MLP LARGE v2 (TUNED)")
    print("=" * 60)
    mlp_large_v2 = build_mlp_large_v2(X_train.shape[1])

    total_steps_v2 = int(np.ceil(X_train_aug.shape[0] * 0.85 / 32)) * 400
    lr_schedule_v2 = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-4, decay_steps=total_steps_v2, alpha=1e-6,
    )
    mlp_large_v2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_v2),
        loss=tf.keras.losses.Huber(delta=0.01), metrics=["mae"],
    )

    train_keras_model(mlp_large_v2, X_train_aug, y_train_aug, epochs=400, patience=40, batch_size=32)
    y_pred_lv2 = np.clip(mlp_large_v2.predict(X_test, verbose=0), 0, 1)
    res_lv2 = evaluate_model("MLP Large v2 Tuned", y_test, y_pred_lv2, screen_w, screen_h)
    results.append(res_lv2)
    keras_models["MLP Large v2 Tuned"] = (mlp_large_v2, y_pred_lv2)
    print(f"  >> MLP Large v2 Tuned  Euc={res_lv2['euc_mean']:.1f}px")

    # =================================================================
    # 6. Conv1D Model
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING CONV1D MODEL")
    print("=" * 60)
    conv1d = build_conv1d_model(X_train.shape[1])
    train_keras_model(conv1d, X_train_aug, y_train_aug, epochs=200, patience=25, reduce_lr=True)
    y_pred_c1d = np.clip(conv1d.predict(X_test, verbose=0), 0, 1)
    res_c1d = evaluate_model("Conv1D", y_test, y_pred_c1d, screen_w, screen_h)
    results.append(res_c1d)
    keras_models["Conv1D"] = (conv1d, y_pred_c1d)
    print(f"  >> Conv1D  Euc={res_c1d['euc_mean']:.1f}px")

    # =================================================================
    # 7. Attention MLP
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING ATTENTION MLP")
    print("=" * 60)
    attn_mlp = build_attention_mlp(X_train.shape[1])
    train_keras_model(attn_mlp, X_train_aug, y_train_aug, epochs=300, patience=30, reduce_lr=True)
    y_pred_attn = np.clip(attn_mlp.predict(X_test, verbose=0), 0, 1)
    res_attn = evaluate_model("Attention MLP", y_test, y_pred_attn, screen_w, screen_h)
    results.append(res_attn)
    keras_models["Attention MLP"] = (attn_mlp, y_pred_attn)
    print(f"  >> Attention MLP  Euc={res_attn['euc_mean']:.1f}px")

    # =================================================================
    # 8. Separate X/Y Models
    # =================================================================
    print("\n" + "=" * 60)
    print("  TRAINING SEPARATE X/Y MODELS")
    print("=" * 60)
    model_x, model_y = build_separate_xy_models(X_train.shape[1])

    print("  Training X-axis model...")
    train_keras_model(model_x, X_train_aug, y_train_aug[:, 0:1], epochs=300, patience=30, reduce_lr=True)
    print("  Training Y-axis model...")
    train_keras_model(model_y, X_train_aug, y_train_aug[:, 1:2], epochs=300, patience=30, reduce_lr=True)

    pred_x = np.clip(model_x.predict(X_test, verbose=0), 0, 1).flatten()
    pred_y = np.clip(model_y.predict(X_test, verbose=0), 0, 1).flatten()
    y_pred_sep = np.column_stack([pred_x, pred_y])
    res_sep = evaluate_model("Separate X/Y MLP", y_test, y_pred_sep, screen_w, screen_h)
    results.append(res_sep)
    print(f"  >> Separate X/Y MLP  Euc={res_sep['euc_mean']:.1f}px")

    # =================================================================
    # 9. Ensemble (weighted average of top keras models)
    # =================================================================
    print("\n" + "=" * 60)
    print("  COMPUTING ENSEMBLE")
    print("=" * 60)

    # Sort keras models by performance
    keras_results = [(name, pred, evaluate_model(name, y_test, pred, screen_w, screen_h)["euc_mean"])
                     for name, (_, pred) in keras_models.items()]
    keras_results.sort(key=lambda x: x[2])

    # Take top 3 models for ensemble
    top_n = min(3, len(keras_results))
    top_models = keras_results[:top_n]
    print(f"  Ensembling top {top_n} models:")
    for name, _, euc in top_models:
        print(f"    - {name}: {euc:.1f}px")

    # Weighted average — inverse error weighting
    total_inv_err = sum(1.0 / euc for _, _, euc in top_models)
    weights = [(1.0 / euc) / total_inv_err for _, _, euc in top_models]
    y_pred_ensemble = sum(w * pred for (_, pred, _), w in zip(top_models, weights))
    y_pred_ensemble = np.clip(y_pred_ensemble, 0, 1)

    res_ens = evaluate_model("Ensemble (Top-3 Weighted)", y_test, y_pred_ensemble, screen_w, screen_h)
    results.append(res_ens)
    print(f"  >> Ensemble  Euc={res_ens['euc_mean']:.1f}px")

    # Also include sklearn predictions in a larger ensemble
    all_preds = [pred for _, pred, _ in top_models]
    for sk_name in ["Gradient Boosting"]:
        if sk_name in sk_predictions:
            all_preds.append(sk_predictions[sk_name])

    if len(all_preds) > top_n:
        y_pred_mega_ens = np.mean(all_preds, axis=0)
        y_pred_mega_ens = np.clip(y_pred_mega_ens, 0, 1)
        res_mega = evaluate_model("Mega Ensemble (All)", y_test, y_pred_mega_ens, screen_w, screen_h)
        results.append(res_mega)
        print(f"  >> Mega Ensemble  Euc={res_mega['euc_mean']:.1f}px")

    # =================================================================
    # Results comparison
    # =================================================================
    print_results_table(results)

    best = min(results, key=lambda r: r["euc_mean"])
    print(f"\n>>> Best model: {best['name']}  (Mean Euclidean Error = {best['euc_mean']:.1f}px)")

    # =================================================================
    # Save best single keras model
    # =================================================================
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Find the best single keras model (not ensemble)
    best_keras_results = sorted(keras_results, key=lambda x: x[2])
    best_keras_name = best_keras_results[0][0]
    best_keras_model = keras_models[best_keras_name][0]

    best_keras_model.save(MODEL_SAVE_PATH)
    print(f"\n  Keras model saved  -> {MODEL_SAVE_PATH}  ({best_keras_name})")

    # Save all keras model variants
    for name, (model, _) in keras_models.items():
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        variant_path = os.path.join(MODEL_SAVE_DIR, f"{safe_name}.keras")
        model.save(variant_path)
        print(f"  Variant saved      -> {variant_path}")

    # Convert best to TFLite
    convert_to_tflite(best_keras_model, TFLITE_SAVE_PATH)

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
        n_train_augmented=X_train_aug.shape[0],
        n_test=X_test.shape[0],
        best_model=best["name"],
        best_euc_mean=best["euc_mean"],
        best_keras_saved=best_keras_name,
        all_results=results,
    )
    with open(META_PATH, "wb") as f:
        pickle.dump(training_meta, f)
    print(f"  Metadata saved     -> {META_PATH}")

    elapsed = time.time() - t_start
    print(f"\n  Total training time: {elapsed / 60:.1f} minutes")
    print(f"  Feature columns ({len(meta['col_names'])}): {meta['col_names']}")

    # Final comparison with previous best
    print("\n" + "=" * 60)
    print("  IMPROVEMENT SUMMARY")
    print("=" * 60)
    prev_best = 52.6  # Previous best from v2
    new_best = best["euc_mean"]
    if new_best < prev_best:
        improvement = ((prev_best - new_best) / prev_best) * 100
        print(f"  Previous best: {prev_best:.1f}px")
        print(f"  New best:      {new_best:.1f}px")
        print(f"  Improvement:   {improvement:.1f}%")
    else:
        print(f"  Previous best: {prev_best:.1f}px")
        print(f"  New best:      {new_best:.1f}px")
        print(f"  (No improvement over previous — more data needed)")
    print("=" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
