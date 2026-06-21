import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = "../dataset"
MODEL_SAVE_PATH = "gaze_model.keras"

def load_data():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}!")
        return None, None
    
    print(f"Found {len(csv_files)} dataset files.")
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # Drop rows where face tracking failed (zeros or nulls)
    df = df.dropna()
    
    print(f"Total samples loaded: {len(df)}")
    
    # --- Feature Engineering ---
    # We extract the 10 core features
    X = df[[
        "head_pitch", "head_yaw", "head_roll",
        "l_iris_x", "l_iris_y", "l_iris_z",
        "r_iris_x", "r_iris_y", "r_iris_z",
        "inter_ocular_dist"
    ]].values
    
    # --- Target Variables ---
    # Normalize the target X and Y coordinates to be between 0 and 1
    # This makes the neural network train much faster and more stable!
    y_x = df["target_x"] / df["screen_w"]
    y_y = df["target_y"] / df["screen_h"]
    y = np.column_stack((y_x, y_y))
    
    return X, y

def build_model(input_dim):
    # A lightweight Multi-Layer Perceptron (MLP)
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(2, activation='sigmoid') # Sigmoid forces output between 0 and 1
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    X, y = load_data()
    if X is None:
        return
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print("Building model...")
    model = build_model(input_dim=X.shape[1])
    model.summary()
    
    print("Training model...")
    # Early stopping prevents overfitting if we train for too many epochs
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Mean Absolute Error (Normalized): {test_mae:.4f}")
    
    # Translate normalized error back to approximate pixels (assuming 1080p screen)
    px_error_x = test_mae * 1920
    px_error_y = test_mae * 1080
    print(f"Approximate Average Pixel Error: X: ±{px_error_x:.1f}px | Y: ±{px_error_y:.1f}px")
    
    # Save the model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}!")

if __name__ == "__main__":
    main()
