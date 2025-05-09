import ast
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_parse_data(filepath, vector_columns, target_columns):
    df = pd.read_csv(filepath)

    # Convert stringified vectors to NumPy arrays
    for col in vector_columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float64) if isinstance(x, list) else np.nan)

    return df


def extract_scalar_features(df, exclude_columns):
    scalar_columns = [col for col in df.columns if col not in exclude_columns]
    scalar_features = df[scalar_columns].apply(pd.to_numeric, errors='coerce')

    if scalar_features.isna().any().any():
        scalar_features = scalar_features.fillna(scalar_features.mean())

    return scalar_features


def extract_vector_features(df, vector_columns):
    vector_arrays = []

    for col in vector_columns:
        col_data = df[col].apply(lambda x: x if isinstance(x, np.ndarray) else np.nan)

        # Get max vector length
        max_len = max((len(x) for x in col_data if isinstance(x, np.ndarray)), default=0)

        # Pad or replace with NaN-filled arrays
        padded_data = col_data.apply(lambda x: np.pad(x, (0, max_len - len(x)), mode='constant') 
                                     if isinstance(x, np.ndarray) else np.full(max_len, np.nan))

        stacked = np.stack(padded_data.values)
        vector_arrays.append(stacked)

    # Concatenate all vector features
    vector_features = np.hstack(vector_arrays)

    # Handle NaNs
    if np.isnan(vector_features).any():
        col_means = np.nanmean(vector_features, axis=0)
        inds = np.where(np.isnan(vector_features))
        vector_features[inds] = np.take(col_means, inds[1])

    return vector_features


def extract_targets(df, target_columns):
    y = df[target_columns].apply(pd.to_numeric, errors='coerce')

    if y.isna().any().any():
        y = y.fillna(y.mean())

    return y


def train_and_save_model(X, y, model_path="torcs_driver_model.pkl", scaler_path="torcs_scaler.pkl"):
    # Final safety check: impute any NaNs
    if np.isnan(X).any():
        print("⚠️ Final NaNs found in X. Using SimpleImputer to replace them.")
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

    # Scale and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    # Train model
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print("✅ Model training completed and saved.")



def main():
    vector_columns = ['Track', 'Focus', 'WheelSpinVel']
    target_columns = ['Accel', 'Brake', 'Gear', 'Steer']
    filepath = "logfile.csv"

    df = load_and_parse_data(filepath, vector_columns, target_columns)
    scalar_features = extract_scalar_features(df, exclude_columns=vector_columns + target_columns)
    vector_features = extract_vector_features(df, vector_columns)
    X = np.hstack([scalar_features.values, vector_features])
    y = extract_targets(df, target_columns)

    train_and_save_model(X, y)


if __name__ == "__main__":
    main()
