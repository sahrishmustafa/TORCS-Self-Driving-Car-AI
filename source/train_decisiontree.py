import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import joblib
import ast

def load_data(filepath):
    """Load dataset from CSV file and convert column names to lowercase."""
    try:
        data = pd.read_csv(filepath)
        data.columns = [col.lower() for col in data.columns]  # Convert column names to lowercase
        print(f"Data loaded successfully with {len(data)} samples.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def safe_convert_to_array(vector_str):
    """Safely convert string representation of vector to numpy array."""
    try:
        cleaned_str = vector_str.strip().replace('\n', '').replace('\r', '')
        if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
            return np.array(ast.literal_eval(cleaned_str))
        return np.array([float(cleaned_str)])
    except:
        return np.nan

def expand_vector_features(df, keep_original=False):
    """
    Expand vector features into individual columns.
    - Focus: 5 elements
    - WheelSpinVel: 4 elements
    - Track: 19 elements
    """
    # Handle Focus
    if 'focus' in df.columns:
        focus_data = df['focus'].apply(safe_convert_to_array)
        focus_cols = [f'focus_{i}' for i in range(5)]
        clean_focus_data = [[item] if isinstance(item, float) else item for item in focus_data]
        df[focus_cols] = pd.DataFrame(clean_focus_data, index=df.index)
        if not keep_original:
            df.drop('focus', axis=1, inplace=True)
    
    # Handle WheelSpinVel
    if 'wheelspinvel' in df.columns:
        wheel_data = df['wheelspinvel'].apply(safe_convert_to_array)
        wheel_cols = [f'wheelspinvel_{i}' for i in range(4)]
        clean_wheel_data = [[item] if isinstance(item, float) else item for item in wheel_data]
        df[wheel_cols] = pd.DataFrame(clean_wheel_data, index=df.index)
        if not keep_original:
            df.drop('wheelspinvel', axis=1, inplace=True)
    
    # Handle Track
    if 'track' in df.columns:
        track_data = df['track'].apply(safe_convert_to_array)
        track_cols = [f'track_{i}' for i in range(19)]
        clean_track_data = [[item] if isinstance(item, float) else item for item in track_data]
        df[track_cols] = pd.DataFrame(clean_track_data, index=df.index)
        if not keep_original:
            df.drop('track', axis=1, inplace=True)
    
    return df

def preprocess_data(data, input_features, output_features):
    """Preprocess data with proper handling of vector features and debugging."""
    data = data.copy()

    # Expand vector features if present
    if 'focus' in data.columns:
        data = expand_vector_features(data, keep_original=True)
        focus_cols = [f'focus_{i}' for i in range(5)]
    else:
        focus_cols = []

    # Separate outputs
    y = data[output_features + focus_cols].copy()

    # Process input features
    updated_input_features = []
    for feat in input_features:
        if feat == 'focus':
            continue  # Skip original Focus
        elif feat == 'wheelspinvel':
            wheel_cols = [f'wheelspinvel_{i}' for i in range(4)]
            updated_input_features.extend(wheel_cols)
        elif feat == 'track':
            track_cols = [f'track_{i}' for i in range(19)]
            updated_input_features.extend(track_cols)
        else:
            updated_input_features.append(feat)

    # Try selecting the columns
    try:
        X = data[updated_input_features].copy()
    except KeyError as e:
        print("KeyError while selecting features. Details:", e)
        raise

    # Handle missing values
    imputer_X = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)

    imputer_y = SimpleImputer(strategy='mean')
    y = pd.DataFrame(imputer_y.fit_transform(y), columns=y.columns)

    # Scale features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Custom scaling for outputs
    output_scalers = {}
    y_scaled = np.zeros_like(y)
    for i, col in enumerate(y.columns):
        if col in ['steer']:
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif col in ['accel', 'brake'] or col.startswith('focus_'):
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:  # Gear and others
            scaler = StandardScaler()

        y_scaled[:, i] = scaler.fit_transform(y[[col]].values.reshape(-1, 1)).flatten()
        output_scalers[col] = scaler

    return X_scaled, y_scaled, scaler_X, output_scalers


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """
    Train a multi-output regression model using DecisionTreeRegressor.
    """
    base_model = DecisionTreeRegressor(
        max_depth=10,           
        min_samples_split=5,    
        random_state=42
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model

import matplotlib.pyplot as plt

def evaluate_model(model, X_train, y_train, X_test, y_test, output_features, output_scalers):
    """Evaluate model for signs of overfitting."""
    
    def inverse_transform(y, scalers):
        y_orig = np.zeros_like(y)
        for i, col in enumerate(output_features):
            scaler = scalers[col]
            y_orig[:, i] = scaler.inverse_transform(y[:, i].reshape(-1, 1)).flatten()
        return y_orig

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Inverse transform
    y_train_orig = inverse_transform(y_train, output_scalers)
    y_test_orig = inverse_transform(y_test, output_scalers)
    y_train_pred_orig = inverse_transform(y_train_pred, output_scalers)
    y_test_pred_orig = inverse_transform(y_test_pred, output_scalers)

    # Compare Train vs Test MSE
    train_mse_scores = {}
    test_mse_scores = {}

    for i, feature in enumerate(output_features):
        train_mse = mean_squared_error(y_train_orig[:, i], y_train_pred_orig[:, i])
        test_mse = mean_squared_error(y_test_orig[:, i], y_test_pred_orig[:, i])
        
        train_mse_scores[feature] = train_mse
        test_mse_scores[feature] = test_mse
        
        print(f"{feature}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")
    
    # Plotting to visualize overfitting
    plt.figure(figsize=(8, 4))
    bar_width = 0.35
    index = np.arange(len(output_features))
    
    plt.bar(index, [train_mse_scores[f] for f in output_features], bar_width, label='Train MSE')
    plt.bar(index + bar_width, [test_mse_scores[f] for f in output_features], bar_width, label='Test MSE')
    
    plt.xlabel('Output Features')
    plt.ylabel('MSE')
    plt.title('Train vs Test MSE')
    plt.xticks(index + bar_width / 2, output_features, rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return train_mse_scores, test_mse_scores

def save_model(model, scaler_X, output_scalers, metadata, filepath):
    """Save model and scalers for later use."""
    joblib.dump({
        'model': model,
        'scaler_X': scaler_X,
        'output_scalers': output_scalers,
        'metadata': metadata
    }, filepath)
    print(f"Model saved to {filepath}")

def main():
    # Configuration
    DATA_PATH = 'dataset/dirt2.csv'
    MODEL_SAVE_PATH = 'torcs_car_control_model.joblib'
    
    # Define features.
    input_features = [
        'speedx', 'speedy', 'speedz', 'rpm', 'angle', 'trackpos',
        'track', 'fuel', 'curlaptime', 'lastlaptime',
        'distfromstart', 'distraced', 'damage',
        'racepos', 'wheelspinvel', 'z', 'trackname', 'carname'
    ]

    output_features = ['accel', 'steer', 'brake', 'gear']
    
    # Step 1: Load data
    data = load_data(DATA_PATH)
    if data is None:
        return
    
    # Step 2: Preprocess data
    print("Preprocessing data...")
    X, y, scaler_X, output_scalers = preprocess_data(data, input_features, output_features)
    
    # Step 3: Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    print("\nModel evaluation:")
    mse_scores = evaluate_model(model, X_train, y_train, X_test, y_test, output_features, output_scalers)
    
    # Step 6: Save model
    metadata = {
        'input_features': input_features,
        'output_features': output_features + ['focus_0', 'focus_1', 'focus_2', 'focus_3', 'focus_4'],
        'feature_importances': dict(zip(
            scaler_X.get_feature_names_out(),
            model.estimators_[0].feature_importances_
        ))
    }
    save_model(model, scaler_X, output_scalers, metadata, MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()