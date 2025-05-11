import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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
        df[focus_cols] = pd.DataFrame(focus_data.tolist(), index=df.index)
        if not keep_original:
            df.drop('focus', axis=1, inplace=True)
    
    # Handle WheelSpinVel
    if 'wheelspinvel' in df.columns:
        wheel_data = df['wheelspinvel'].apply(safe_convert_to_array)
        wheel_cols = [f'wheelspinvel_{i}' for i in range(4)]
        df[wheel_cols] = pd.DataFrame(wheel_data.tolist(), index=df.index)
        if not keep_original:
            df.drop('wheelspinvel', axis=1, inplace=True)
    
    # Handle Track
    if 'track' in df.columns:
        track_data = df['track'].apply(safe_convert_to_array)
        track_cols = [f'track_{i}' for i in range(19)]
        df[track_cols] = pd.DataFrame(track_data.tolist(), index=df.index)
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
    Train a multi-output regression model.
    Using RandomForest with reasonable defaults.
    """
    print("training")
    base_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    print("part 1")
    model = MultiOutputRegressor(base_model)
    print("part2")
    model.fit(X_train, y_train)
    print("part3")
    return model

def evaluate_model(model, X_test, y_test, output_features, output_scalers):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    # Inverse transform the scaled outputs
    y_test_orig = np.zeros_like(y_test)
    y_pred_orig = np.zeros_like(y_pred)
    
    for i, col in enumerate(output_features):
        scaler = output_scalers[col]
        y_test_orig[:, i] = scaler.inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()
        y_pred_orig[:, i] = scaler.inverse_transform(y_pred[:, i].reshape(-1, 1)).flatten()
    
    # Calculate MSE for each output
    mse_scores = {}
    for i, feature in enumerate(output_features):
        mse = mean_squared_error(y_test_orig[:, i], y_pred_orig[:, i])
        mse_scores[feature] = mse
        print(f"MSE for {feature}: {mse:.4f}")
    
    return mse_scores

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
    DATA_PATH = 'dataset/dirt3.csv'
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
    mse_scores = evaluate_model(model, X_test, y_test, output_features, output_scalers)
    
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