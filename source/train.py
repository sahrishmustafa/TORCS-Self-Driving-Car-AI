import ast
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset"""
    df = pd.read_csv(filepath)
    
    # Convert stringified lists to actual arrays
    vector_cols = ['Track', 'Focus', 'Opponents', 'WheelSpinVel']
    for col in vector_cols:
        df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
    
    return df

def prepare_features_and_targets(df):
    """Prepare features and targets with proper dimensions"""
    # Scalar features (individual columns)
    scalar_features = df[[
        'Timestamp', 'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 
        'Angle', 'TrackPos', 'Fuel', 'CurLapTime', 'LastLapTime',
        'DistFromStart', 'DistRaced', 'Damage', 'RacePos', 'Z'
    ]].values
    
    # Vector features (expanded arrays)
    track_features = np.stack(df['Track'].values)  # 19 features
    focus_features = np.stack(df['Focus'].values)  # 5 features (input)
    wheel_features = np.stack(df['WheelSpinVel'].values)  # 4 features
    
    # Combine all features (43 total)
    X = np.hstack([
        scalar_features,
        track_features,
        focus_features,
        wheel_features
    ])
    
    # Targets (4 controls + 5 focus outputs)
    y_controls = df[['Accel', 'Brake', 'Gear', 'Steer']].values
    y_focus = np.stack(df['Focus'].values)  # Predict focus again
    
    y = np.hstack([y_controls, y_focus])
    
    return X, y

def train_model(X, y):
    """Train and save the model"""
    print(f"\n🔧 Raw feature shape: {X.shape}, Target shape: {y.shape}")

    # Save and log training feature order
    feature_order = [
        'Timestamp', 'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 
        'Angle', 'TrackPos', 'Fuel', 'CurLapTime', 'LastLapTime',
        'DistFromStart', 'DistRaced', 'Damage', 'RacePos', 'Z',
        *[f'Track_{i}' for i in range(19)],
        *[f'Focus_{i}' for i in range(5)],
        *[f'WheelSpinVel_{i}' for i in range(4)]
    ]
    with open("torcs_feature_order.txt", "w") as f:
        for feat in feature_order:
            f.write(f"{feat}\n")
    print("📄 Saved feature order to torcs_feature_order.txt")

    # Imputers
    imputer_X = SimpleImputer(strategy='mean',add_indicator=True)
    imputer_y = SimpleImputer(strategy='mean',add_indicator=True)
    
    X_imputed = imputer_X.fit_transform(X)
    y_imputed = imputer_y.fit_transform(y)

    print(f"\n🧼 First 5 values after imputation: {X_imputed[0][:5]}")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Target scaling
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y_imputed)

    print(f"\n📊 Scaler mean (first 5): {scaler.mean_[:5]}")
    print(f"📈 Scaler scale (first 5): {scaler.scale_[:5]}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Train model
    model = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation='relu',
        solver='adam',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\n✅ Training R²: {train_score:.3f}, Testing R²: {test_score:.3f}")
    
    # Save artifacts
    joblib.dump(model, 'torcs_model.pkl')
    joblib.dump(scaler, 'torcs_scaler.pkl')
    joblib.dump(imputer_X, 'torcs_imputer_X.pkl')
    joblib.dump(imputer_y, 'torcs_imputer_y.pkl')
    joblib.dump(target_scaler, 'torcs_target_scaler.pkl')
    
    print(f"💾 Model and preprocessing artifacts saved.")

def main():
    # Configuration
    DATA_PATH = "logfile.csv"
    
    # Load and preprocess
    df = load_and_preprocess_data(DATA_PATH)
    
    # Prepare features and targets
    X, y = prepare_features_and_targets(df)
    
    # Print sample data
    print("\n🧠 Sample input features:")
    print(f"Scalars: {X[0, :15]}...")
    print(f"Track: {X[0, 15:15+19][:5]}...")
    print(f"Focus: {X[0, 15+19:15+19+5]}")
    print(f"Wheel: {X[0, 15+19+5:15+19+5+4]}")
    
    print("\n🎯 Sample targets:")
    print(f"Controls: {y[0, :4]}")
    print(f"Focus: {y[0, 4:]}")
    
    # Train and save
    train_model(X, y)

if __name__ == "__main__":
    main()
