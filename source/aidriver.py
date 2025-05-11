import joblib
import numpy as np
import pandas as pd

class AIDriver:
    def __init__(self, track_name, car_name, model_path='torcs_car_control_model.joblib'):
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler_X = self.model_data['scaler_X']
        self.output_scalers = self.model_data['output_scalers']
        self.metadata = self.model_data['metadata']
        
        # Store current context
        self.track_name = track_name
        self.car_name = car_name
        
        # Get feature names from scaler (this is the ground truth)
        self.expected_features = self.scaler_X.feature_names_in_
        print("Model expects features:", len(self.expected_features))
        print("Sample features:", self.expected_features[:5], "...")
        
        # Debug: print output features and scalers
        print("Output features:", list(self.output_scalers.keys()))
        
        # Initialize debug counters
        self.call_count = 0
        self.debug_interval = 100  # Print debug info every X calls
        
        # Store previous controls for debugging
        self.prev_controls = None
        
        # Startup boost phase
        self.startup_phase = True
        self.startup_steps = 0
        self.max_startup_steps = 100  # Apply startup boost for this many steps

    def _prepare_inputs(self, sensor_data):
        """Convert raw sensor data to properly ordered, scaled features"""
        # Debug: occasionally print raw sensor data
        self.call_count += 1
        if self.call_count % self.debug_interval == 1:
            print(f"Call #{self.call_count}")
            print("Raw sensor keys:", list(sensor_data.keys()))
            for key in ['speedx', 'angle', 'trackpos']:
                if key in sensor_data:
                    print(f"  {key}: {sensor_data.get(key)}")
        
        # Normalize keys to lowercase
        sensor_data = {k.lower(): v for k, v in sensor_data.items()}

        features = {}

        # Use lowercase feature names
        simple_features = ['speedx', 'speedy', 'speedz', 'rpm', 'angle',
                        'trackpos', 'fuel', 'curlaptime', 'lastlaptime',
                        'distfromstart', 'distraced', 'damage', 'racepos', 'z']
        
        for feat in simple_features:
            features[feat] = sensor_data.get(feat, 0.0)
        
        # Add track_name and car_name if needed
        if 'trackname' in self.expected_features:
            features['trackname'] = self.track_name
        if 'carname' in self.expected_features:
            features['carname'] = self.car_name

        # Expand vector features from lowercased keys
        features.update({
            f'wheelspinvel_{i}': sensor_data.get('wheelspinvel', [0]*4)[i] 
            if i < len(sensor_data.get('wheelspinvel', [])) else 0.0
            for i in range(4)
        })
        
        features.update({
            f'track_{i}': sensor_data.get('track', [0]*19)[i]
            if i < len(sensor_data.get('track', [])) else 0.0
            for i in range(19)
        })

        # Create a pandas DataFrame with the correct feature names
        df = pd.DataFrame([features])
        
        # Ensure all expected features are in the DataFrame (fill missing with 0)
        for feat in self.expected_features:
            if feat not in df.columns:
                df[feat] = 0.0
                
        # Order columns to match the expected order
        df = df[self.expected_features]
        
        # Debug: occasionally show prepared features
        if self.call_count % self.debug_interval == 1:
            print("Features shape:", df.shape)
            print("Missing expected features:", [f for f in self.expected_features if f not in df.columns])
        
        return df

    def get_control(self, sensor_data):
        """Get control outputs from sensor inputs"""
        # Check if we're in startup phase (car isn't moving)
        speed_x = sensor_data.get('speedx', 0.0)
        if abs(speed_x) < 5.0:
            self.startup_steps += 1
        else:
            self.startup_phase = False
            self.startup_steps = 0

        # During startup phase, apply special override
        if self.startup_phase and self.startup_steps < self.max_startup_steps:
            print(f"STARTUP PHASE: Step {self.startup_steps}/{self.max_startup_steps}, Speed: {speed_x}")
            return {
                'accel': 1.0,   # Full acceleration
                'brake': 0.0,   # No braking
                'steer': 0.0,   # Straight ahead
                'gear': 1       # First gear
            }
            
        # Normal model prediction
        X = self._prepare_inputs(sensor_data)
        X_scaled = pd.DataFrame(
            self.scaler_X.transform(X),
            columns=self.expected_features
        )
        
        # Debug check for NaN or infinity values
        if np.any(~np.isfinite(X_scaled.values)):
            print("WARNING: Non-finite values detected in scaled input!")
            print("Non-finite columns:", X_scaled.columns[~np.isfinite(X_scaled.values[0])])
            # Replace NaN/inf with 0 to prevent crashes
            X_scaled = X_scaled.fillna(0).replace([np.inf, -np.inf], 0)
            
        y_pred_scaled = self.model.predict(X_scaled.values)

        # Debug: print raw predictions sometimes
        if self.call_count % self.debug_interval == 1:
            print("Raw predictions shape:", y_pred_scaled.shape)
            if hasattr(y_pred_scaled, 'shape') and len(y_pred_scaled.shape) > 1 and y_pred_scaled.shape[1] >= 4:
                print("Raw scaled values:", y_pred_scaled[0, :4])

        # Get the output feature names to ensure correct order
        output_keys = ['accel', 'steer', 'brake', 'gear']
        controls = {}

        for i, name in enumerate(output_keys):
            # Skip if we don't have enough predictions
            if i >= y_pred_scaled.shape[1]:
                print(f"WARNING: Missing prediction for {name} (index {i})")
                controls[name] = 0 if name != 'gear' else 1
                continue
                
            try:
                scaler = self.output_scalers[name]
                value = float(scaler.inverse_transform(
                    y_pred_scaled[:, i].reshape(-1, 1)))
                    
                # Apply constraints to ensure valid values
                if name == 'accel':
                    value = max(0.0, min(1.0, value))  # Ensure 0-1 range
                    # If going slow, boost acceleration
                    if abs(speed_x) < 10.0:
                        value = max(value, 0.5)  # At least 50% acceleration when slow
                elif name == 'brake':
                    value = max(0.0, min(1.0, value))  # Ensure 0-1 range
                    # Don't brake when starting
                    if abs(speed_x) < 10.0:
                        value = 0.0  # No braking when slow
                elif name == 'steer':
                    value = max(-1.0, min(1.0, value))  # Ensure -1 to 1 range
                elif name == 'gear':
                    # Make sure gear makes sense for speed
                    if abs(speed_x) < 5:
                        value = 1  # First gear when slow
                    elif abs(speed_x) < 25:
                        value = max(value, 1)  # At least first gear
                    elif abs(speed_x) < 50:
                        value = max(value, 2)  # At least second gear
                    elif abs(speed_x) < 75:
                        value = max(value, 3)  # At least third gear
                        
                controls[name] = value
            except Exception as e:
                print(f"Error processing {name}: {e}")
                # Provide default values if there's an error
                if name == 'accel':
                    controls[name] = 0.5  # Default to medium acceleration
                elif name == 'gear':
                    controls[name] = 1    # Default to first gear
                elif name == 'steer':
                    controls[name] = 0.0  # Default to straight
                else:
                    controls[name] = 0.0  # Default to no brake

        # Round gear and ensure valid range
        controls['gear'] = int(round(controls['gear']))
        controls['gear'] = max(1, min(6, controls['gear']))
        
        # Make sure we're not braking and accelerating at the same time
        if controls['accel'] > 0.5 and controls['brake'] > 0.1:
            controls['brake'] = 0.0  # Prioritize acceleration over braking
        
        # Debug output changes in controls
        if self.prev_controls is not None and self.call_count % self.debug_interval == 1:
            print("Controls:", controls)
            for k in controls:
                if k in self.prev_controls and abs(controls[k] - self.prev_controls[k]) > 0.01:
                    print(f"  {k} changed: {self.prev_controls[k]:.3f} -> {controls[k]:.3f}")
        
        self.prev_controls = controls.copy()
        return controls