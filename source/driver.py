import joblib
import numpy as np
from msgParser import MsgParser
from carState import CarState

class Driver:
    def __init__(self, stage):
        self.stage = stage
        self.parser = MsgParser()
        try:
            print("🔄 Loading model files...")
            self.model = joblib.load("torcs_model.pkl")
            self.scaler = joblib.load("torcs_scaler.pkl")
            self.imputer_X = joblib.load("torcs_imputer_X.pkl")
            self.imputer_y = joblib.load("torcs_imputer_y.pkl")
            self.target_scaler = joblib.load("torcs_target_scaler.pkl")
            print("✅ Model files loaded successfully.")

        except Exception as e:
            print(f"❌ Error loading model files: {str(e)}")
            raise

        self.feature_order = [
            'Timestamp', 'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 
            'Angle', 'TrackPos', 'Fuel', 'CurLapTime', 'LastLapTime',
            'DistFromStart', 'DistRaced', 'Damage', 'RacePos', 'Z',
            *[f'Track_{i}' for i in range(19)],
            *[f'Focus_{i}' for i in range(5)],
            *[f'WheelSpinVel_{i}' for i in range(4)]
        ]
        self.current_focus = np.array([-1.0] * 5)
        self.feature_order = [
            'Timestamp', 'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 
            'Angle', 'TrackPos', 'Fuel', 'CurLapTime', 'LastLapTime',
            'DistFromStart', 'DistRaced', 'Damage', 'RacePos', 'Z',
            *[f'Track_{i}' for i in range(19)],
            *[f'Focus_{i}' for i in range(5)],
            *[f'WheelSpinVel_{i}' for i in range(4)]
        ]

        # 🔍 Verify feature order matches training
        try:
            with open("torcs_feature_order.txt") as f:
                training_order = [line.strip() for line in f]
            if self.feature_order != training_order:
                raise ValueError("Feature order mismatch between runtime and training.")
            print("✅ Feature order verified and matches training.")
        except Exception as e:
            print(f"❌ Feature order check failed: {e}")
            raise

        

    def init(self):
        return f"(init {self.stage})"

    def drive(self, str_sensors, track, car):
        try:
            car_state = CarState()
            car_state.setFromMsg(str_sensors)

            print("🚗 Parsing car state...")

            features = [
                0.0,
                car_state.speedX,
                car_state.speedY,
                car_state.speedZ,
                car_state.rpm,
                car_state.angle,
                car_state.trackPos,
                car_state.fuel,
                car_state.curLapTime,
                car_state.lastLapTime,
                car_state.distFromStart,
                car_state.distRaced,
                car_state.damage,
                car_state.racePos,
                car_state.z,
                *car_state.track,
                *self.current_focus,
                *car_state.wheelSpinVel
            ]

            print("🧠 Raw feature vector:")
            for name, val in zip(self.feature_order, features):
                print(f"   {name}: {val:.4f}" if isinstance(val, float) else f"   {name}: {val}")

            features = np.array(features).reshape(1, -1)
            features = self.imputer_X.transform(features)
            features = self.scaler.transform(features)

            print("📏 Scaled feature vector (first 10):", features[0][:10])
            print("📐 Feature vector shape:", features.shape)

            output = self.model.predict(features)                         # shape: (1, 9)
            output = self.imputer_y.inverse_transform(output)             # still (1, 9)
            output = self.target_scaler.inverse_transform(output)[0]      # extract to 1D shape: (9,)

            print("🔮 Raw model output:", output)

            accel = np.clip(output[0], 0, 1)
            brake = np.clip(output[1], 0, 1)
            gear = int(np.clip(round(output[2]), 1, 6))
            steer = np.clip(output[3], -1, 1)
            new_focus = np.clip(output[4:9], -1, 1)

            print(f"🚦 Predicted controls → Accel: {accel:.3f}, Brake: {brake:.3f}, Gear: {gear}, Steer: {steer:.3f}")
            print("👀 Predicted focus:", new_focus)

            self.current_focus = new_focus

            return f"(accel {accel:.3f})(brake {brake:.3f})(gear {gear})(steer {steer:.3f})"

        except Exception as e:
            print(f"❌ Error in drive(): {str(e)}")
            self.current_focus = np.array([-1.0] * 5)
            return "(accel 0.2)(brake 0)(gear 1)(steer 0)"

    def get_state_features(self, car_state):
        return {
            'Timestamp': 0.0,
            'SpeedX': car_state.speedX,
            'SpeedY': car_state.speedY,
            'SpeedZ': car_state.speedZ,
            'RPM': car_state.rpm,
            'Angle': car_state.angle,
            'TrackPos': car_state.trackPos,
            'Fuel': car_state.fuel,
            'CurLapTime': car_state.curLapTime,
            'LastLapTime': car_state.lastLapTime,
            'DistFromStart': car_state.distFromStart,
            'DistRaced': car_state.distRaced,
            'Damage': car_state.damage,
            'RacePos': car_state.racePos,
            'Z': car_state.z,
            **{f'Track_{i}': val for i, val in enumerate(car_state.track)},
            **{f'Focus_{i}': val for i, val in enumerate(self.current_focus)},
            **{f'WheelSpinVel_{i}': val for i, val in enumerate(car_state.wheelSpinVel)}
        }
