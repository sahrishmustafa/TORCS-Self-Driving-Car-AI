import joblib
import numpy as np
from msgParser import MsgParser  # Import the message parser
from carState import CarState

class Driver:
    def __init__(self, stage):
        self.stage = stage
        self.parser = MsgParser()  # Create a message parser instance
        self.model = joblib.load("torcs_driver_model.pkl")
        self.scaler = joblib.load("torcs_scaler.pkl")
        
        # Define expected features in the order the model expects them
        self.input_features = [
            'speedX', 'speedY', 'speedZ', 'gear', 'rpm', 'angle', 'trackPos',
            'fuel', 'curLapTime', 'lastLapTime', 'distFromStart', 'distRaced',
            'damage', 'racePos', 'z', 'stage', 'carname'
        ]

    def init(self):
        return "(init {})".format(self.stage)

    def drive(self, str_sensors, track, car):
        try:
            # Create and populate CarState object
            car_state = CarState()
            car_state.setFromMsg(str_sensors)
            
            features = self.extract_features(car_state)
            # Ensure all features are present and in correct order
            input_data = np.array([[features.get(key, 0.0) for key in self.input_features]])
            input_scaled = self.scaler.transform(input_data)
            output = self.model.predict(input_scaled)[0]
            
            # Assuming output is [accel, brake, gear, steer, focus0, focus1, focus2, focus3, focus4]
            accel, brake, gear, steer, *focus = output
            focus = [int(round(f)) for f in focus]  # Convert all focus values
            
            return f"(accel {accel})(brake {brake})(gear {int(round(gear))})(steer {steer})(focus {' '.join(map(str, focus))})"
        except Exception as e:
            print(f"Error in drive(): {str(e)}")
            # Return default safe values in case of error
            return "(accel 0.2)(brake 0)(gear 1)(steer 0)(focus 0 0 0 0 0)"

    def extract_features(self, car_state):
        """Extract features from CarState object"""
        features = {
            'speedX': car_state.getSpeedX() if car_state.getSpeedX() is not None else 0.0,
            'speedY': car_state.getSpeedY() if car_state.getSpeedY() is not None else 0.0,
            'speedZ': car_state.getSpeedZ() if car_state.getSpeedZ() is not None else 0.0,
            'gear': car_state.getGear() if car_state.getGear() is not None else 1,
            'rpm': car_state.getRpm() if car_state.getRpm() is not None else 0.0,
            'angle': car_state.getAngle() if car_state.getAngle() is not None else 0.0,
            'trackPos': car_state.getTrackPos() if car_state.getTrackPos() is not None else 0.0,
            'fuel': car_state.getFuel() if car_state.getFuel() is not None else 0.0,
            'curLapTime': car_state.getCurLapTime() if car_state.getCurLapTime() is not None else 0.0,
            'lastLapTime': car_state.getLastLapTime() if car_state.getLastLapTime() is not None else 0.0,
            'distFromStart': car_state.getDistFromStart() if car_state.getDistFromStart() is not None else 0.0,
            'distRaced': car_state.getDistRaced() if car_state.getDistRaced() is not None else 0.0,
            'damage': car_state.getDamage() if car_state.getDamage() is not None else 0.0,
            'racePos': car_state.getRacePos() if car_state.getRacePos() is not None else 1,
            'z': car_state.getZ() if car_state.getZ() is not None else 0.0,
            'stage': self.stage,
            'carname': 1  # Default value
        }
        
        # Handle focus if available
        if hasattr(car_state, 'focus') and car_state.focus is not None:
            try:
                focus_values = list(car_state.focus)[:5]  # Take first 5 focus values
                features.update({f'focus{i}': val for i, val in enumerate(focus_values)})
            except:
                pass
                
        return features