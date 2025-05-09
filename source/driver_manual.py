import keyboard
import time
import carState  
import msgParser
from logger import Logger  

class Driver:
    def __init__(self, stage):
        self.gear = 1
        self.stage = stage
        self.reverse_mode = False 
        
        self.logger = Logger()  
        self.parser = msgParser.MsgParser()  
        self.state = carState.CarState()  

        self.prev_gear = 1
        self.gear_changed_time = 0

    def init(self):
        return "(init {})".format(self.stage)

    def drive(self, sensors, trackname, carname):
        self.state.setFromMsg(sensors)

        steer = 0.0
        accel = 0.0
        brake = 0.0
        moving = False

        speedX = self.get_speed(sensors)
        rpm = self.state.rpm
        
        # Track recent gear change
        gear_downshifted_recently = False
        if self.gear < self.prev_gear:
            self.gear_changed_time = time.time()
        if time.time() - self.gear_changed_time < 0.5:
            gear_downshifted_recently = True

        # Left and right movement
        if keyboard.is_pressed("left"):
            steer = 1.0  
            moving = True
        elif keyboard.is_pressed("right"):
            steer = -1.0  
            moving = True

        # Gear shifting with RPM safety
        if keyboard.is_pressed("q") and self.gear != 1:
            self.gear = 1
        elif keyboard.is_pressed("w") and self.gear != 2:
            self.gear = 2
        elif keyboard.is_pressed("e") and self.gear != 3:
            self.gear = 3
        elif keyboard.is_pressed("r") and self.gear != 4:
            self.gear = 4
        elif keyboard.is_pressed("t") and self.gear != 5:
            self.gear = 5
        elif keyboard.is_pressed("y"):  
            self.gear = -1  
            self.reverse_mode = True  

        # Acceleration & max speed by gear
        gear_accel_map = {  -1: 0.3, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0  }
        max_speed_map = {   -1: -20,  1: 30, 2: 60, 3: 120, 4: 180, 5: 250   }

        # Reverse logic
        if self.gear == -1: 
            if keyboard.is_pressed("down"):
                if speedX > max_speed_map[self.gear]:
                    accel = gear_accel_map[self.gear]
                brake = 0.0
            elif keyboard.is_pressed("up"):
                brake = 1.0  
                moving = True
        else:
            if keyboard.is_pressed("up"):
                if speedX < max_speed_map[self.gear] - 5:
                    accel = gear_accel_map[self.gear]  
                brake = 0.0  
            elif keyboard.is_pressed("down"):
                brake = 1.0  
                moving = True

        # Traction control during turns
        if abs(steer) > 0.8 and speedX > 50:
            accel *= 0.7  # reduce power during hard turn

        # Reduce jerk after downshift
        if gear_downshifted_recently:
            accel *= 0.5
            steer *= 0.5
            brake += 0.2

        # Log data including track information
        self.logger.log_data(
            self.state.speedX, self.state.speedY, self.state.speedZ, self.state.gear, self.state.rpm, 
            accel, brake, steer, self.state.angle, self.state.trackPos, self.state.track, self.state.focus, 
            self.state.fuel, self.state.curLapTime, self.state.lastLapTime, 
            self.state.distFromStart, self.state.distRaced, self.state.damage, 
            self.state.opponents, self.state.racePos, self.state.wheelSpinVel, self.state.z, 
            trackname, carname
        )

        self.prev_gear = self.gear  # update previous gear

        return "(steer {})(accel {})(brake {})(gear {})".format(steer, accel, brake, self.gear)


    def get_speed(self, sensors):
        try:
            sensor_data = dict(item.split() for item in sensors.replace('(', '').replace(')', '').split())
            return float(sensor_data.get("speedX", 0))
        except:
            return 0.0  

    def onShutDown(self):
        pass

    def onRestart(self):
        pass
