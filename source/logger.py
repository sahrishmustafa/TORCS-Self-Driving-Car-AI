import os
import csv
import time

class Logger:
    def __init__(self, filename="logfile.csv"):
        self.filename = filename

        # Makes a file, if it doesn't exists already.
        # Appends the header names in it. 
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Timestamp", "SpeedX", "SpeedY", "SpeedZ", "Gear", "RPM", "Accel", "Brake", "Steer",
                    "Angle", "TrackPos", "Track", "Focus", "Fuel", "CurLapTime", "LastLapTime",
                    "DistFromStart", "DistRaced", "Damage", "Opponents", "RacePos", "WheelSpinVel", "Z",
                    "TrackName", "CarName"
                ])

    def log_data(self, speedX, speedY, speedZ, gear, rpm, accel, brake, steer,
                 angle, trackPos, track, focus, fuel, curLapTime, lastLapTime,
                 distFromStart, distRaced, damage, opponents, racePos, wheelSpinVel, z,
                 stage, carname):
        
        with open(self.filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                time.time(), speedX, speedY, speedZ, gear, rpm, accel, brake, steer,
                angle, trackPos, track, focus, fuel, curLapTime, lastLapTime,
                distFromStart, distRaced, damage, opponents, racePos, wheelSpinVel, z,
                stage, carname
            ])
            
            file.flush()
