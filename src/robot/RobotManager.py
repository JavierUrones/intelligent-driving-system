import MotorCore as mc
import Sensor as s


class RobotManager:
    def __init__(self):
        self.motor_core = mc.MotorCore()
        self.sensor = s.Sensor()

    def detect_obstacles(self):
        return self.sensor.detect_distance()

    def drive(self, steering_angle, speed):
        self.motor_core.drive(steering_angle, speed)
