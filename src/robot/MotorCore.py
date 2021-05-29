import sys

sys.path.append('/home/pi/Desktop/TFG/intelligent-driving-system')

from src.robot.MotorController import LeftMotor, RightMotor
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)

class MotorCore:

    def __init__(self):
        self.right_motors = RightMotor(4, 3, 2)
        self.left_motors = LeftMotor(21, 20, 16)


    def drive(self, steering_angle=0, speed=100, time=0):
        if(steering_angle == 0):
            self.right_motors.go_ahead(speed)
            self.left_motors.go_ahead(speed)

        elif (steering_angle < 0):
            l_decreasing_speed = self.calc_speed_increase(speed, steering_angle)
            r_increasing_speed = self.calc_speed_decrease(speed, steering_angle)
            print(l_decreasing_speed)
            print(r_increasing_speed)
            self.right_motors.go_ahead(r_increasing_speed)
            self.left_motors.go_ahead(l_decreasing_speed)

        elif (steering_angle > 0):
            r_decreasing_speed = self.calc_speed_decrease(speed, steering_angle)
            l_increasing_speed = self.calc_speed_increase(speed, steering_angle)
            print(r_decreasing_speed)
            print(l_increasing_speed)
            self.right_motors.go_ahead(r_decreasing_speed)
            self.left_motors.go_ahead(l_increasing_speed)
        sleep(time)

    def stop(self,time=0):
        self.left_motors.stop()
        self.right_motors.stop()
        sleep(time)
    
    def calc_speed_decrease(self, speed, steering_angle):
        return self.check_cycle(speed - speed*abs(steering_angle/100))

    def calc_speed_increase(self, speed, steering_angle):
        print("Vel", speed)
        print("steering angle" , steering_angle)
        print("result", speed + speed*abs(steering_angle))
        return self.check_cycle(speed + speed*abs(steering_angle/100))

    def check_cycle(self, value):
        if (value > 100):
            return 100
        elif (value < 0):
            return 0
        else:
            return value
        

def main():
    motor.stop()
    motor.drive(1, 100, 1)
    motor.stop()
    #motor.drive(0.5, 100, 1)
    GPIO.cleanup()

    
if __name__ == '__main__':
    motor= MotorCore()
    main()




