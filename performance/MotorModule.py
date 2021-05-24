from performance import MotorController
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)

class MotorManager():

    def __init__(self):
        self.right_motor_front = MotorController.RightMotor(3, 4 , 12)
        self.right_motor_back = MotorController.RightMotor(20, 21, 13)
        self.left_motor_front = MotorController.LeftMotor(5, 6, 19)
        self.left_motor_back = MotorController.RightMotor(14, 15, 18)




    def drive(self, steering_angle=0, speed=100, time=0.5):
        if(steering_angle == 0):
            self.right_motor_front.go_ahead(speed)
            self.right_motor_back.go_ahead(speed)
            self.left_motor_front.go_ahead(speed)
            self.left_motor_back.go_ahead(speed)

        elif (steering_angle < 0):
            l_decreasing_speed = self.calc_speed_decrease(speed, steering_angle)
            r_increasing_speed = self.calc_speed_increase(speed, steering_angle)
            print(l_decreasing_speed)
            print(r_increasing_speed)
            self.right_motor_front.go_ahead(r_increasing_speed)
            self.right_motor_back.go_ahead(r_increasing_speed)
            self.left_motor_front.go_ahead(l_decreasing_speed)
            self.left_motor_back.go_ahead(l_decreasing_speed)

        elif (steering_angle > 0):
            r_decreasing_speed = self.calc_speed_decrease(speed, steering_angle)
            l_increasing_speed = self.calc_speed_increase(speed, steering_angle)
            print(r_decreasing_speed)
            print(l_increasing_speed)
            self.right_motor_front.go_ahead(r_decreasing_speed)
            self.right_motor_back.go_ahead(r_decreasing_speed)
            self.left_motor_front.go_ahead(l_increasing_speed)
            self.left_motor_back.go_ahead(l_increasing_speed)
        sleep(time)
        self.stop()

    def stop(self,time=0):
        self.right_motor_front.stop()
        self.right_motor_back.stop()
        self.left_motor_front.stop()
        self.left_motor_back.stop()
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
    motor= MotorManager()
    main()




