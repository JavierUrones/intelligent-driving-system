import RPi.GPIO as GPIO
import abc
from time import sleep


class Motor(metaclass=abc.ABCMeta):
    def __init__(self, in1, in2, en):
        self.in1 = in1
        self.in2 = in2
        self.en = en
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.en, GPIO.OUT)
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        self.pwm = GPIO.PWM(self.en, 100)
        self.pwm.start(0)

    def go_ahead(self, speed):
        print("go ahead", vars(self))

        # Se establece el sentido de giro con los pines in1 e in2
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)

    def go_back(self, speed):
        print("go back", vars(self))
        # Se establece el sentido de giro con los pines in1 e in2
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.HIGH)
        self.pwm.ChangeDutyCycle(speed)

    def stop(self):
        print("stop", vars(self))
        self.pwm.ChangeDutyCycle(0)

    @abc.abstractmethod
    def turn_right(self, speed):
        pass

    @abc.abstractmethod
    def turn_left(self, speed):
        pass


@Motor.register
class RightMotor(Motor):

    def turn_left(self, speed):
        print("right motor turn right", vars(self))
        # Establecemos el sentido de giro con los pines IN1 e IN2
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)

    def turn_right(self, speed):
        print("right motor turn left", vars(self))
        # Establecemos el sentido de giro con los pines IN1 e IN2
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)


@Motor.register
class LeftMotor(Motor):

    def turn_left(self, speed):
        print("left motor turn right", vars(self))
        # Establecemos el sentido de giro con los pines IN1 e IN2
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)

    def turn_right(self, speed):
        print("left motor turn left", vars(self))
        # Establecemos el sentido de giro con los pines IN1 e IN2
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)


def main():
    front_right = RightMotor(4, 3, 2)
    back_right = RightMotor(21, 20, 16)
    #back_left = LeftMotor(13, 6, 5)
   # front_left = LeftMotor(23, 24, 12)
    # back_left.go_ahead(100)
    # front_left.go_ahead(100)
    back_right.go_ahead(50)
    front_right.go_ahead(50)

    sleep(2)
    # back_left.stop()
    # front_left.stop()
    back_right.stop()
    front_right.stop()
    sleep(2)
#         front_left.go_ahead(100)
#         sleep(2)
#         front_left.stop()
#         sleep(2)


if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)

    main()
    GPIO.cleanup()
