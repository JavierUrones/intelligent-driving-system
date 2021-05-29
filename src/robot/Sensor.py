import RPi.GPIO as GPIO
import time


class Sensor:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        self.trig = 23
        self.echo = 24
        GPIO.setup(self.trig, GPIO.OUT)
        GPIO.setup(self.echo, GPIO.IN)

    def detect_distance(self):
        GPIO.output(self.trig, GPIO.LOW)
        time.sleep(0.5)

        GPIO.output(self.trig, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig, GPIO.LOW)

        while True:
            initial_pulse = time.time()
            if GPIO.input(self.echo) == GPIO.HIGH:
                break
        while True:
            final_pulse = time.time()
            if GPIO.input(self.echo) == GPIO.LOW:
                break

        duration = final_pulse - initial_pulse

        # sound speed = 340m/s
        distance = (34300 * duration) / 2

        return distance

