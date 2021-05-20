import cv2
import numpy as np
import sys

sys.path.append('/home/pi/Desktop/TFG/intelligent-driving-system')

from webcam import WebcamModule
from performance import MotorModule
import RPi.GPIO as GPIO
from keras.models import load_model

GPIO.setmode(GPIO.BCM)

motor_manager = MotorModule.MotorManager()
webcam = WebcamModule

model_trained = load_model('/home/pi/Desktop/TFG/intelligent-driving-system/ia/model.h5')


def process_image_to_predict(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


while True:
    instant_image = webcam.get_img(240, 120)
    instant_image = np.asarray(instant_image)
    instant_image = process_image_to_predict(instant_image)
    steering_predicted = float(model_trained.predict(instant_image))
    print(steering_predicted)
    speed = 100
    motor_manager.drive(steering_predicted, speed)
    cv2.waitKey(1)
