# from performance import MotorModule
from webcam import WebcamModule

from modulefinder import Module
from time import sleep
import cv2
# import Data as data
# import RPi.GPIO as GPIO
import numpy as np

# GPIO.setmode(GPIO.BCM)

# motorManager = MotorModule.MotorManager()
webcam = WebcamModule


def start_driving_process():
    saved_frame = 0
    total_frame = 0
    start = cv2.getTickCount()
    input_size = 4
    x = np.empty((0, input_size))
    y = np.empty((0, 4))

    read_value = ""
    recording = False
    number_images = 0
    while read_value != 0:
        img_height = 500
        img_width = 500
        img = webcam.get_img()
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width, channels = img.shape

        roi = img[int(height / 2):height, :]

        cv2.imshow('Image', img)

        array_img = img.reshape(1, int(height/2)*width).astype(np.float32)
        number_images += 1
        read_value = input()
        if read_value == "w":
            if recording:
                # img = webcam.getInstantImage()
                # data.saveData(img, 0)
                # motorManager.drive(0, 100)
                print(array_img)
            else:
                # motorManager.drive(0, 100)
                print(array_img)
        if read_value == "a":
            if recording:
                # img = webcam.getInstantImage()
                # data.saveData(img, -1)
                # motorManager.drive(1, 100)
                print(array_img)
            else:
                # motorManager.drive(1, 100)
                print(array_img)
        if read_value == "d":
            if recording:
                # img = webcam.getInstantImage()
                # data.saveData(img, 1)
                # motorManager.drive(-1, 100)
                print(array_img)
            else:
                # motorManager.drive(-1, 100)
                print(array_img)

        if read_value == "s":
            if recording:
                # img = webcam.getInstantImage()
                # data.saveData(img, 0)
                # motorManager.drive(0, 50)
                print(array_img)
            else:
                # motorManager.drive(0, 50)
                print(array_img)

        if read_value == "p":
            # motorManager.stop()
            print("stop")
        if read_value == "1":
            recording = True
            print("now recording")
        if read_value == "2":
            recording = False
            print("stop recording  ")
        if read_value == "0":
            recording = False
            # data.saveLog()
            # GPIO.cleanup()
            break
        cv2.waitKey(1)


if __name__ == '__main__':
    start_driving_process()
