from modulefinder import Module
from time import sleep
from src import MotorManager, WebcamManager as webcam
import cv2
import Data as data
import RPi.GPIO as GPIO
import numpy as np
GPIO.setmode(GPIO.BCM)

motorManager = MotorManager()


def startDrivingProcess():

    saved_frame = 0
    total_frame = 0
    start = cv2.getTickCount()
    input_size = 4
    x = np.empty((0, input_size))
    y = np.empty((0, 4))

    readValue = ""
    recording = False
    numberOfImages = 0
    while(readValue != 0):
        img_height = 500
        img_width = 500
        img = webcam.getImg(500, 500)
        transformed_image = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Image', transformed_image)

        array_img = transformed_image.reshape(1, int(img_height/2)*img_width).astype(np.float32)
        numberOfImages+=1
        readValue = input()
        if(readValue == "w"):
            if(recording):
                #img = webcam.getInstantImage()
                #data.saveData(img, 0)
                #motorManager.drive(0, 100)
                print(array_img)
            else:
                #motorManager.drive(0, 100)
                print(array_img)
        if(readValue == "a"):
            if(recording):
                #img = webcam.getInstantImage()
                #data.saveData(img, -1)
                #motorManager.drive(1, 100)
                print(array_img)
            else:
                #motorManager.drive(1, 100)
                print(array_img)
        if(readValue == "d"):
            if(recording):
                #img = webcam.getInstantImage()
                #data.saveData(img, 1)
                #motorManager.drive(-1, 100)
                print(array_img)
            else:
                #motorManager.drive(-1, 100)
                print(array_img)

        if(readValue == "s"):
            if(recording):
                #img = webcam.getInstantImage()
                #data.saveData(img, 0)
                #motorManager.drive(0, 50)
                print(array_img)
            else:
                #motorManager.drive(0, 50)
                print(array_img)

        if(readValue == "p"):
            motorManager.stop()
        if(readValue == "1"):
            recording = True
            print("now recording")
        if(readValue == "2"):
            recording = False
            print("stop recording  ")
        if(readValue == "0"):
            recording = False
            data.saveLog()
            GPIO.cleanup()
            break
        cv2.waitKey(1)


if __name__ == '__main__':
    startDrivingProcess()
