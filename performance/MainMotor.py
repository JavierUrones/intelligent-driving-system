from modulefinder import Module
from time import sleep
from src import MotorManager, WebcamManager as webcam
import cv2
import Data as data
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

motorManager = MotorManager()


def startDrivingProcess():

    readValue = ""
    recording = False
    while(readValue != 0):
        readValue = input()
        if(readValue == "w"):
            if(recording):
                img = webcam.getInstantImage()
                data.save_data(img, 0)
                motorManager.drive(0, 90)
            else:
                motorManager.drive(0, 90)
        if(readValue == "a"):
            if(recording):
                img = webcam.getInstantImage()
                data.save_data(img, -1)
                motorManager.drive(1, 90)
            else:
                motorManager.drive(1, 90)
        if(readValue == "d"):
            if(recording):
                img = webcam.getInstantImage()
                data.save_data(img, 1)
                motorManager.drive(-1, 90)
            else:
                motorManager.drive(-1, 90)

        if(readValue == "s"):
            if(recording):
                img = webcam.getInstantImage()
                data.save_data(img, 0)
                motorManager.drive(0, 50)
            else:
                motorManager.drive(0, 50)

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
