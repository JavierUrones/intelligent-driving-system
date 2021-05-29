import sys

sys.path.append('/home/pi/Desktop/TFG/intelligent-driving-system')

from src.robot import MotorCore
from src.cvision.Webcam import Webcam

from modulefinder import Module
from time import sleep
import cv2
import RPi.GPIO as GPIO
import numpy as np
import pandas as pd
import os
import cv2
from datetime import datetime
import pygame
import os

class RemoteControl:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        pygame.init()
        window = pygame.display.set_mode((100, 100))
        self.motor_core = MotorCore.MotorCore()
        self.webcam = Webcam()
        self.total_count = 0
        self.list_images = []
        self.list_steering_angle = []
        self.list_speed = []
        self.folder_count = 0
        self.directory = ""
        self.speed = 20

    def init_data_training_collector(self):
        global folder_count, directory
        directory = os.path.join(os.getcwd(), 'training_data')
        while os.path.exists(os.path.join(directory, f'Images{str(folder_count)}')):
            folder_count += 1
        path_to_create = directory + "/Images" + str(folder_count)
        os.makedirs(path_to_create)


    def save_all_data(self, image_to_save, steering_angle_value, speed_value):
        path = directory + "/Images" + str(self.folder_count)
        file_name = os.path.join(path, f'Image_{self.get_timestamp()}.jpg')
        cv2.imwrite(file_name, image_to_save)
        self.list_images.append(file_name)
        self.list_steering_angle.append(steering_angle_value)
        self.list_speed.append(speed_value)

    def get_timestamp(self):
        return str(datetime.timestamp(datetime.now())).replace('.', '')

    def save_csv_file(self):
        raw_data = {'Image': self.list_images, 'Steering_Angle': self.list_steering_angle, 'Speed': self.list_speed}
        print(raw_data)
        df = pd.DataFrame(raw_data)
        df.to_csv(os.path.join(directory, f'log_{str(folder_count)}.csv'), index=False, header=False)
        print('CSV File has been saved')
        print('Number of Images collected: ', len(self.list_images))

    def get_key_pressed(self, keyName):
        flag = False
        for press in pygame.event.get(): pass
        keyInput = pygame.key.get_pressed()
        keyPressed = getattr(pygame, 'K_{}'.format(keyName))
        if keyInput[keyPressed]:
            flag = True
        pygame.display.update()
        return flag

    def start_driving_process(self):
        steering_angle_value = ""
        speed_value = ""
        recording = False
        height, width = 500, 500
        try:
            while True:
                if self.get_key_pressed('t'):
                    print('t was pressed')
                    if recording:
                        image = self.webcam.get_img(height, width)
                        self.save_all_data(image, 0, self.speed)
                        self.motor_core.drive(0, self.speed)
                    else:
                        self.motor_core.drive(0, self.speed)
                if self.get_key_pressed('d'):
                    if recording:
                        image = self.webcam.get_img(height, width)
                        self.save_all_data(image, -100, self.speed)
                        self.core.drive(-100, self.speed)
                    else:
                        self.motor_core.drive(-100, self.speed)
                if self.get_key_pressed('j'):
                    if recording:
                        image = self.webcam.get_img(height, width)
                        self.save_all_data(image, 100, self.speed)
                        self.motor_core.drive(100, self.speed)
                    else:
                        self.motor_core.drive(100, self.speed)
                if self.get_key_pressed("h"):
                    if recording:
                        image = self.webcam.get_img(height, width)
                        self.save_all_data(image, 75, self.speed)
                        self.motor_core.drive(75, self.speed)
                    else:
                        self.motor_core.drive(75, self.speed)
                if self.get_key_pressed("f"):
                    if recording:
                        image = self.webcam.get_img(height, width)
                        self.save_all_data(image, -75, self.speed)
                        self.motor_core.drive(-75, self.speed)
                    else:
                        self.motor_core.drive(-75, self.speed)
                if self.get_key_pressed("s"):
                    if recording:
                        image = self.webcam.get_img(height, width)
                        self.save_all_data(image, 0, 0)
                        self.motor_core.stop()
                    else:
                        self.motor_core.stop()
                if self.get_key_pressed('p'):
                    print("finish...")
                    recording = False
                    self.save_csv_file()
                    GPIO.cleanup()
                    break
                if self.get_key_pressed('r'):
                    print('recording...')
                    recording = True
                    self.init_data_training_collector()
            cv2.waitKey(1)
        except:
            print("finish...")
            recording = False
            self.save_csv_file()
            GPIO.cleanup()

if __name__ == '__main__':
    rc = RemoteControl()
    rc.start_driving_process()
