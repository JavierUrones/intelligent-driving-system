from performance import MotorModule
from webcam import WebcamModule

from modulefinder import Module
from time import sleep
import cv2
import RPi.GPIO as GPIO
import numpy as np
import pandas as pd
import os
import cv2
from datetime import datetime

GPIO.setmode(GPIO.BCM)

motorManager = MotorModule.MotorManager()
webcam = WebcamModule
total_count = 0
list_images = []
list_steering_angle = []
list_speed = []
folder_count = 0
directory = ""


def start_driving_process():
    read_value = ""
    recording = False
    height, width = 500, 500
    while read_value != 0:
        read_value = input()
        if read_value == "w":
            if recording:
                image = webcam.get_img(height, width)
                save_data(image, 0)
                motorManager.drive(0, 100)
            else:
                motorManager.drive(0, 100)
        if read_value == "a":
            if recording:
                image = webcam.get_img(height, width)
                save_data(image, -1)
                motorManager.drive(1, 100)
            else:
                motorManager.drive(1, 100)
        if read_value == "d":
            if recording:
                image = webcam.get_img(height, width)
                save_data(image, 1)
                motorManager.drive(-1, 100)
            else:
                motorManager.drive(-1, 100)
        if read_value == "s":
            if recording:
                image = webcam.get_img(height, width)
                save_data(image, 0)
                motorManager.drive(0, 50)
            else:
                motorManager.drive(0, 50)

        if read_value == "p":
            motorManager.stop()
        if read_value == "1":
            recording = True
            print("now recording")
        if read_value == "2":
            recording = False
            print("stop recording")
        if read_value == "0":
            print("finish...")
            recording = False
            save_csv_file()
            GPIO.cleanup()
            break
        cv2.waitKey(1)


def start_driving_process_2():
    steering_angle_value = ""
    speed_value = ""
    recording = False
    height, width = 500, 500
    while steering_angle_value != "q" or speed_value != "q":
        steering_angle_value = input("Input steering angle")
        speed_value = input("Input steering angle")
        if steering_angle_value == "r" and speed_value == "r":
            recording = True
            init_data_training_collector()
        elif steering_angle_value == "p" and speed_value == "p":
            print("finish...")
            recording = False
            save_csv_file()
            GPIO.cleanup()
            break
        elif recording:
            image = webcam.get_img(height, width)
            save_data(image, 0)
            motorManager.drive(steering_angle_value, speed_value, 1)
        else:
            motorManager.drive(steering_angle_value, speed_value, 1)
    cv2.waitKey(1)


def init_data_training_collector():
    global folder_count, directory
    directory = os.path.join(os.getcwd(), 'training_data')
    while os.path.exists(os.path.join(directory, f'Images{str(folder_count)}')):
        folder_count += 1
    path_to_create = directory + "/Images" + str(folder_count)
    os.makedirs(path_to_create)


def save_data(image_to_save, steering_angle_value):
    list_images, list_steering_angle
    path = directory + "/Images" + str(folder_count)
    file_name = os.path.join(path, f'Image_{get_timestamp()}.jpg')
    cv2.imwrite(file_name, image_to_save)
    list_images.append(file_name)
    list_steering_angle.append(steering_angle_value)


def save_all_data(image_to_save, steering_angle_value, speed_value):
    list_images, list_steering_angle, list_speed
    path = directory + "/Images" + str(folder_count)
    file_name = os.path.join(path, f'Image_{get_timestamp()}.jpg')
    cv2.imwrite(file_name, image_to_save)
    list_images.append(file_name)
    list_steering_angle.append(steering_angle_value)
    list_speed.append(speed_value)


def get_timestamp():
    return str(datetime.timestamp(datetime.now())).replace('.', '')


def save_csv_file():
    raw_data = {'Image': list_images,
                'Steering_Angle': list_steering_angle,
                'Speed': list_speed}
    df = pd.DataFrame(raw_data)
    df.to_csv(os.path.join(directory, f'log_{str(folder_count)}.csv'), index=False, header=False)
    print('CSV File has been saved')
    print('Number of Images collected: ', len(list_images))


if __name__ == '__main__':
    start_driving_process()
