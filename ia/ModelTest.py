from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from imgaug import augmenters as img_augmenters
import matplotlib.image as mpimage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


def pre_training_process(image):
    image = image[250:500, 0:500]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image
def pre_training_process_2(image):
    image = image[250:500, 0:500]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    kernel_size = 5
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image
def evaluate_steering_predicted(steering_predicted_value):
    if -25 < steering_predicted_value < 25:
        return 0
    elif 25<= steering_predicted_value < 65:
        return 50
    elif steering_predicted_value >= 65:
        return 100
    elif -25 >= steering_predicted_value > -65:
        return -50
    elif steering_predicted_value <= -65:
        return -100

import csv
model_trained = load_model('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\model.h5')
with open('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\training_data\log_1_path_edited.csv', newline='') as File:
    reader = csv.reader(File)
    for row in reader:
        i2 = cv2.imread(row[0])
        cv2.imshow('img', i2)
        cv2.waitKey(0)
        img = mpimage.imread(row[0])
        img = np.asarray(img)
        img = pre_training_process_2(img)
        img = np.array([img])
        steering = float(model_trained.predict(img))
        print("Steering Predicted" , steering)
        print("Steering Predicted after evaluate" , evaluate_steering_predicted(steering))

