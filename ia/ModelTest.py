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
    image = image[54:120, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
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


model_trained = load_model('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\model.h5')
img =  mpimage.imread('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\training_data\\Images458\\Image_162161605527838.jpg')
img = np.asarray(img)
img = pre_training_process(img)
img = np.array([img])
steering = float(model_trained.predict(img))
print("Steering Predicted" , steering)
print("Steering Predicted after evaluate" , evaluate_steering_predicted(steering))

