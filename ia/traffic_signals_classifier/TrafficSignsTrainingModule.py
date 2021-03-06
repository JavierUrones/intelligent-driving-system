import random
import csv

import numpy as np
import pandas as pd
import cv2
from skimage import exposure
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, \
    Dropout, Convolution2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def load_data(path_dataset, csv_path):
    images_list = []
    class_label_list = []
    rows = open(csv_path).read().strip().split("\n")[1:]
    random.shuffle(rows)
    counter = 0
    for (i, row) in enumerate(rows):
        row = row.split(',')
        class_label = row[6]
        image_path = path_dataset + "\\" + row[7].replace('/', '\\')
        image = pre_process_images(image_path)
        images_list.append(image)
        class_label_list.append(int(class_label))
        if counter % 1000 == 0:
            print("Total images loaded", counter)
        counter += 1
    return np.asarray(images_list), np.asarray(class_label_list)


def pre_process_images(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255
    return image

def create_model(num_classes):
    '''model_conv = Sequential()
    model_conv.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model_conv.add(MaxPooling2D(pool_size=(2, 2)))
    model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model_conv.add(MaxPooling2D(pool_size=(2, 2)))
    model_conv.add(BatchNormalization())
    model_conv.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model_conv.add(MaxPooling2D(pool_size=(2, 2)))
    model_conv.add(BatchNormalization())
    model_conv.add(Dropout(0.25))
    model_conv.add(Dropout(0.5))
    model_conv.add(Flatten())
    model_conv.add(Dense(128, activation='relu'))
    model_conv.add(Dropout(0.5))
    model_conv.add(Dense(num_classes, activation='softmax'))
    return model_conv'''
    filters_n = 60
    size_filter = (5, 5)
    size_filter_2 = (3,3)
    size_pool = (2,2)
    n_nodes = 500
    model = Sequential()
    model.add((Convolution2D(filters_n, size_filter, input_shape=(32, 32, 1), activation = 'relu')))
    model.add((Convolution2D(filters_n, size_filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))

    model.add((Convolution2D(filters_n//2, size_filter_2, activation='relu')))
    model.add((Convolution2D(filters_n, size_filter_2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_labels, activation="softmax"))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model




images, classes = load_data(
    'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\traffic_signals_classifier\\gtsrb-german-traffic-sign',
    'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\traffic_signals_classifier\\gtsrb-german-traffic-sign\\Train.csv')

x_train, x_test, y_train, y_test = train_test_split(images, classes, test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

labels = pd.read_csv('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\traffic_signals_classifier\\signnames.csv')


'''
x_train, y_train = load_data(
    'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\traffic_signals_classifier\\gtsrb-german-traffic-sign',
    'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\traffic_signals_classifier\\gtsrb-german-traffic-sign\\Train.csv')

x_val, y_val = load_data(
    'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\traffic_signals_classifier\\gtsrb-german-traffic-sign',
    'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\traffic_signals_classifier\\gtsrb-german-traffic-sign\\Test.csv')
'''
number_of_labels = len(np.unique(y_train))

print("Number of training examples: ", len(x_train))
print("Number of validation examples: ", len(x_val))
print("Image data shape =", x_train[0].shape)
print("Number of classes", number_of_labels)


'''
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
y_train = to_categorical(y_train, number_of_labels)
y_val = to_categorical(y_val, number_of_labels)
'''
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
'''
total_images_per_class = y_train.sum(axis=0)
class_weight = dict()
for i in range(0, len(total_images_per_class)):
    class_weight[i] = total_images_per_class.max() / total_images_per_class[i]
'''
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15)


aug.fit(x_train)

batches = aug.flow(x_train, y_train, batch_size=20)
x_batch, y_batch = next(batches)

y_train = to_categorical(y_train, number_of_labels)
y_test = to_categorical(y_test, number_of_labels)

y_val = to_categorical(y_val, number_of_labels)

model = create_model(number_of_labels)
history = model.fit_generator(aug.flow(x_train, y_train, batch_size = 50), steps_per_epoch=130, epochs=30, validation_data=(x_val, y_val))
'''print("Compiling model.")

EPOCHS = 30
LEARNING_RATE = 0.001
BATCH_SIZE = 64

model = create_model(number_of_labels)
opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / (EPOCHS))
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print("[INFO] training network...")
history = model.fit(
    aug.flow(x_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(x_val, y_val),
    steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight,
    verbose=1)
'''

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
model.save("model_sign_detector.h5")
print("Model .h5 saved")


model_trained = load_model('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\traffic_signals_classifier\\model_sign_detector.h5')

from tensorflow import lite

converter = lite.TFLiteConverter.from_keras_model(model_trained)
model_tensorflow_lite = converter.convert()
open("model_sign_detector.tflite", "wb").write(model_tensorflow_lite)
print("Model .tflite saved")
