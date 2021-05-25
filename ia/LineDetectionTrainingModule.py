import os

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
from utils import PathCsvConverter


def load_data_training(folder, loading_number, origin_path):
    columns_titles = ['Image', 'Steering_Angle', 'Speed']
    data_frame = pd.DataFrame()
    data = pd.read_csv(origin_path + '\\log_' + str(loading_number) + '_path_edited.csv', names=columns_titles)
    data_frame = data_frame.append(data, True)
    return data_frame



def show_data_loaded(data_loaded):
    figure, ax1 = plt.subplots(1, 1)
    ax1.hist(data_loaded['Steering_Angle'], bins=[-100, -50, 0, 50, 100])
    ax1.set_title("Data Steering Angle")
    ax1.set_xticks([-100, -50, 0, 50, 100])
    ax1.set_xlabel('SteeringAngle')
    ax1.set_ylabel('Number')
    plt.show()
    y_coordinates, x_coordinates = np.histogram(data_loaded['Steering_Angle'], 30)
    figure_2, ax2 = plt.subplots()
    ax2.plot(x_coordinates[:-1], y_coordinates)
    figure_2.show()


def remove_highly_repeated(data_loaded):
    margin_repetition_value = 20  # Maximum number of samples with the same steering angle value
    steering_angle_values = {}
    index_list = {}
    # Creating a dictionary with the number of repetitions for each Steering Angle value.
    # Also creating a dictionary with the index of the values for each Steering Angle value
    for i in range(0, len(data_loaded['Steering_Angle'])):
        if data_loaded['Steering_Angle'][i] not in steering_angle_values:
            steering_angle_values[data_loaded['Steering_Angle'][i]] = 1
            index_list[data_loaded['Steering_Angle'][i]] = []
            index_list[data_loaded['Steering_Angle'][i]].append(i)
        else:
            steering_angle_values[data_loaded['Steering_Angle'][i]] += 1
            index_list[data_loaded['Steering_Angle'][i]].append(i)

    print("Data Info Pre-Remove", data_loaded)

    # print("Index List", index_list)
    print(steering_angle_values)

    # The index list of each value is shuffled to avoid removing concrete values.
    for element in index_list.keys():
        index_list[element] = shuffle(index_list[element])
    # print("Index List Shuffle:", index_list)

    index_to_remove = []
    # If one steering angle value has more appareances than the limit (margin_repetition_value), the difference are calculated
    # and that amount of indexes are removed in the loaded data frame.
    for element in steering_angle_values.keys():
        if steering_angle_values[element] > margin_repetition_value:
            dif = steering_angle_values[element] - margin_repetition_value
            for i in range(0, dif):
                index_to_remove.append(index_list[element].pop(0))

    # print("Index List Removed", index_list)
    print("Index To Remove", index_to_remove)
    data_loaded.drop(data_loaded.index[index_to_remove], inplace=True)
    # print("Data Info Post-Remove", data_loaded)
    return data_loaded


def load_images(origin_path, data_loaded):
    images_routes_list = []
    steering_angle_list = []
    for i in range(0, len(data_loaded)):
        data_on_index = data_loaded.iloc[i]
        images_routes_list.append(data_on_index[0])
        steering_angle_list.append(float(data_on_index[1]))
    return images_routes_list, steering_angle_list


def increase_images(route_img, steering_angle):
    image_to_transform = mpimage.imread(route_img)
    random_value = np.random.rand()
    if random_value <= 0.25:
        scaling_image = img_augmenters.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        image_to_transform = scaling_image.augment_image(image_to_transform)
    elif 0.5 >= random_value > 0.25:
        zoom_image = img_augmenters.Affine(scale=(1.2, 1.2))
        image_to_transform = zoom_image.augment_image(image_to_transform)
    if 0.75 >= random_value > 0.5:
        contrast_image = img_augmenters.GammaContrast((0.5, 1.5))
        image_to_transform = contrast_image.augment_image(image_to_transform)
    if random_value > 0.75:
        image_to_transform = cv2.flip(image_to_transform, 1)
        steering_angle = -steering_angle
    return image_to_transform, steering_angle


def pre_training_process(image):
    image = image[54:120, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image


def create_model():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model


def generation_data_for_training(images_routes_list, steering_angle_list, batch_size, train_flag):
    while True:
        image_batch_list = []
        steering_batch_list = []

        for i in range(batch_size):
            index = np.random.randint(0, len(images_routes_list) - 1)
            if train_flag:
                image, steering = increase_images(images_routes_list[index], steering_angle_list[index])
            else:
                image = mpimage.imread(images_routes_list[index])
                steering = steering_angle_list[index]
            image = pre_training_process(image)
            image_batch_list.append(image)
            steering_batch_list.append(steering)
        yield np.asarray(image_batch_list), np.asarray(steering_batch_list)

data_folder = "training_data"
path = "C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\" + data_folder


#PathCsvConverter.windows_path_modification(470, True)
data_info = load_data_training(data_folder, 470, path)

print(data_info.head())


show_data_loaded(data_info)

#data_info = remove_highly_repeated(data_info)
images_routes, steering_angles = load_images(path, data_info)
print(images_routes)
print(steering_angles)

# 30% validation, 70% training
x_train, x_val, y_train, y_val = train_test_split(images_routes, steering_angles, test_size=0.3, random_state=5)

print("Images Training", x_train)
print("Images Validation", x_val)

print("Steering Angles Training", y_train)
print("Steering Angles Validation", y_val)

model = create_model()

trained_model = model.fit(generation_data_for_training(x_train, y_train, 128, 1), steps_per_epoch=256, epochs=40,
                          validation_data=generation_data_for_training(x_val, y_val, 64, 0), validation_steps=128)

plt.plot(trained_model.history['loss'])
plt.plot(trained_model.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

model.save("model.h5")
print("Model .h5 saved")

# Convert model to a TensorFlow-Lite model to use in Raspberry Pi
from tensorflow import lite
converter = lite.TFLiteConverter.from_keras_model(model)
model_tensorflow_lite = converter.convert()
open("model.tflite","wb").write(model_tensorflow_lite)
print("Model .tflite saved")
'''
model_trained = load_model('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\model.h5')
img =  mpimage.imread('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\training_data\\Images415\\Image_1621523407695866.jpg')
img = np.asarray(img)
img = pre_training_process(img)
img = np.array([img])
steering = float(model_trained.predict(img))
print(steering)'''

