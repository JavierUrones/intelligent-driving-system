import cv2
import numpy as np
import os


def pre_process_negatives(directory):
    counter = 0
    images_path = os.path.join(directory, "images_negatives")
    negatives_processed_path = os.path.join(directory, "processed_negatives")
    for image in os.listdir(images_path):
        try:
            load_path = os.path.join(images_path, image)
            print("IMAGE PATH", load_path)
            # It is necessary to convert  negative images to GRAY SCALE and resized them to train haarcascades.
            image = cv2.imread(load_path, cv2.IMREAD_GRAYSCALE)
            image_resized = cv2.resize(image, (100, 100))
            save_path = os.path.join(negatives_processed_path, str(counter) + ".jpg")
            cv2.imwrite(save_path, image_resized)
            counter += 1
        except Exception as e:
            print(str(e))



def create_negatives_file(directory):
    negatives_processed_path = os.path.join(directory, "processed_negatives")
    with open(directory + "\\bg.txt", 'w') as f:
        for image in os.listdir(negatives_processed_path):
            f.write(image+'\n')


def create_positive_file(directory, signal):
    positive_path = os.path.join(directory, signal)
    with open(positive_path + '\\info.dat', 'w') as f:
        for image in os.listdir(os.path.join(positive_path, "positives")):
            data = 'positive\\' + image + ' 1 0 0 50 50\n'
            f.write(data)

computer_vision_directory = "C:\\Users\\javie\\OneDrive\Escritorio\\TFG\\intelligent-driving-system\\computer-vision"
#Step 1 - Process negative files.
#pre_process_negatives(computer_vision_directory)

#Step 2 - create negative description file
#create_negatives_file(computer_vision_directory)

#Step 3 - create positive description file
#create_positive_file(computer_vision_directory, "stop-signal")

#Step 4 - Execute 'opencv_createsamples -img signal.jpg -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1950'