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
            f.write(negatives_processed_path + "\\" + image + '\n')


def create_positive_file(directory, signal):
    positive_path = os.path.join(directory, signal)
    with open(positive_path + '\\info.dat', 'w') as f:
        for image in os.listdir(os.path.join(positive_path, "positives")):
            data = 'positive\\' + image + ' 1 0 0 50 50\n'
            f.write(positive_path + "\\" + data)


computer_vision_directory = "C:\\Users\\javie\\OneDrive\Escritorio\\TFG\\intelligent-driving-system\\computer-vision"
# Step 1 - Process negative files.
# pre_process_negatives(computer_vision_directory)

# Step 2 - create negative description file
# create_negatives_file(computer_vision_directory)

# Step 3 - create positive description file
create_positive_file(computer_vision_directory, "green-traffic-light")

# Step 4 - Execute opencv_createsamples and opencv_traincascade
# C:\Users\javie\Downloads\opencv\build\x64\vc14\bin\opencv_createsamples.exe -img stop-signal\positives\stop-signal-image.jpg -bg bg.txt -info stop-signal\info\info.lst -pngoutput stop-signal\info -maxxangle 0.5 -maxyangle -0.5 -maxzangle 0.5 -num 270
# C:\Users\javie\Downloads\opencv\build\x64\vc14\bin\opencv_createsamples.exe -info stop-signal\info\info.lst -num 270 -w 20 -h 20 -vec stop-signal\positives\positives.vec
# C:\Users\javie\Downloads\opencv\build\x64\vc14\bin\opencv_traincascade.exe -data stop-signal\data -vec stop-signal\positives\positives.vec -bg bg.txt -numPos 500 -numNeg 250 -numStages 15 -w 20 -h 20
