import csv
import cv2
import numpy as np


def windows_path_modification(data_training_number, check_for_uglies):
    data_training_path = "C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\training_data\\"
    with open(data_training_path + 'log_' + str(data_training_number) + '.csv', newline='') as File:
        reader = csv.reader(File)
        with open(data_training_path + 'log_' + str(data_training_number) + '_path_edited.csv', 'w', newline='') as f:
            w = csv.writer(f)
            remove_counter = 0
            for row in reader:
                print(row)
                values = row[0].split("/")
                row[0] = 'C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\' + values[6] + "\\" \
                         + values[7] + "\\" + values[8] + "\\" + values[9]
                print(row[0])
                if check_for_uglies:
                    value = check_uglies(row[0])
                    if value > 2:
                        w.writerow([row[0], row[1], row[2]])
                    else:
                        remove_counter += 1
                else:
                    w.writerow([row[0], row[1], row[2]])
            if check_for_uglies:
                print("Total uglies images removed: ", remove_counter)


def check_uglies(image):
    image = cv2.imread(image)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_edges = cv2.Canny(image_grey, 50, 150, apertureSize=3)

    image_lines = cv2.HoughLinesP(image_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return len(image_lines)


if __name__ == '__main__':
    check_uglies('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\training_data\\Images703\\Image_162196777841367.jpg')
