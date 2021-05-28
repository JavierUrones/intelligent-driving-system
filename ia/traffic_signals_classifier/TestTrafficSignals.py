
import numpy as np

import cv2
import pickle
import tflite_runtime.interpreter as tflite

def pre_process(image):
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255
    return image

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.90

font = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

interpreter = tflite.Interpreter(model_path='/home/pi/Desktop/TFG/intelligent-driving-system/ia/traffic_signals_classifier/model_sign_detector.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


label_names = open('/home/pi/Desktop/TFG/intelligent-driving-system/ia/signals-detector/signnames.csv')
list_labels = []
for label in label_names:
    list_labels.append(label.split(",")[1])
print("label names", list_labels)


while True:

    success, image_taken = cap.read()
    image = np.asarray(image_taken)
    image = pre_process(image)
    cv2.imshow("proccesed image", image)
    image = image.reshape(1, 32, 32, 1)
    cv2.putText(image_taken, "CLASS : " , (20, 35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(image_taken, "PROBABILITY : " , (20, 75), font, 0.75, (255,0,0), 2, cv2.LINE_AA)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    print("predicting...")
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    print("Predicted: ", output_data)
    predicted_value = output_data.argmax()
    print("Predicted value cleaned:", predicted_value)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break