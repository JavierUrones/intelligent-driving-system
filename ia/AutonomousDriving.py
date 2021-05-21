import cv2
import numpy as np
import sys

sys.path.append('/home/pi/Desktop/TFG/intelligent-driving-system')

from webcam import WebcamModule
from performance import MotorModule
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite

GPIO.setmode(GPIO.BCM)

motor_manager = MotorModule.MotorManager()
webcam = WebcamModule

interpreter = tflite.Interpreter(model_path='/home/pi/Desktop/TFG/intelligent-driving-system/ia/model.tflite')


interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


print("Input details")
print("shape ", input_details[0]['shape'])
print("Output details")
print("shape ", output_details[0]['shape'])



#model_trained = load_model('/home/pi/Desktop/TFG/intelligent-driving-system/ia/model.h5', compile=False)


def process_image_to_predict(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

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

while True:
    try:
        instant_image = webcam.get_img(240, 120)
        instant_image = process_image_to_predict(instant_image)
        instant_image = np.array([instant_image], dtype=np.float32)
        #steering_predicted = float(model_trained.predict(instant_image))
        interpreter.set_tensor(input_details[0]['index'], instant_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        #steering_predicted = float(interpreter.predict(instant_image))
        print("Steering Predicted"+  output_data)

        print("Steering Predicted After Evaluate" +  evaluate_steering_predicted(output_data))

        speed = 100
        motor_manager.drive(output_data, speed)
        cv2.waitKey(1)
    except KeyboardInterrupt:
        GPIO.cleanup()



