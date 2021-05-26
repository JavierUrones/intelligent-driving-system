import imutils
import cv2
import numpy as np
import sys
from skimage import exposure, transform, io
sys.path.append('/home/pi/Desktop/TFG/intelligent-driving-system')
from webcam import WebcamModule
import tflite_runtime.interpreter as tflite

webcam = WebcamModule
interpreter = tflite.Interpreter(model_path='/home/pi/Desktop/TFG/intelligent-driving-system/ia/signals-detector/pyimagesearch/trafficsign.tflite')
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
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
	try:
		print("taking picture...")
		sucess,instant_image =  cap.read()
		image_taken = instant_image
		print("image?: ", sucess)
		cv2.imshow("image", instant_image)
		#instant_image = webcam.get_img(32, 32)
		instant_image = transform.resize(instant_image, (32, 32))
		instant_image = exposure.equalize_adapthist(instant_image, clip_limit=0.1)
		instant_image = instant_image.astype("float32") / 255.0
		instant_image = np.expand_dims(instant_image, axis=0)
		interpreter.set_tensor(input_details[0]['index'], instant_image)
		interpreter.invoke()
		print("predicting...")
		output_data = interpreter.get_tensor(output_details[0]['index'])[0]
		print("Predicted: ", output_data)
		predicted_value = output_data.argmax()
		print("Predicted value cleaned:" , predicted_value)
		label = list_labels[predicted_value]
		print("img shape", instant_image.shape)
		width = int(480)
		height = int(640)
		dim = (width, height)
		image_to_show = cv2.resize(image_taken, dsize = dim)
		cv2.putText(image_to_show, label, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
		cv2.imshow("image", image_to_show)
		if cv2.waitKey(1) and 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()	
			break
	except KeyboardInterrupt:
		cap.release()
		cv2.destroyAllWindows()
		sys.exit()