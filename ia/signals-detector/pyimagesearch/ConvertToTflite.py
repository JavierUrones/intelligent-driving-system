from tensorflow import lite
from tensorflow.keras.models import load_model

def convert_to_tflite():
    # Convert model to a TensorFlow-Lite model to use in Raspberry Pi
    model = load_model('C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia\\signals-detector\\output\\trafficsignnet.model')
    converter = lite.TFLiteConverter.from_keras_model(model)
    model_tensorflow_lite = converter.convert()
    open("trafficsign.tflite", "wb").write(model_tensorflow_lite)
    print("TrafficSign Model .tflite saved")

convert_to_tflite()