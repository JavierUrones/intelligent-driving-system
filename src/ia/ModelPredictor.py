from tensorflow.keras.models import load_model
import TrainModelUtils as tmu
import numpy as np
import matplotlib.image as mpimage


class ModelPredictor:

    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.utils = tmu.TrainModelUtils()

    def predict(self, image):
        img = mpimage.imread(image)
        img = np.asarray(img)
        img = self.utils.pre_training_process_2(img)
        img = np.array([img])
        steering = float(self.model.predict(img))
        print(steering)

