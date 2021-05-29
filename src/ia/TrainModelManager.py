from random import shuffle

from sklearn.model_selection import train_test_split

import TrainModelUtils as utils
import matplotlib.pyplot as plt


class TrainModelManager:

    def __init__(self, model_name, folder_path):
        self.train_model_utils = utils.TrainModelUtils()
        self.model_name = model_name
        self.folder_path = folder_path

    def start_training_process(self, data_folder, path_data):
        path = path_data + "\\" + data_folder

        # PathCsvConverter.windows_path_modification(1, True)
        data_info = utils.load_data_training(data_folder, 1, path)

        print(data_info.head())

        utils.show_data_loaded(data_info)

        data_info = utils.remove_highly_repeated(data_info)
        utils.show_data_loaded(data_info)

        images_routes, steering_angles = utils.load_images(path, data_info)

        # 20% validation, 80% training
        x_train, x_val, y_train, y_val = train_test_split(images_routes, steering_angles, test_size=0.2, random_state=5)
        x_train, y_train = shuffle(x_train, y_train)
        print("Images Training", len(x_train))
        print("Images Validation", len(x_val))

        model = utils.create_model()

        trained_model = model.fit(utils.generation_data_for_training(x_train, y_train, 50, 1), steps_per_epoch=100,
                                  epochs=100,
                                  validation_data=utils.generation_data_for_training(x_val, y_val, 25, 0),
                                  validation_steps=25)

        self.plotResults(trained_model)
        self.save_model(trained_model)

    def plotResults(trained_model):
        plt.plot(trained_model.history['loss'])
        plt.plot(trained_model.history['val_loss'])
        plt.legend(['Training', 'Validation'])
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.show()
        plt.plot(trained_model.history['accuracy'], label='accuracy')
        plt.plot(trained_model.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def save_model(model):
        model.save("model.h5")
        print("Model .h5 saved")

        # Convert to tensorflow lite model to use it in Raspberry pi
        from tensorflow import lite

        converter = lite.TFLiteConverter.from_keras_model(model)
        model_tensorflow_lite = converter.convert()
        open("model.tflite", "wb").write(model_tensorflow_lite)
        print("Model .tflite saved")


if __name__ == '__main__':
    trainer = TrainModelManager()
    trainer.start_training_process("training_data",
                                   "C:\\Users\\javie\\OneDrive\\Escritorio\\TFG\\intelligent-driving-system\\ia")
