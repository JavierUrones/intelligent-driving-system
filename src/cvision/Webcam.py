import cv2


class Webcam:
    def __init__(self):
        self.height = 480
        self.width = 240
        self.source = cv2.VideoCapture(0)

    def get_image(self):
        x, image = self.source.read()
        image = cv2.resize(image, (self.height, self.width))
        return image
