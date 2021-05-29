import cv2


class Webcam:
    def __init__(self):
        self.height = 500
        self.width = 500
        self.source = cv2.VideoCapture(0)

    def get_image(self):
        x, image = self.source.read()
        image = cv2.resize(image, (self.height, self.width))
    