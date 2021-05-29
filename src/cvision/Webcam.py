import cv2


class Webcam:
    def __init__(self):
        self.height = 340
        self.width = 220
        self.source = cv2.VideoCapture(0)

    def get_image(self):
        x, image = self.source.read()
        image = cv2.resize(image, (self.height, self.width))
        return image
