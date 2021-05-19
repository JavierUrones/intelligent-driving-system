import cv2


def get_img(height, width):
    capture = cv2.VideoCapture(-1)
    x, image = capture.read()
    image = cv2.resize(image, (height, width))
    return image
