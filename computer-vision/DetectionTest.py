import numpy as np
import cv2

stop_signal_detector = cv2.CascadeClassifier('/home/pi/Desktop/TFG/intelligent-driving-system/computer-vision/stop-signal/data/stopsign_classifier.xml')

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop_signals = stop_signal_detector.detectMultiScale(gray, 1.3, 5)

    for x,y,w,h in stop_signals:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)
    cv2.imshow('img', img)

cap.release()
cv2.destroyAllWindows()
