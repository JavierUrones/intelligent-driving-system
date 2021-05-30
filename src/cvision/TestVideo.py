import cv2
import VisionManager as vm
cap = cv2.VideoCapture('C:\\Users\\javie\OneDrive\\Escritorio\\video3.avi')
vmanager = vm.VisionManager()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        vmanager.lane_detection_frame(frame)

        #cv2.imshow('video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()