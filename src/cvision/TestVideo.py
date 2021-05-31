import cv2
import VisionManager as vm
import numpy as np
import Webcam as webcam
cap = cv2.VideoCapture('C:\\Users\\javie\OneDrive\\Escritorio\\video4.avi')
vmanager = vm.VisionManager()

def calculate_curve(img):
    mask_img = get_mask(img)
    height, width, _ = img.shape

    #points = np.float32([(102, 80), (width-102, 80), (20, 214), (width-20, 214)])
    points = valTrackbars()
    img_perspective = isolate_image(mask_img, points, width, height)
    cv2.imshow('Mask', mask_img)
    #img_perspective = show_points(img_perspective, points)
    #img_points = show_points(img, points)
    cv2.imshow('points1', img_perspective)

    middle_point, img_hist = get_color_histogram(img_perspective, min_percentage=0.5, region=4)
    base_point, img_hist = get_color_histogram(img_perspective, min_percentage=0.5)
    print("CURVATION", base_point-middle_point)
    cv2.imshow('hist', img_hist)
    # cv2.imshow('points2', img_points)

    curve_val = base_point-middle_point
    list_curve.append(curve_val)
    if len(list_curve) > avgVal:
        list_curve.pop(0)
    avg = sum(list_curve)/len(list_curve)
    curve = int(avg)
    print("CURVE", curve)
    '''widthTransform = 420
    heightTransform = 240
    img_inv_isolated = isolate_image(img_perspective, points, widthTransform, heightTransform )
    img_inv_isolated = cv2.cvtColor(img_inv_isolated, cv2.COLOR_GRAY2BGR)
    img_inv_isolated[0: heightTransform//3, 0: widthTransform] = 0, 0, 0
    img_lane = np.zeros_like(img)
    img_lane[:] = 0, 255, 0
    img_lane = cv2.bitwise_and(img_inv_isolated, img_lane)
    img_result = cv2.addWeighted(img_result, 1, img_lane, 1, 0)
    middle_y = 450'''

    curve = curve * 10
    if curve > 100:
        curve = 100
    if curve < -100:
        curve = -100
    return curve

def get_mask(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sensitivity = 120
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask = cv2.inRange(imgHsv, lower_white, upper_white)
    return mask

def increase_bright(img):
    i_matrix = np.ones(img.shape, dtype='uint8')*60
    return cv2.add(img, i_matrix)

def isolate_image(image, points, width, height, inverse = False):
    points_normal = np.float32(points)
    points_frontal = np.float32([(0,0),(width,0),(0, height), (width, height)])
    if inverse:
        perspective_matrix = cv2.getPerspectiveTransform(points_frontal, points_normal)
    else:
        perspective_matrix = cv2.getPerspectiveTransform(points_normal, points_frontal)
    img_perspective = cv2.warpPerspective(image, perspective_matrix, (width, height))
    return img_perspective

def show_points(image, points):
    for x in range(4):
        cv2.circle(image, (int(points[x][0]), int(points[x, 1])), 15, (0, 0, 255), cv2.FILLED)

        print("POINTS", points)
    return image

def nothing(a):
    pass
def initializeTrackbars(initialTrackbarVals, widthTransform, heightTransform):
    cv2.namedWindow("trackbars")
    cv2.resizeWindow("trackbars", 360, 240)
    cv2.createTrackbar("width top", "trackbars", initialTrackbarVals[0], widthTransform//2, nothing)
    cv2.createTrackbar("height top", "trackbars", initialTrackbarVals[1], heightTransform, nothing)
    cv2.createTrackbar("width bottom", "trackbars", initialTrackbarVals[2], widthTransform//2, nothing)
    cv2.createTrackbar("height bottom", "trackbars", initialTrackbarVals[3], heightTransform, nothing)

def valTrackbars(widthTransform=480, heightTrasformed=240):
    widthTop = cv2.getTrackbarPos("width top", "trackbars")
    heightTop = cv2.getTrackbarPos("height top", "trackbars")

    widthBottom = cv2.getTrackbarPos("width bottom", "trackbars")

    heightBottom = cv2.getTrackbarPos("height bottom", "trackbars")
    points = np.float32([(widthTop, heightTop), (widthTransform-widthTop, heightTop),(widthBottom, heightBottom), (widthTransform-widthBottom,heightBottom)])
    return points

def get_color_histogram(img, min_percentage = 0.1, show=True, region=1):
    if region == 1:
        histogram_Val = np.sum(img, axis=0)
    else:
        image_scaled = img[int(img.shape[0]/region):img.shape[0], 0:img.shape[1]]
        histogram_Val = np.sum(image_scaled, axis= 0)

    #print(histogram_Val)
    max_value = np.max(histogram_Val)
    #print(max_value)
    min_value_path = min_percentage * max_value
    index_array = np.where(histogram_Val>=min_value_path)
    point_base = int(np.average(index_array))
   # print("BASE P", point_base)
    if show:
        img_histogram = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histogram_Val):
            cv2.line(img_histogram, (x, img.shape[0]), (x,img.shape[0]-int(intensity//255//region)), (255, 0, 255), 1)
            cv2.circle(img_histogram, (point_base, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return point_base, img_histogram
    else:
        point_base

initialValuesTrackbar = np.float32([61, 100, 0, 158])
initializeTrackbars(initialValuesTrackbar, 480, 240)
list_curve = []
avgVal = 10
while cap.isOpened():
    ret, frame = cap.read()
    counter = 0

    if ret:
        #vmanager.lane_detection_frame(frame)
        img = cv2.resize(frame, (480, 240))
        calculate_curve(img)
        cv2.imshow('video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()