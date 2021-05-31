import math

import cv2
import numpy as np
import Webcam as wb
import os


class VisionManager:
    def __init__(self):
        self.webcam = wb.Webcam()
        self.height = 480
        self.width = 240


    def get_image(self):
        return self.webcam.get_image()
    def show_image(self, title, img):
        cv2.imshow(title, img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

    def lane_detection_frame(self, img):
        #image = cv2.resize(img, (self.height, self.width))
        image = img
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        '''
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([50, 50, 100])
        mask_black = cv2.inRange(gray_image, lower_black, upper_black)
        '''
        edges_image = cv2.Canny(gray_image, 100, 600)
        #self.show_image('a',edges_image)
        roi = self.select_ROI(edges_image)
        self.show_image('roi', roi)
        lines_detected = self.isolate_lines(roi)
        print("line", lines_detected)
        lanes = self.combine_lines_detected(image, lines_detected)
        #cv2.imshow('l_img', roi)
        #cv2.waitKey(0)
        line_color = (0, 255, 0)
        line_width = 10
        if lanes is not None:
            image_line = np.zeros_like(image)
            for line in lanes:
                for x_1, x_2, y_1, y_2 in line:
                    cv2.line(image_line, (x_1, y_1), (x_2, y_2), line_color, line_width)
            l_img = cv2.addWeighted(image, 0.8, image_line, 1, 1)
        #cv2.imshow('l_img', l_img)
        #cv2.waitKey(0)
        steer = self.calculate_steering_angle(image, lanes)
        if steer == None:
            steer = 0
        print(steer)
        medium = self.show_medium_line(image, steer)
        cv2.imshow('med', medium)
        #cv2.waitKey(0)'''
        return steer

    def select_ROI(self, image):
        roi = image[int(image.shape[0]/2):image.shape[0], 0:image.shape[1]]
        return roi

    def isolate_lines(self, image):
        rho = 1
        angle_radians = np.pi / 180
        threshold = 10
        line_segments = cv2.HoughLinesP(image, rho, angle_radians, threshold, np.array([]), minLineLength=8, maxLineGap=4)
        return line_segments

    def combine_lines_detected(self, image, lines_coordinates):
        lanes = []
        left_points = []
        right_points = []
        if lines_coordinates is not None:
            region_left = image.shape[1] * 2/3
            region_right = image.shape[1] * 1/3

            for line in lines_coordinates:
                for x_1, y_1, x_2, y_2 in line:
                    fit = np.polyfit((x_1, x_2), (y_1, y_2), 1)
                    if fit[0] < 0:
                        if x_1 < region_left and x_2 < region_left:
                            left_points.append((fit[0], fit[1]))
                    else:
                        if x_1 > region_right and x_2 > region_right:
                            right_points.append((fit[0], fit[1]))
            left_points_avg = np.average(left_points, axis=0)
            if len(left_points) > 0:
                lanes.append(self.calculate_points(image, left_points_avg))
            right_points_avg = np.average(right_points, axis=0)
            if len(right_points) > 0:
                lanes.append(self.calculate_points(image, right_points_avg))
            else:
                print("No lines detected")
                return []
            return lanes


    def calculate_points(self, image, points_avg):
        print("POINTS", points_avg)
        y_1 = image.shape[0]
        y_2 = int(y_1*1/2)
        x_1 = max(-image.shape[1], min(2*image.shape[1], int((y_1 - points_avg[1])/points_avg[0])))
        x_2 = max(-image.shape[1], min(2*image.shape[1], int((y_2 - points_avg[1])/points_avg[0])))
        return [[x_1, x_2, y_1, y_2]]

    def calculate_steering_angle(self, image, lanes):
        if len(lanes) == 0:
            print("No lanes detected")
            return None

        elif(len(lanes)) == 1:
            print("aqui")
            x_1_initial, _, x_2_final, _ = lanes[0][0]
            x_med = x_2_final - x_1_initial
        else:
            print("LANES: ", lanes)
            _, _, x_2_left, _ = lanes[0][0]
            _, _, x_2_right, _ = lanes[1][0]
            camera_calibration_percent = 0.0
            medium_line = int(image.shape[1] / 2 * (1 + camera_calibration_percent))
            x_med = (x_2_left + x_2_right) / 2 - medium_line
        y_med = int(image.shape[0] / 2)
        angle = math.atan(x_med / y_med)
        angle_deg = int(angle * 180 / math.pi)
        steering = angle_deg
        return steering

    def show_medium_line(self, image, steering):
        medium = np.zeros_like(image)
        steering_angle = steering / 180 * math.pi
        if steering_angle < 0:
            steering_angle = steering_angle + 90
        elif steering_angle > 0:
            steering_angle = steering_angle - 90
        sum_factor = 0.1
        if steering_angle == 0:
            steering_angle += sum_factor
        x1 = int(image.shape[1]/2)
        y1 = image.shape[0]
        x2 = int(x1-image.shape[0]/2 / math.tan(steering_angle))
        y2 = int(image.shape[0] / 2)
        cv2.line(medium, (x1, y1), (x2, y2), (0,0,255), 5)
        medium = cv2.addWeighted(image, 0.8, medium, 1, 1)
        return medium
    def transform_image(self):
        SIZE = 500
        img = cv2.imread("img-test.jpg")
        img = cv2.resize(img, (SIZE, SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #detect edges

        edges = cv2.Canny(gray, 150, 300)

        #mask

        mask = np.zeros(img.shape[:2], dtype ="uint8")

        pts = np.array([[0, SIZE*3/4], [SIZE/2, SIZE/2], [SIZE, SIZE*3/4], [SIZE, SIZE], [0, SIZE]], np.int32)

        #pts = np.array([[SIZE /2, SIZE/2], [0, SIZE], [SIZE, SIZE]], np.int32)
        pts2 = np.array([[SIZE /2, SIZE/2], [0, SIZE], [SIZE, SIZE]], np.int32)

        cv2.fillPoly(mask, [pts], 255)
        cv2.fillPoly(mask, [pts2], 0)

        self.show_image("mask", mask)

        #lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta = np.pi/180,
            threshold=20, minLineLength = 30,
            maxLineGap= 10
            )
        print(lines)

        #draw lines

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
        line_color = [0, 255, 0]
        line_thickness = 1
        dot_color = [0, 255, 0]
        dot_size = 3

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_thickness)
                cv2.circle(line_img, (x1, y1), dot_size, dot_color, -1)
                cv2.circle(line_img, (x2, y2), dot_size, dot_color, -1)
        line_img = cv2.bitwise_and(line_img, line_img, mask = mask)

        overlay = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
        self.show_image("overlay", overlay)


if __name__ == '__main__':
    v = VisionManager()
    #img = cv2.imread('image.jpg')

    ejemplo_dir = 'pictures'
    contenido = os.listdir(ejemplo_dir)
    imagenes = []
    for fichero in contenido:
        print(fichero)
        im = cv2.imread(ejemplo_dir + "\\" + fichero)
        im = cv2.resize(im, (220, 340))
        v.lane_detection_frame(im)