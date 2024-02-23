import cv2 as cv
import numpy as np
import os
import random

video = cv.VideoCapture('./img/car_driving_video.mp4')
cv.namedWindow("Video")

while True:
    ret, frame = video.read()
    if not ret:
        print("failed to grab frame")
        break

    k = cv.waitKey(1)

    cap = frame
    cv.imwrite('pic.jpg', cap)
    img = cv.imread('pic.jpg')
    os.remove('pic.jpg')

    cv.imshow('Video', img)

    height = img.shape[0]
    width = img.shape[1]

    triangle = np.array([[(500, int(height*4/5)), (width-500, int(height*4/5)), (int(width/2-100), int(height/2.3)), (int(width/2+50), int(height/2.3))]])
    black_image = np.zeros_like(img)
    mask = cv.fillPoly(black_image, triangle, (255, 255, 255))
    masked_image = cv.bitwise_and(img, mask)
    edged = cv.Canny(masked_image, 300, 400)

    # cv.imshow('edged', edged)
    lines = cv.HoughLinesP(edged, 1.4, np.pi/800, 60, np.array([]), minLineLength=30, maxLineGap=40)

    right_lines = []
    left_lines = []

    right_x1 = 0
    right_y1 = 0
    right_x2 = 0
    right_y2 = 0

    left_x1 = 0
    left_y1 = 0
    left_x2 = 0
    left_y2 = 0

    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # converting to 1d array
            if x1 > width/2 and x2 > width/2:
                right_lines.append([x1, y1, x2, y2])
            elif x1 < width/2 and x2 < width/2:
                left_lines.append([x1, y1, x2, y2])
            else:
                pass

        for item in right_lines:
            right_x1 += int(item[0])
            right_y1 += int(item[1])
            right_x2 += int(item[2])
            right_y2 += int(item[3])

        for item in left_lines:
            left_x1 += int(item[0])
            left_y1 += int(item[1])
            left_x2 += int(item[2])
            left_y2 += int(item[3])

        length_r = len(right_lines)
        length_l = len(left_lines)

        right_x1 /= length_r
        right_y1 /= length_r
        right_x2 /= length_r
        right_y2 /= length_r

        left_x1 /= length_l
        left_y1 /= length_l
        left_x2 /= length_l
        left_y2 /= length_l

        cv.line(img, (int(right_x1), int(right_y1)), (int(right_x2), int(right_y2)), (0, 255, 0), 10)
        cv.line(img, (int(left_x1), int(left_y1)), (int(left_x2), int(left_y2)), (0, 255, 0), 10)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    except:
        print("No lines detected")
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cv.imshow("Video", img)

video.release()
cv.destroyAllWindows()





