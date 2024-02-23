import numpy as np
from mss import mss
import cv2 as cv
import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

driver = webdriver.Safari()
driver.get("https://gamaverse.com/one-more-lap-game/#google_vignette")
action = ActionChains(driver)

action.click(on_element=driver.find_element(By.CLASS_NAME, "fc-cta-consent"))
action.click(on_element=driver.find_element(By.CLASS_NAME, "game-preloader-play-button"))
action.click(on_element=driver.find_element(By.CLASS_NAME, "go-fullscreen"))
action.perform()
time.sleep(6)

action.move_to_element(driver.find_element(By.TAG_NAME, "html")).move_by_offset(100, 40).click().perform()
time.sleep(2)

for i in range(200):
    action.send_keys("w").perform()
time.sleep(0.1)

bounding_box = {'top': 0, 'left': 0, 'width': 1425, 'height': 900}

sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    frame = np.array(sct_img)

    k = cv.waitKey(1)

    img = frame

    cv.imshow('Video', img)

    height = img.shape[0]
    width = img.shape[1]

    triangle = np.array([[(1250, int(height/1.5) - 50), (1600, int(height/1.5) - 50), (1600, int(height/1.9)), (1250, int(height/1.9))]])
    black_image = np.zeros_like(img)
    mask = cv.fillPoly(black_image, triangle, (255, 255, 255))
    masked_image = cv.bitwise_and(img, mask)

    edged = cv.Canny(masked_image, 300, 300)

    cv.imshow('edged', edged)
    lines = cv.HoughLinesP(edged, 1.4, np.pi/800, 30, np.array([]), minLineLength=110, maxLineGap=40)

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

        cv.line(img, (int(right_x1), int(right_y1)), (int(right_x2), int(right_y2)), (0, 0, 255), 10)
        cv.line(img, (int(left_x1), int(left_y1)), (int(left_x2), int(left_y2)), (0, 0, 255), 10)

        if int(left_x1 - left_x2) < 0:
            for i in range(40):
                action.send_keys("d").perform()
            time.sleep(0.1)
            for i in range(20):
                action.send_keys("w").perform()
            time.sleep(0.1)
        if int(right_x1 - right_x2) < 0:
            for i in range(40):
                action.send_keys("a").perform()
            time.sleep(0.1)
            for i in range(20):
                action.send_keys("w").perform()
            time.sleep(0.1)
        else:
            for i in range(10):
                action.send_keys("w").perform()
            time.sleep(0.1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    except:
        # print("No lines detected")
        for i in range(10):
            action.send_keys("w").perform()
        time.sleep(0.1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cv.imshow("Video", img)

cv.destroyAllWindows()