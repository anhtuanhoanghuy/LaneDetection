import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors

img2 = cv2.VideoCapture('test.mp4')
img = img2
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 360, 240)
# hls = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
# lower_white = np.array([207, 251, 27])
# upper_white = np.array([255, 255, 255])
# mask = cv2.inRange(img2, lower_white, upper_white)
# hls_result = cv2.bitwise_and(img2, img2, mask=mask)

# Convert image to grayscale, apply threshold, blur & extract edges
# gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 87, 255, cv2.THRESH_BINARY)
wT = 1280
hT = 720
def nothing():
    pass
cv2.createTrackbar("Width Top", "Trackbars",0, wT, nothing)
cv2.createTrackbar("Height Top", "Trackbars",0, hT, nothing)
cv2.createTrackbar("Width Bottom", "Trackbars",0, wT, nothing)
cv2.createTrackbar("Height Bottom", "Trackbars",0, hT, nothing)
while True:
    ret, frame = img2.read()
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop), (widthBottom, heightBottom),
                         (wT - widthBottom, heightBottom)])
    # widthTop = 0
    # heightTop = 700
    # widthBottom = 0
    # heightBottom = 720
    # #
    # points = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop), (widthBottom, heightBottom),
    #                      (wT - widthBottom, heightBottom)])
    # points = np.float32([(0, 251), (wT - 0, 251), (0, 255),
    #                      (wT - 0, 255)])
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [wT, 0], [0, hT], [wT, hT]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(frame, matrix, (wT, hT))

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower_white = np.array([197, 245, 95])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(frame, lower_white, upper_white)
    hls_result = cv2.bitwise_and(hls, hls, mask=mask)
    #
    # # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    imgWarp2 = cv2.warpPerspective(thresh, matrix, (wT, hT))
    cv2.imshow('imgWarp',imgWarp)
    cv2.imshow('img2',frame)
    # # cv2.imshow('c',hls)
    # # cv2.imshow('d', hls_result)
    # # cv2.imshow('e', gray)
    # # cv2.imshow('g',thresh)
    cv2.imshow('ccc',imgWarp2)
    print(frame.shape)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
img.release()
cv2.destroyAllWindows()
