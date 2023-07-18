import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
path = "caumaidich.mp4"
inpVideo = cv2.VideoCapture(path)

def nothing(x):
    pass

wT = 1920
hT = 1080
cv2.namedWindow("filter")
cv2.createTrackbar("H","filter",0,255,nothing)
cv2.createTrackbar("S","filter",0,255,nothing)
cv2.createTrackbar("L","filter",0,255,nothing)
cv2.createTrackbar("Gray","filter",0,255,nothing)
cv2.namedWindow("points")
cv2.resizeWindow("points", 360, 350)
cv2.createTrackbar("Left-Top_x","points",0,wT,nothing)
cv2.createTrackbar("Left-Top_y","points",0,hT,nothing)
cv2.createTrackbar("Right-Top_x","points",0,wT,nothing)
cv2.createTrackbar("Right-Top_y","points",0,hT,nothing)
cv2.createTrackbar("Left-Bottom_x","points",0,wT,nothing)
cv2.createTrackbar("Left-Bottom_y","points",0,hT,nothing)
cv2.createTrackbar("Right-Bottom_x","points",0,wT,nothing)
cv2.createTrackbar("Right-Bottom_y","points",0,hT,nothing)



def processImage(inpVideo):

    hls = cv2.cvtColor(inpVideo, cv2.COLOR_BGR2HLS)
    h = cv2.getTrackbarPos("H","filter")
    s = cv2.getTrackbarPos("S", "filter")
    l = cv2.getTrackbarPos("L", "filter")
    Gray = cv2.getTrackbarPos("Gray", "filter")
#0,135,0,141
    lower_white = np.array([h, s, l])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(hls, lower_white, upper_white)
    hls_result = cv2.bitwise_and(inpVideo, inpVideo, mask=mask)

    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    ret, thresh = cv2.threshold(gray, Gray, 255, cv2.THRESH_BINARY)
    # cv2.imshow('mask',mask)
    # cv2.imshow('hls',hls)
    # cv2.imshow('hls_result',hls_result)
    # cv2.imshow('gray',gray)
    # cv2.imshow("canny",canny)
    # cv2.imshow('thresh',thresh)
    return hls, hls_result, gray, canny, thresh,h,s,l,Gray

####################################################################################
def regionOfInterest(inpVideo):
    height = inpVideo.shape[0]
    polygons = np.array([
        [(0,990),(1920,990),(1192,378),(776,378)]
    ])
    mask = np.zeros_like(inpVideo)
    cv2.fillPoly(mask,polygons,255)
    masked_video = cv2.bitwise_and(inpVideo,mask)
    return masked_video

####################################################################################
def valTrackbars(wT,hT):
    LeftTop_x = cv2.getTrackbarPos("Left-Top_x", "points")
    LeftTop_y = cv2.getTrackbarPos("Left-Top_y", "points")
    RightTop_x = cv2.getTrackbarPos("Right-Top_x", "points")
    RightTop_y = cv2.getTrackbarPos("Right-Top_y", "points")
    LeftBottom_x = cv2.getTrackbarPos("Left-Bottom_x", "points")
    LeftBottom_y = cv2.getTrackbarPos("Left-Bottom_y", "points")
    RightBottom_x = cv2.getTrackbarPos("Right-Bottom_x", "points")
    RightBottom_y = cv2.getTrackbarPos("Right-Bottom_y", "points")
    # leftTop = np.float32([LeftTop_x,LeftTop_y])
    # rightTop = np.float32([RightTop_x,RightTop_y])
    # leftBottom = np.float32([LeftBottom_x, LeftBottom_y])
    # rightBottom = np.float32([RightBottom_x, RightBottom_y])
    return LeftTop_x,LeftTop_y,RightTop_x,RightTop_y,LeftBottom_x, LeftBottom_y,RightBottom_x, RightBottom_y
####################################################################################
def drawPoints(inpVideo,LeftTop_x,LeftTop_y,RightTop_x,RightTop_y,LeftBottom_x, LeftBottom_y,RightBottom_x, RightBottom_y):
    cv2.circle(inpVideo,(LeftTop_x,LeftTop_y),5,(0,0,255),cv2.FILLED)
    cv2.circle(inpVideo, (RightTop_x,RightTop_y), 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(inpVideo, (LeftBottom_x, LeftBottom_y), 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(inpVideo,(RightBottom_x, RightBottom_y),5,(0,0,255),cv2.FILLED)
    return inpVideo
####################################################################################
frameCounter = 0
while True:
    frameCounter += 1
    if inpVideo.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        inpVideo.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter = 0
    ret,frame = inpVideo.read()
    hls, hls_result, gray, canny, thresh,h,s,l,Gray = processImage(frame)
    masked_video = regionOfInterest(canny)
    cv2.imshow("masked_video", masked_video)
    LeftTop_x,LeftTop_y,RightTop_x,RightTop_y,LeftBottom_x, LeftBottom_y,RightBottom_x, RightBottom_y = valTrackbars(wT, hT)
    frame = drawPoints(frame, LeftTop_x,LeftTop_y,RightTop_x,RightTop_y,LeftBottom_x, LeftBottom_y,RightBottom_x, RightBottom_y)
    cv2.imshow("frame",frame)
    # cv2.imshow('image',frame)
    print(frame.shape)
    if cv2.waitKey(1) == 13:
        file = open("filterData.txt", "a+")
        data = path + " " + str(h) + " " + str(s) + " " + str(l) + " " + str(Gray)
        print(data)
        file.writelines(data + "\n")
        file.close()
        break
inpVideo.release()
cv2.destroyAllWindows()