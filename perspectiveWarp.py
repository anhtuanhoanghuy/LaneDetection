import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors


path = 'caumaidich.mp4'
inpVideo = cv2.VideoCapture(path)

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 360, 240)

#kich thuoc cua Video
wT = 1920
hT = 1080
def nothing():
    pass

cv2.createTrackbar("Width Top", "Trackbars",0, wT, nothing)
cv2.createTrackbar("Height Top", "Trackbars",0, hT, nothing)
cv2.createTrackbar("Width Bottom", "Trackbars",0, wT, nothing)
cv2.createTrackbar("Height Bottom", "Trackbars",0, hT, nothing)
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
def regionOfInterest(inpVideo):
    height = inpVideo.shape[0]
    polygons = np.array([
        [(227,1080),(1920,1080),(1920,0),(227,0)]
    ])
    mask = np.zeros_like(inpVideo)
    cv2.fillPoly(mask,polygons,255)
    masked_video = cv2.bitwise_and(inpVideo,mask)
    return masked_video

def valTrackbars(wT,hT):
    LeftTop_x = cv2.getTrackbarPos("Left-Top_x", "points")
    LeftTop_y = cv2.getTrackbarPos("Left-Top_y", "points")
    RightTop_x = cv2.getTrackbarPos("Right-Top_x", "points")
    RightTop_y = cv2.getTrackbarPos("Right-Top_y", "points")
    LeftBottom_x = cv2.getTrackbarPos("Left-Bottom_x", "points")
    LeftBottom_y = cv2.getTrackbarPos("Left-Bottom_y", "points")
    RightBottom_x = cv2.getTrackbarPos("Right-Bottom_x", "points")
    RightBottom_y = cv2.getTrackbarPos("Right-Bottom_y", "points")
    return LeftTop_x,LeftTop_y,RightTop_x,RightTop_y,LeftBottom_x, LeftBottom_y,RightBottom_x, RightBottom_y
####################################################################################
def drawPoints(inpVideo,LeftTop_x,LeftTop_y,RightTop_x,RightTop_y,LeftBottom_x, LeftBottom_y,RightBottom_x, RightBottom_y):
    cv2.circle(inpVideo,(LeftTop_x,LeftTop_y),5,(0,0,255),cv2.FILLED)
    cv2.circle(inpVideo, (RightTop_x,RightTop_y), 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(inpVideo, (LeftBottom_x, LeftBottom_y), 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(inpVideo,(RightBottom_x, RightBottom_y),5,(0,0,255),cv2.FILLED)
    return inpVideo

########################################################################

frameCounter = 0
while True:
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    # widthTop = 681
    # heightTop = 569
    # widthBottom = 0
    # heightBottom = 814
    points = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop), (widthBottom, heightBottom),
                         (wT - widthBottom, heightBottom)])
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [wT, 0], [0, hT], [wT, hT]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    frameCounter += 1
    if inpVideo.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        inpVideo.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter = 0
    ret,frame = inpVideo.read()
    cv2.imshow('image',frame)
    print(frame.shape)
    imgWarp = cv2.warpPerspective(frame, matrix, (wT, hT))
    # imgWarp = regionOfInterest(imgWarp)
    LeftTop_x, LeftTop_y, RightTop_x, RightTop_y, LeftBottom_x, LeftBottom_y, RightBottom_x, RightBottom_y = valTrackbars(
        wT, hT)
    imgWarp = drawPoints(imgWarp, LeftTop_x, LeftTop_y, RightTop_x, RightTop_y, LeftBottom_x, LeftBottom_y, RightBottom_x,
                       RightBottom_y)
    cv2.imshow("frame", imgWarp)
    hls = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2HLS)
    # 0,135,0,141
    lower_white = np.array([0, 135, 0])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(hls, lower_white, upper_white)
    hls_result = cv2.bitwise_and(imgWarp, imgWarp, mask=mask)

    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 141, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    # canny = regionOfInterest(canny)
    # cv2.imshow("canny",canny)
    lines = cv2.HoughLinesP(canny, 1, np.pi/180,150, minLineLength = 200, maxLineGap = 250 )
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(imgWarp, (x1,y1),(x2,y2),(0,255,0),10)
    # cv2.imshow('imgWarp',imgWarp)


    if cv2.waitKey(60) == 13:
        file = open("perspectiveWarpData.txt","a+")
        data = path + " " + str(widthTop) + " " + str(heightTop) + " " + str(widthBottom) + " " + str(heightBottom)
        print(data)
        file.writelines(data + "\n")
        file.close()
        break
inpVideo.release()
cv2.destroyAllWindows()