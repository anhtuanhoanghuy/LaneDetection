import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors

# Defining variables to hold meter-to-pixel conversion
ym_per_pix = 30 / 720
# Standard lane width is 3.7 meters divided by lane width in pixels which is
# calculated to be approximately 720 pixels not to be confused with frame height
xm_per_pix = 1.7 / 720

# Get path to the current working directory
CWD_PATH = os.getcwd()


path = "caumaidich.mp4"
inpVideo = cv2.VideoCapture(path)

def processImage(inpVideo):

    hls = cv2.cvtColor(inpVideo, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 135, 0])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(hls, lower_white, upper_white)
    hls_result = cv2.bitwise_and(inpVideo, inpVideo, mask=mask)
    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 141, 255, cv2.THRESH_BINARY)
    polygons = np.array([
        [(0, 990), (1920, 990), (1192, 378), (776, 378)]
    ])
    mask = np.zeros_like(inpVideo)
    cv2.fillPoly(mask, polygons, 255)
    masked_video = cv2.bitwise_and(inpVideo, mask)

    return hls,hls_result, gray, thresh, masked_video
def perspectiveWarp(inpVideo):
    img_size = (inpVideo.shape[1], inpVideo.shape[0])
    wT = 1920
    hT = 1080
    widthTop = 681
    heightTop=569
    widthBottom=0
    heightBottom = 814

    points = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop), (widthBottom, heightBottom),
                         (wT - widthBottom, heightBottom)])
    src = np.float32(points)

    dst = np.float32([[0, 0], [wT, 0], [0, hT], [wT, hT]])

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(src, dst)
    # Inverse matrix to unwarp the image for final window
    minv = cv2.getPerspectiveTransform(dst, src)
    birdseye = cv2.warpPerspective(inpVideo, matrix, img_size)

    # Get the birdseye window dimensions
    height, width = birdseye.shape[:2]
    return birdseye, minv, img_size
def plotHistogram(inpVideo):

    histogram = np.sum(inpVideo[inpVideo.shape[0] // 2:, :], axis = 0) #tình tổng giá trị của pixel theo chiều cao dọc thoe trục X
    midpoint = np.int64(histogram.shape[0] / 2)     #tọa độ trung điểm x
    leftxBase = np.argmax(histogram[:midpoint])     #lấy tọa độ x nửa bên trái từ 0 -> midpoint
    rightxBase = np.argmax(histogram[midpoint:]) + midpoint    #lấy tọa độ x nửa bên phải từ midpoint -> hết

    # Chiều rộng làn đường trong ảnh tính từ tâm của vạch kẻ bằng (rightxBase - leftxBase)
    # => tham chiếu đến làn đường thực tế chỉ cần nhân với tỉ lệ thực tế/ pixel

    plt.xlabel("Tọa độ X")
    plt.ylabel("Số lượng điểm ảnh trắng")

    return histogram, leftxBase, rightxBase

# Mã này trình bày một hàm tìm kiếm dùng để tìm vị trí của đường màu trắng (lanes)
# trong một ảnh nhị phân (binary_warped), được sử dụng trong bài toán xác định vị trí của ô tô trên đường.
#
# Các bước thực hiện của hàm bao gồm:
#
# Tìm vị trí bắt đầu của đường xe bên trái và bên phải bằng cách tính histogram
# của ảnh nhị phân (biểu diễn số lượng pixel trắng ở mỗi cột) và tìm vị trí có giá trị cao nhất.
#
# Chia ảnh thành 9 cửa sổ (windows) và tìm kiếm lại vị trí của đường xe trong mỗi
# cửa sổ này bằng cách tìm tất cả các pixel trắng nằm trong cửa sổ đó. Nếu số lượng
# pixel trắng nằm trong cửa sổ đó lớn hơn một ngưỡng (minpix), vị trí mới của cửa sổ
# sẽ được dịch chuyển sang trái hoặc phải để bao quanh các pixel trắng này. Tại mỗi
# cửa sổ, chỉ số của các pixel trắng được tìm thấy cho đường xe bên trái và bên phải được lưu trữ.
#
# Với các chỉ số của pixel trắng tìm thấy ở các cửa sổ, hàm sẽ tìm kiếm lại đường xe
# bằng cách sử dụng một khu vực trượt (sliding window) xung quanh các chỉ số này và
# tìm tất cả các pixel trắng trong khu vực này.
#
# Cuối cùng, hàm trả về chỉ số của tất cả các pixel trắng được tìm thấy trong đường xe bên trái và bên phải.
def slide_window_search(binary_warped, histogram):
    # Find the start of left and right lane lines using histogram info
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int64(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # A total of 9 windows will be used
    nwindows = 20
    window_height = np.int64(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 100
    left_lane_inds = []
    right_lane_inds = []


    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high),
        (0,255,0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds] #Mảng các tọa độ X bên trái theo thực tế
    lefty = nonzeroy[left_lane_inds] #Mảng các tọa độ Y bên trái theo thực tế
    rightx = nonzerox[right_lane_inds] #Mảng các tọa độ X bên phải theo thực tế
    righty = nonzeroy[right_lane_inds] #Mảng các tọa độ Y bên phải theo thực tế

    # Apply 2nd degree polynomial fit to fit curves
    left_fit = np.polyfit(lefty, leftx, 2)   ## Hàm đường cong làn bên trái X = a.Y^2 + b.Y + c
    right_fit = np.polyfit(righty, rightx, 2) ## Hàm đường cong làn bên phải
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]) #Giá trị Y trong dải từ 0 -> binary_warped.shape[0]-1

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]      # Mảng tọa độ X bên trái theo đường fit
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]   # Mảng tọa độ X bên phải theo đường fit
    # plt.plot(right_fitx)
    # plt.show()

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]  #Tô làn đường bên trái đã nhận diện được
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] #Tô làn đường bên phải đã nhận diện được
    # plt.imshow(out_img)
    plt.plot(left_fitx,  ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    cv2.imshow("slide", out_img)


    return ploty, left_fit, right_fit
#
# Hàm thứ hai general_search() sử dụng các đường cong phù hợp từ khung trước đó làm điểm bắt
# đầu để tìm kiếm các pixel của làn đường trong một lề. Nó lại áp dụng phép điều chỉnh đa thức
# cấp hai để tìm ra các đường cong phù hợp nhất cho các làn đường bên trái và bên phải.
#
# Mã này cũng bao gồm trực quan hóa các đường làn đường được xác định và các cửa sổ tìm kiếm
# tương ứng trên hình ảnh gốc.
#
# Nhìn chung, mã này dường như là một triển khai tiêu chuẩn của thuật toán phát hiện làn đường
# bằng cách sử dụng cửa sổ trượt và tìm kiếm trong lề.
def general_search(binary_warped, left_fit, right_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ## VISUALIZATION ###########################################################

    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # window_img = np.zeros_like(out_img)
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #
    # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
    #                               ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))
    #
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    #
    # plt.plot(left_fitx,  ploty, color = 'yellow')
    # plt.plot(right_fitx, ploty, color = 'yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)

    ret = {}
    ret['leftx'] = leftx    #Mảng X bên trái chưa fit
    ret['rightx'] = rightx  #Mảng X bên phải chưa fit
    ret['left_fitx'] = left_fitx # Mảng X bên trái đã fit
    ret['right_fitx'] = right_fitx  # Mảng X bên phải đã fit
    ret['ploty'] = ploty  # Mảng Y
    # cv2.imshow("cc",result)
    return ret, left_fitx, right_fitx

# Tiếp theo, nó khớp các đa thức mới với x, y trong không gian thế giới bằng cách sử
# dụng các biến left_fit_cr và right_fit_cr đã được tính toán trong hàm Calibre_camera
# bằng cách sử dụng các hệ số chuyển đổi ym_per_pix và xm_per_pix.
#
# Sau đó, nó tính toán bán kính cong mới cho làn đường bên trái và bên phải bằng cách
# sử dụng các công thức tính bán kính cong của hình parabol:
#
# $R_{curve} = \frac{(1 + (2Ay + B)^2)^{3/2}}{\left|\frac{\partial^2 y}{\partial x^2}\right| }$
# R = |(1 + (dy/dx)^2) ^ (3/2)/ ((d^2)y / d(x^2))|
#
# trong đó A, B và C là các hệ số của phương trình bậc hai và y được đo bằng mét.
#
# Cuối cùng, nó lấy trung bình bán kính cong bên trái và bên phải để có được bán kính
# cong tổng thể cho làn đường và xác định xem đó là đường cong bên trái, đường cong
# bên phải hay làn đường thẳng dựa trên sự khác biệt giữa tọa độ x của bên trái và bên phải.
# làn đường bên phải ở dưới cùng của hình ảnh.
#
# Hàm trả về bán kính cong trung bình và hướng của đường cong.
def measure_lane_curvature(ploty, left_fitx, right_fitx):

    leftx = left_fitx[::-1]  # Reverse to match top-to-bottom in y
    rightx = right_fitx[::-1]  # Reverse to match top-to-bottom in y
    averagex = np.mean((leftx, rightx), axis=0) # Lấy tạo độ X trung bình của 2 đường fit
    average_fit = np.polyfit(ploty, averagex, 2) # Hàm fit trung bình với toạ độ X mới
    fitx = average_fit[0] * ploty ** 2 + average_fit[1] * ploty + average_fit[2] # Tọa độ X theo hàm fit trung bình
    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)


    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    fit_cr = np.polyfit(ploty*ym_per_pix, averagex*xm_per_pix, 2)
    # Tính bán kính cong hệ met
    left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    fit_curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    # print(left_curverad, 'm', right_curverad, 'm')
    if fitx[0] - fitx[-1] > 60:   ## Cái này chỉ nói xem nó cong về bên nào
        curve_direction = 'Left Curve'
    elif fitx[-1] - fitx[0] > 60:
        curve_direction = 'Right Curve'
    else:
        curve_direction = 'Straight'
    return fit_curverad, curve_direction

# Sau đó, nó tạo một hình ảnh màu trống (color_warp) và vẽ các đường làn bằng cách tạo
# một đa giác giữa các đường làn bên trái và bên phải bằng cách sử dụng hàm cv2.fillPoly.
# Ngoài ra, nó vẽ một đa giác màu vàng giữa đường trung bình của làn đường bên trái và
# bên phải, biểu thị diện tích của làn đường.
#
# Cuối cùng, nó áp dụng biến đổi phối cảnh nghịch đảo cho hình ảnh color_warp để đưa nó
# trở lại phối cảnh ban đầu và trộn nó với hình ảnh gốc bằng cách sử dụng hàm cv2.addWeighted.
# Nó trả về các đỉnh của đa giác trung bình và hình ảnh kết quả.
def draw_lane_lines(original_image, warped_image, Minv, draw_info):

    leftx = draw_info['leftx']
    rightx = draw_info['rightx']
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    meanPts = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([meanPts]), (0, 255, 255))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    return meanPts, result

# Chức năng này tính toán độ lệch của ô tô so với tâm làn đường tính bằng mét.
# Nó lấy đầu vào là meanPts là đầu ra của hàm draw_lane_lines và inpFrame là
# hình ảnh gốc. meanPts chứa tọa độ x trung bình của các làn đường bên trái
# và bên phải ở cuối hình ảnh và inpFrame được sử dụng để tính toán hệ số chuyển đổi từ pixel sang mét.
#
# Đầu tiên, hàm trích xuất tọa độ x của các điểm trung bình và tính toán độ
# lệch pixel của ô tô so với trung tâm của hình ảnh. Sau đó, nó chuyển đổi
# độ lệch pixel thành mét bằng cách sử dụng hệ số chuyển đổi và trả về độ
# lệch cũng như hướng của độ lệch dưới dạng một chuỗi.
def offCenter(meanPts, inpFrame):

    # Calculating deviation in meters
    mpts = meanPts[-1][-1][-2].astype(int)
    pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
    deviation = pixelDeviation * xm_per_pix
    direction = "left" if deviation < 0 else "right"

    return deviation, direction
def addText(img, radius, direction, deviation, devDirection):

    # # Add the radius and center position to the image
    font = cv2.FONT_HERSHEY_TRIPLEX

    if (direction != 'Straight'):
        text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius) + 'm'
        text1 = 'Curve Direction: ' + (direction)

    else:
        text = 'Radius of Curvature: ' + 'N/A'
        text1 = 'Curve Direction: ' + (direction)

    cv2.putText(img, text , (50,100), font, 0.8, (0,100, 200), 2, cv2.LINE_AA)
    cv2.putText(img, text1, (50,150), font, 0.8, (0,100, 200), 2, cv2.LINE_AA)
    #
    # # Deviation
    deviation_text = 'Off Center: ' + str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
    cv2.putText(img, deviation_text, (50, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,100, 200), 2, cv2.LINE_AA)

    return img



# Read the input image
while True:

    ret, frame = inpVideo.read()
    birdView, minverse,img_size = perspectiveWarp(frame)
    img, hls_result, grayscale, thresh, masked_video = processImage(birdView)
    hist, leftBase, rightBase = plotHistogram(thresh)
    plt.plot(hist)
    ploty, left_fit, right_fit = slide_window_search(thresh, hist)
    draw_info, left_fitx, right_fitx = general_search(thresh, left_fit, right_fit)
    curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx)
    meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)
    deviation, directionDev = offCenter(meanPts, frame)
    finalImg = addText(result, curveRad, curveDir, deviation, directionDev)
    cv2.imshow("Final", finalImg)
    # cv2.imshow('hls_result',hls_result)
    # cv2.imshow('thresh',thresh)
    cv2.imshow('birdseye',birdView)
    if cv2.waitKey(60) == 13:
        break
inpVideo.release()
cv2.destroyAllWindows()
































##
