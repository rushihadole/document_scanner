import cv2
import numpy as np

width, height = 480,640

cap = cv2.VideoCapture(1)
cap.set(3,480)
cap.set(4,360)
cap.set(10,100)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def pre_processing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)
    img_canny = cv2.Canny(img_blur, 200,200)
    kernel = np.ones((5,5))
    img_dilation = cv2.dilate(img_canny, kernel,iterations=2)
    img_threshold = cv2.erode(img_dilation, kernel,iterations=1)

    return img_canny

def get_contour(img):
    biggest = np.array([])
    maxArea = 0
    contours,heirarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area>500:
            #cv2.drawContours(img_con, contour, -1, (255, 0, 0), 2)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour,0.015*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(img_con, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder(my_points):
    my_points = my_points.reshape((4,2))
    temp_points = np.zeros((4,1,2), np.int32)
    add = my_points.sum(1)
    diff = np.diff(my_points, axis=1)

    temp_points[0] = my_points[np.argmin(add)]
    temp_points[1] = my_points[np.argmin(diff)]
    temp_points[2] = my_points[np.argmax(diff)]
    temp_points[3] = my_points[np.argmax(add)]

    return temp_points


def get_warp(img, biggest):
    if len(biggest) != 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        img_output = cv2.warpPerspective(img,matrix, (width,height))

    return img_output

while True:
    success, img = cap.read()
    img = cv2.resize(img,(width,height))
    img_con = img.copy()

    img_threshold = pre_processing(img)
    biggest = get_contour(img_threshold)
    if len(biggest) != 0:
        print(biggest)
        img_warped = get_warp(img,biggest)
        #cv2.imshow("warped", img_warped)
        imgArray = ([img, img_con], [img_threshold, img_warped])
        stacked_img = stackImages(0.5,imgArray)

        cv2.imshow("result",stacked_img)

    if cv2.waitKey(1) and 0xff == ord("q"):
        break

