import cv2
import numpy as np

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def rotateText(thresh):
    nz = cv2.findNonZero(thresh)
    rect = cv2.minAreaRect(nz)

    (cx,cy), (w,h), ang = rect
    if w>h:
        w,h = h,w
        ang += 90

    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv2.warpAffine(thresh, M, (thresh.shape[1], thresh.shape[0]))
    return rotated

def getVerticalLines(thresh):
    vertical = np.copy(thresh)
    verticalsize = 50
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (2, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    return vertical

def get_structure_mask(gray, vertical, uppers):
    H,W = gray.shape[:2]
    demo = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for y in uppers:
        cv2.line(demo, (0,y), (W, y), (0, 255, 0), 2)

    vertical_mask = cv2.cvtColor(vertical, cv2.COLOR_GRAY2BGR)
    vertical_mask[:,:,0] = 0
    vertical_mask[:,:,1] = 0

    overlay = cv2.addWeighted(demo,1,vertical_mask,1,0)

    return cv2.resize(overlay, (1000, 1400)) 