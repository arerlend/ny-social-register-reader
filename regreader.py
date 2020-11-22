import cv2
import math
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from regreaderutils import getVerticalLines, getLineIndex, getFragmentPredictions, getSimplePredictions

def getVerticalLines(thresh):
    vertical = np.copy(thresh)
    rows = vertical.shape[0]
    verticalsize = 50
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    vertical_ranges = cv2.findNonZero 
    return vertical

pdf_file = convert_from_path("social_register_p135.pdf")
img = np.array(pdf_file[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# crop
TOP_BORDER_CROP = 200
BOTTOM_BORDER_CROP = 150
L_BORDER_CROP = 200
R_BORDER_CROP = 200
gray = gray[TOP_BORDER_CROP:-BOTTOM_BORDER_CROP, L_BORDER_CROP:-R_BORDER_CROP]

inverted = cv2.bitwise_not(gray)
ret, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(gray, (3,3), 0)
ret,thresh = cv2.threshold(inverted,127,255,cv2.THRESH_BINARY)

th2 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

vertical = getVerticalLines(thresh)

hist = cv2.reduce(thresh,1, cv2.REDUCE_AVG).reshape(-1)

# horizontal axis histogram to detect spaces between lines
th = 5
H,W = thresh.shape[:2]

filtered_uppers = []
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
for i in range(0, len(uppers)):
    if np.sum(vertical[uppers[i],:]) == 0:
        filtered_uppers.append(uppers[i])
        y = uppers[i]
        cv2.line(thresh, (0,y), (W, y), (0, 255, 255), 1)
    else:
        # don't immediately reject, check if we are between different lines
        vert_indexes_above = np.argwhere(vertical[uppers[i] - 10,:] > 0)
        vert_indexes_below = np.argwhere(vertical[uppers[i] + 10,:] > 0)
        
        if (len(vert_indexes_above) == 0 or len(vert_indexes_below) == 0):
            filtered_uppers.append(uppers[i])
        elif (abs(np.average(vert_indexes_above) - np.average(vert_indexes_below)) > 5):
            # probably different lines
            filtered_uppers.append(uppers[i])
            
# demo = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# for y in filtered_uppers:
#     cv2.line(demo, (0,y), (W, y), (0, 255, 0), 2)

# vertical_mask = cv2.cvtColor(vertical, cv2.COLOR_GRAY2BGR)
# vertical_mask[:,:,0] = 0
# vertical_mask[:,:,1] = 0

# # for y in lowers:
# #     cv2.line(thresh, (0,y), (W, y), 255, 1)

# overlay = cv2.addWeighted(demo,1,vertical_mask,1,0)

# imS = cv2.resize(overlay, (1000, 1400)) 
# cv2.imshow('image', imS)
# cv2.waitKey(0)

line_split_border = 2
for i in range(0,len(filtered_uppers)):
    upper_bound = 0 if i == 0 else filtered_uppers[i] - line_split_border
    lower_bound = filtered_uppers[i + 1] + line_split_border if i != len(filtered_uppers) - 1 else -1
    entry = gray[upper_bound:lower_bound,:]
    thresh_entry = thresh[upper_bound:lower_bound,:]
    divider = getLineIndex(entry,vertical[upper_bound:lower_bound,:])
    if (divider != None):
        divider = math.floor(divider)
        h, _ = entry.shape[:2]
        
        # get right entry first
        right_entry = entry[:,divider + line_split_border:]
        right_prediction = getSimplePredictions(right_entry)

        left_entry = entry[:,:divider - line_split_border]
        for p in getFragmentPredictions(left_entry, thresh_entry):
            print((p, right_prediction))
    else:
        print((getSimplePredictions(entry),'needs split'))