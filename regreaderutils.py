import cv2
import math, re
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890\"\'.-()[\  --psm 6'
line_split_border = 2

def rotateText(thresh):
    nz = cv2.findNonZero(thresh)
    rect = cv2.minAreaRect(nz)

    # box = np.int0(cv2.boxPoints(rect))
    # cv2.drawContours(thresh,[box],-1,255,2)

    (cx,cy), (w,h), ang = rect
    if w>h:
        w,h = h,w
        ang += 90

    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv2.warpAffine(thresh, M, (thresh.shape[1], thresh.shape[0]))
    return rotated

def getVerticalLines(thresh):
    vertical = np.copy(thresh)
    rows = vertical.shape[0]
    verticalsize = 50
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    vertical_ranges = cv2.findNonZero 
    return vertical

def getLineIndex(entry, vertical):
    H,W = entry.shape[:2]
    line_index_proposals = []
    for h in range(0, H):
        for prop in np.argwhere(vertical[h,:] > 0):
            line_index_proposals.append(prop)
    
    if len(line_index_proposals) < 5: # reject if not enough proposals
        return None
    return np.median(line_index_proposals)

def getFragmentPredictions(fragment, thresh_fragment):
    th = 5
    H,W = thresh_fragment.shape[:2]
    hist = cv2.reduce(thresh_fragment,1, cv2.REDUCE_AVG).reshape(-1)
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    cv2.cvtColor(fragment, cv2.COLOR_GRAY2BGR)
    
    linePredictions = []
    for i in range(len(uppers)):
        y = uppers[i]
        upper_bound = 0 if i == 0 else uppers[i] - line_split_border
        lower_bound = uppers[i + 1] + line_split_border if i != len(uppers) - 1 else -1
        if (lower_bound - upper_bound < 5):
            continue
        entry = fragment[upper_bound:lower_bound,:]
        linePredictions.append(getSimplePredictions(entry))
    return linePredictions

def getSimplePredictions(simple):
    p = cleanPrediction(pytesseract.image_to_string(simple,config=custom_config))
    return re.sub(r'[.aecuo0238 ]{4,}', ' ', p) 

def cleanPrediction(tsOutput):
    return tsOutput.replace('\n',' ')[:-1]