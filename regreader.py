import cv2
import math
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import pytesseract
from matplotlib import pyplot as plt
from regreaderutils import *
from cvutils import *

pdf_file = convert_from_path("social_register_p135.pdf")
img = np.array(pdf_file[0])
g = get_grayscale(img)
 
# crop
TOP_BORDER_CROP = 200
BOTTOM_BORDER_CROP = 150
L_BORDER_CROP = 200
R_BORDER_CROP = 200
g = g[TOP_BORDER_CROP:-BOTTOM_BORDER_CROP, L_BORDER_CROP:-R_BORDER_CROP]

# thresholding images for structure
ret,thresh = cv2.threshold(~g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# for OCR
ret, g_intermediate = cv2.threshold(g,50,255,cv2.THRESH_TOZERO)
ret, g_inv = cv2.threshold(~g_intermediate,50,255,cv2.THRESH_TOZERO)
ocr_source_img = ~g_inv

th2 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

vertical = getVerticalLines(thresh)

# cv2.imshow("vert", vertical)
# cv2.waitKey(0)

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

# cv2.imshow("structure_mask", get_structure_mask(ocr_source_img, vertical, filtered_uppers))
# cv2.waitKey(0)

data = []
line_split_border = 2
for i in range(0,len(filtered_uppers)):
    upper_bound = 0 if i == 0 else filtered_uppers[i]
    lower_bound = filtered_uppers[i + 1] if i != len(filtered_uppers) - 1 else H-1
    if (lower_bound - upper_bound < 10):
        continue
    ocr_entry = ocr_source_img[upper_bound:lower_bound,:]
    thresh_entry = thresh[upper_bound:lower_bound,:]
    divider = getLineIndex(ocr_entry,vertical[upper_bound:lower_bound,:])
    if (divider != None):
        h, _ = ocr_entry.shape[:2]
        
        right_entry = ocr_entry[:,divider + line_split_border:]
        right_thresh_entry = thresh_entry[:,divider + line_split_border:]
        right_prediction = ' '.join(getFragmentPredictions(right_entry, right_thresh_entry))

        left_entry = ocr_entry[:,:divider - line_split_border]
        left_thresh_entry = thresh_entry[:,:divider - line_split_border]
        for p in getFragmentPredictions(left_entry, left_thresh_entry):
            row = [getLast(p), getFirst(p), getParentheticals(p),getClubAffiliations(p), right_prediction]
            data.append(row)
            print(row, p + right_prediction)
    else:
        p = getSimplePredictions(ocr_entry, thresh_entry)
        c = getClubAffiliations(p)
        row = [getLast(p), getFirst(p), getParentheticals(p), getClubAffiliations(p), None]
        data.append(row)
        print(p)

df = pd.DataFrame(data, columns = ['last', 'first', 'spouse', 'club', 'address'])
print(df)
df.to_csv('out.csv')