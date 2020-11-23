import cv2
import math, re
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from regreaderconstants import *
from cvutils import *

custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890\"\'.-()[\  --psm 6'
line_split_border = 1

# template matching
# minMatchQuality = 0.2
# phone_no_template = thresholding(cv2.bitwise_not(cv2.imread('phone_no_template.jpg', 0)))

def getLineIndex(entry, vertical):
    H,W = entry.shape[:2]
    line_index_proposals = []
    for h in range(0, H):
        for prop in np.argwhere(vertical[h,:] > 0):
            line_index_proposals.append(prop)
    
    if len(line_index_proposals) < 8: # reject if not enough proposals
        return None
    return math.floor(np.median(line_index_proposals))

def getFragmentPredictions(fragment, thresh_fragment):
    th = 6
    h,w = thresh_fragment.shape[:2]
    hist = cv2.reduce(thresh_fragment,1, cv2.REDUCE_AVG).reshape(-1)
    uppers = [y for y in range(h-1) if hist[y]<=th and hist[y+1]>th]

    linePredictions = []
    for i in range(len(uppers)):
        y = uppers[i]
        upper_bound = 0 if uppers[i] - line_split_border < 0 else uppers[i]
        lower_bound = uppers[i + 1] if i != len(uppers) - 1 else h - 1
        if (lower_bound - upper_bound < 10):
            continue
        entry = fragment[upper_bound:lower_bound,:]
        thresh_entry = thresh_fragment[upper_bound:lower_bound,:]
        linePredictions.append(getSimplePredictions(entry, thresh_entry))
    return linePredictions

def getSimplePredictions(simple, thresh_fragment):
    simple = removePeriods(simple, thresh_fragment)
    return cleanPrediction(pytesseract.image_to_string(simple,config=custom_config))
    
def removePeriods(entry, thresh_entry):
    th = 3
    min_width = 20  # at least min_width pixel width before white out
    H,W = thresh_entry.shape[:2]

    cv2.imshow("entry", entry)
    cv2.waitKey(0)

    vertical_hist = cv2.reduce(thresh_entry,1, cv2.REDUCE_AVG).reshape(-1)
    lowers = [y for y in range(H//2, H-1) if vertical_hist[y]>th and vertical_hist[y+1]<=th]
    if lowers == []:
        lowers.append(H)

    # cut out periods vertically, find blank areas
    short_entry = thresh_entry[:math.floor(.75 * lowers[0])]
    horizontal_hist = np.array(cv2.reduce(short_entry,0, cv2.REDUCE_AVG).reshape(-1))
    blanks = [x for x in range(3, W - 3) if np.all(horizontal_hist[x-3:x+3] <= th)]
    
    # filter out blanks to only include spaces > 25 pixels
    i = 0
    blank_start = -1
    blank_length = 0
    filtered_blanks = []
    while i < len(blanks) - 1:
        if blanks[i+1] == blanks[i] + 1:
            if blank_start == -1:
                blank_start = blanks[i]
            blank_length += 1
        else:
            if blank_length > min_width:
                filtered_blanks.append(list(range(blank_start, blank_start + blank_length)))
            blank_start = -1
            blank_length = 0
        i += 1
    if blank_length > min_width:
        filtered_blanks.append(list(range(blank_start, blank_start + blank_length)))

    # flatten
    filtered_blanks = [item for sublist in filtered_blanks for item in sublist]

    for x in filtered_blanks:
        cv2.line(entry, (x,0), (x, H), 255, 2)

    cv2.imshow("entry", entry)
    cv2.waitKey(0)

    return entry

def getParentheticals(prediction):
    result = re.search(r"\(([A-Za-z0-9'. ]+)\)", prediction)
    if result != None:
        return result.group(1)
    return None

def getClubAffiliations(prediction):
    clubs = re.findall(r"[A-Z][a-z]?[a-z]?\.", prediction)
    clubs_with_grad_year = re.findall(r"[A-Z][A-z]?[a-z]?[' ]?[0-9][0-9]\.", prediction)
    result = []
    for c in clubs:
        if (validateClubName(c)):
            result.append((c[:-1],None))
    for c in clubs_with_grad_year:
        if (validateClubName(c)):
            result.append((c[:-3],c[-3:-1]))
    return result

def getLast(prediction):
    tokens = prediction.split(' ')
    if (isMultiTokenLastName(tokens[0].lower())):
        return tokens[0] + ' ' + tokens[1]
    return tokens[0]

def getFirst(prediction):
    tokens = prediction.split(' ')
    if len(tokens) >= 3:
        return tokens[2]
    elif len(tokens) == 2:
        return tokens[1]
    return None

# bad perf
def detectPhoneNumber(entry):
    try:
        res = cv2.matchTemplate(entry,phone_no_template,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        print(max_val)
        if (max_val > minMatchQuality):
            t_h,t_w = phone_no_template.shape[:2]
            bottom_right = (top_left[0] + t_w, top_left[1] + t_h)
            cv2.rectangle(entry,top_left, bottom_right, 255, 2)
    except:
        return None

def cleanPrediction(tsOutput):
    return tsOutput.replace('\n',' ')[:-1]