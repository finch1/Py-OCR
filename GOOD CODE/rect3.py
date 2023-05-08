# mean - average
# mode - most freqently occuring
# median - middle value

import cv2
import statistics as sta
import numpy as np 
import dataarray as da
import pandas as pd 
from operator import itemgetter # https://stackoverflow.com/questions/2173797/how-to-sort-2d-array-by-row-in-python

def calculateOverlap(l_two_x, l_two_y, r_one_x, r_one_y):    
    # if rect is to the right and overlapping
    if l_two_x < r_one_x and l_two_y < r_one_y:
        return True
    # if two elements are close to each other

def randomcolor():
    return np.random.choice(range(256), size=3)

height = 1152
width = 800

black_image = np.zeros((height, width, 3), np.uint8)

boxes = da.Y
boxes = sorted(boxes, key=itemgetter(1)) # sort on the y1 coordinate - horizontally sorted
boxesHieght = [] # list of box heights
tempbin = [] # temp list to bin boxes

bincount = 0
data = {}

for (x1, y1, x2, y2) in boxes:
    boxesHieght.append((y2-y1))

med = int(sta.median(boxesHieght))

while boxes:
    for (i_x1, i_y1, i_x2, i_y2) in boxes:
            if i_y1 < (boxes[0][1] + med): # y1
                    tempbin.append([i_x1, i_y1, i_x2, i_y2])
    
    tempbin = sorted(tempbin, key=itemgetter(0)) # sort on the x1 coordinate - horizontally sorted
    data[bincount] = tempbin.copy() # group bins

    for (x1, y1, x2, y2) in tempbin: # reverse so when we remove, list won't get confused
            boxes.remove([x1, y1, x2, y2]) # remove so we don't come across again

    bincount += 1
    tempbin.clear()

for key in data:
    r, g, b = randomcolor()
    for (x1, y1, x2, y2) in data[key]:
        cv2.rectangle(black_image, (x1, y1), (x2, y2), (int(b), int(g), int(r)), 1)



# prev = boxbin[0]
# boxbin.pop[0] ## remove first element so we don't check against same rect
# for (x1, y1, x2, y2) in boxbin:
#     if calculateOverlap(x1, y1, prev[2], prev[3]):
#          
#          # either use the last rect or the white one
#          # but to check closeness, its better to use the last one. just save the coordinates
#     prev = [x1, y1, x2, y2]
cv2.imwrite('boxes.png',black_image)

cv2.imshow("BI", black_image)

cv2.waitKey(0)
cv2.destroyAllWindows()