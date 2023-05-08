'''
# Plot the images
plt.figure(figsize = (20, 20))
for i in range(3):
    img_copy = img.copy()
    img_copy = cv2.erode(img_copy, kernels[i], iterations = 3)
    plt.subplot(1, 3, i+1)
    plt.imshow(img_copy)
    plt.axis('off')
plt.show()
'''

## Works: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
## https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2
# import the necessary packages
import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
import argparse
import time
from pathlib import Path # https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from operator import itemgetter # https://stackoverflow.com/questions/2173797/how-to-sort-2d-array-by-row-in-python
import statistics as sta
from matplotlib import pyplot as plt

from PIL import Image, ImageFilter, ImageEnhance
from skimage import data, exposure, img_as_float
import skimage as sk
from collections import Counter
import collections

def Resize(image, scale_percent):
    # percent of original size
    imgScale = scale_percent / 100

    newHeight = int(img.shape[0] * imgScale)
    newWidth = int(img.shape[1] * imgScale)
    # multiples of 32
    res = newHeight / 32
    newHeight = int(res) * 32
    res = newWidth / 32
    newWidth = int(res) * 32

    dim = (int(newWidth), int(newHeight))
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def FindMaxLength(lst): 
    #maxList = max((x) for x in lst) 
    maxLength = max(len(x) for x in lst )   
    return maxLength 

## read image
img = cv2.imread('page_2.jpg')
(H, W) = img.shape[:2]

## resize image (pdf) '''https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/'''
print('Original Dimensions : ',img.shape)
resized = Resize(img, 20)
print('Resized Dimensions : ',resized.shape)

# You may need to convert the color.
resize = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)


im_pil = Image.fromarray(thresh)

enh_con = ImageEnhance.Contrast(im_pil)
contrast = 5
image_contrasted = enh_con.enhance(contrast)

contrast = np.asarray(image_contrasted)

line = []
field = []
alllengths = []

print(contrast.shape[0])
print(contrast.shape[1])

for i in range(contrast.shape[0]):
    for j in range(contrast.shape[1]):
        if contrast[i][j] == 0: # if has traces of black
            #field.append(contrast[i,j:])
            alllengths.append(j) # store number of white pixels
            field.append(contrast[i])
            break

# trial and error trick
alllengths = [x for x in alllengths if x < 200 and x > 0]
recounted = Counter(alllengths)
recounted = [x for x in recounted if x > 20]
recounted.sort()

MEAN = int(sta.mean(alllengths))
MODE = int(sta.mode(alllengths))
MEDEAN = int(sta.median(alllengths))
MINIM = int(min(alllengths))

'''
W = FindMaxLength(field)
H = len(field)
#crop = np.empty([0,W], dtype=int)
crop = []


i = 0
'''
# padding
'''
while i < H:        
        T = (list(field[i]) + W * [0])[:W]
        #print(type(T))                               
        #U = np.asarray(T)
        #print(type(U))
        crop.append(T)
        #print(type(crop))
        i = i + 1
 
no_white = np.array(crop)
no_white = no_white.astype(np.uint8)
'''

no_white = np.array(field)
no_white = no_white.astype(np.uint8)

crop_img = no_white[:no_white.shape[0], recounted[0]:]

cv2.imshow("ORIGINAL", resize)
cv2.imshow("CONRAST", contrast)
cv2.imwrite("contrast.jpg", contrast)
cv2.imshow("THRESHOLD", thresh)
cv2.imshow("CROPPED", no_white)
cv2.imshow("REDUCED CROPPED", crop_img)

cv2.waitKey(0)
cv2.destroyAllWindows()