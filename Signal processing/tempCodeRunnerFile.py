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
k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
cv2.imshow("closing", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
