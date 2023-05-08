# import the necessary packages
# from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread("page_2.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)
ret,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)

outerBox = cv2.bitwise_not(thresh)
# edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")

cv2.imshow("Image", image)
cv2.imshow('thresh', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()