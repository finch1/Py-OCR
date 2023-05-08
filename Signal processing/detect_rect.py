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

from PIL import Image, ImageFilter, ImageEnhance
from skimage import data, exposure, img_as_float
import skimage as sk

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

## read image
img = cv2.imread('page_1.jpg')
(H, W) = img.shape[:2]

modelPath = "frozen_east_text_detection.pb"
min_confidence = 0.4

## resize image (pdf) '''https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/'''
print('Original Dimensions : ',img.shape)
resized = Resize(img, 20)
print('Resized Dimensions : ',resized.shape)

# You may need to convert the color.
resize = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(resize)

# # enh_con = ImageEnhance.Contrast(im_pil)
# # contrast = 5
# # image_contrasted = enh_con.enhance(contrast)


# Apply sharp filter https://pillow.readthedocs.io/en/5.1.x/reference/ImageFilter.html
detail = im_pil.filter(ImageFilter.DETAIL);
sharp = detail.filter(ImageFilter.SHARPEN);
#edgeEnh = sharp.filter(ImageFilter.SMOOTH);


# For reversing the operation:
resize = np.asarray(sharp)
resize = cv2.cvtColor(resize, cv2.COLOR_RGB2BGR)

## smoothing filter '''http://opencvexamples.blogspot.com/2013/10/applying-bilateral-filter.html'''
smooth = cv2.bilateralFilter(resize, 9, 75, 75) #11, 17, 17) ## sharper

# b, g, r = cv2.split(smooth)
# thresholdb = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 25)
# thresholdg = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 25)
# thresholdr = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 25)
# # ret, thresholdb = cv2.threshold(b, 10, 255, cv2.THRESH_OTSU)
# # ret, thresholdg = cv2.threshold(g, 10, 255, cv2.THRESH_OTSU)
# # ret, thresholdr = cv2.threshold(r, 10, 255, cv2.THRESH_OTSU)

# thresh = smooth

# thresh[:, :, 0] = thresholdb # blue
# thresh[:, :, 1] = thresholdg # green
# thresh[:, :, 2] = thresholdr # red

# threshold2 = cv2.bitwise_or(smooth ,resized)
# show the output image
# cv2.imshow("resized", resized)
# cv2.imshow("smooth", smooth)
# cv2.imshow("threshold2", threshold2)
# swap or
# otsu
# cv2.waitKey(0)


# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(modelPath)

workable = cv2.bitwise_or(smooth ,resized)
(newH, newW) = workable.shape[:2]
rW = W / float(newW)
rH = H / float(newH)
(H, W) = workable.shape[:2]

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(workable, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

    	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < min_confidence:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

#boxes = sorted(boxes, key=itemgetter(1)) # sort on the y1 coordinate - horizontally sorted
#print(boxes.size)
#a, b = cv2.groupRectangles(boxes.tolist(), 1, 0.1)
#print(a.size)
#boxes = np.asarray(a)
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	# startX = int(startX * rW)
	# startY = int(startY * rH)
	# endX = int(endX * rW)
	# endY = int(endY * rH)

	# draw the bounding box on the image https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
	cv2.rectangle(workable, (startX, startY), (endX, endY), (0, 200, 0), 1)

# show the output image
cv2.imwrite('detect.png',workable)
cv2.imshow("Text Detection", workable)
cv2.waitKey(0)

# # # ## convert to gray
# # # gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
# # # threshold2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# # # # Invert colours, so gridlines have non-zero pixel values.
# # # # Necessary to dilate the image, otherwise will look like erosion instead.
# # # threshold2 = cv2.bitwise_not(threshold2, threshold2)

# # # # Find the contours in the image
# # # # cv2.RETR_TREE indicates how the contours will be retrieved:
# # # # See: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
# # # # cv2.CHAIN_APPROX_SIMPLE condenses the contour information, only storing the minimum number of points to describe
# # # # the contour shape.
# # # new_img, ext_contours, hier = cv2.findContours(threshold2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # new_img, contours, hier = cv2.findContours(threshold2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# # # processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# # # # Draw all of the contours on the image in 2px red lines
# # # all_contours = cv2.drawContours(processed.copy(), contours, -1, (255, 0, 0), 2)
# # # external_only = cv2.drawContours(processed.copy(), ext_contours, -1, (255, 0, 0), 2)



# # # cv2.imshow("all_contours", all_contours)
# # # cv2.imshow("external_only", external_only)
# # # cv2.waitKey(0) 
# # # cv2.destroyAllWindows()

##text = pytesseract.image_to_string(gray)

''' makes image worse
## Erodes away the boundaries of foreground object. Used to diminish the features of an image.
## https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(gray,kernel,iterations = 2)

## Increases the object area. Used to accentuate features
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(erosion,kernel,iterations = 2)
'''