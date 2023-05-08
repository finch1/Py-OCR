## Works: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
# import the necessary packages
import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
import argparse
import time
from pathlib import Path # https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f


from PIL import Image, ImageFilter
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

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < min_confidence:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)


## read image
img = cv2.imread('page_1.jpg')
(H, W) = img.shape[:2]

modelPath = "frozen_east_text_detection.pb"
min_confidence = 0.5
padding = 0.0

## resize image (pdf) '''https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/'''
print('Original Dimensions : ',img.shape)
resized = Resize(img, 20)
print('Resized Dimensions : ',resized.shape)

# You may need to convert the color.
resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(resized)

# Apply sharp filter
sharpened1 = im_pil.filter(ImageFilter.DETAIL);
sharpened2 = sharpened1.filter(ImageFilter.DETAIL);
sharpened1 = sharpened2.filter(ImageFilter.SHARPEN);
im_pil = sharpened1.filter(ImageFilter.SHARPEN);
# Show the sharpened images

# For reversing the operation:
resized = np.asarray(im_pil)
resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(modelPath)

## smoothing filter '''http://opencvexamples.blogspot.com/2013/10/applying-bilateral-filter.html'''
smooth = cv2.bilateralFilter(resized, 11, 17, 17) ## sharper
(newH, newW) = smooth.shape[:2]
rW = W / float(newW)
rH = H / float(newH)
(H, W) = smooth.shape[:2]

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(smooth, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# initialize the list of results
results = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	# startX = int(startX * rW)
	# startY = int(startY * rH)
	# endX = int(endX * rW)
	# endY = int(endY * rH)

	# in order to obtain a better OCR of the text we can potentially
	# apply a bit of padding surrounding the bounding box -- here we
	# are computing the deltas in both the x and y directions
	dX = int((endX - startX) * padding)
	dY = int((endY - startY) * padding)

	# apply padding to each side of the bounding box, respectively
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(W, endX + (dX * 2))
	endY = min(H, endY + (dY * 2))

	# extract the actual padded ROI
	roi = smooth[startY:endY, startX:endX]

    	# in order to apply Tesseract v4 to OCR text we must supply
	# (1) a language, (2) an OEM flag of 4, indicating that the we
	# wish to use the LSTM neural net model for OCR, and finally
	# (3) an OEM value, in this case, 7 which implies that we are
	# treating the ROI as a single line of text
	config = ("-l eng --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)

	# add the bounding box coordinates and OCR'd text to the list
	# of results
	results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])

# loop over the results
for ((startX, startY, endX, endY), text) in results:
	# display the text OCR'd by Tesseract
	print("OCR TEXT")
	print("========")
	print("{}\n".format(text))

	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = smooth.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(output, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

	# show the output image
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)