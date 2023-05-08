import numpy as np
import cv2
from PIL import Image
from numpy.lib.type_check import imag
import pytesseract
from matplotlib import pyplot as plt
from pytesseract.pytesseract import image_to_alto_xml
import pandas as pd

from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

import OCRFunctions as ocrf
import os


fixed_Height = 860
# directory = "C:\\Users\\bminn\\Downloads\\New folder\\Neo4j Books\\OCR\\IDFolder\\"

# for filename in os.listdir(directory):
#     if filename.endswith(".jpg") or filename.endswith(".png"): 
#         print(os.path.join(directory, filename))
#         continue

image_file = "OCR\pass16.jpg"
image_file = "OCR\invoice.png"

gray_Scaled = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
df = pytesseract.image_to_data(gray_Scaled)
# len 15 or > = iban
# Contains @ = email
# First char or all char caps
    # match country list - 2 char - 3 char - whole name
    # remaining is name. if neccessary use image to string to get the context. otherwise create a node with jsut the name.

print(pytesseract.image_to_data(gray_Scaled))

## testing how powerfull teseract is...
##image_to_boxes_tes("medium Test", th)

original_image = cv2.imread(image_file)

height = original_image.shape[0]
scale_Value = round((fixed_Height / height), 1)

orig_Scaled = cv2.resize(original_image, (0,0),fx=scale_Value, fy=scale_Value)

gray_Scaled = cv2.cvtColor(orig_Scaled, cv2.COLOR_BGR2GRAY)

gray_Scaled = ocrf.add_background_border(gray_Scaled)

no_border = ocrf.remove_border(gray_Scaled)
##display_img("borderRemoved", no_border)

## testing how powerfull teseract is...
ocrf.image_to_boxes_tes("Original GraY", no_border.copy())

gamma_Corr = ocrf.gammaCorrection(no_border, 2.35)
##display_img("gammaCorrection", gamma_Corr)

gauss_Blur = cv2.GaussianBlur(gamma_Corr, (5,5), 0)
ocrf.display_img("GaussianBlur", gauss_Blur)

## PLOTING
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Sharing x per column, y per row')

hist_org = cv2.calcHist([gray_Scaled], [0], None, [256], [0, 256])
hist_gam = cv2.calcHist([gamma_Corr], [0], None, [256], [0, 256])
hist_gau = cv2.calcHist([gauss_Blur], [0], None, [256], [0, 256])
#hist_noi = cv2.calcHist([medi_Blur], [0], None, [256], [0, 256])


y = hist_gau.flatten()
yhat = ocrf.savitzky_golay(y, 51, 3) # window size 51, polynomial order 3

ax1.plot(hist_org, color='g')
ax2.plot(hist_gam, color='r')
ax3.plot(hist_gau, color='k')
ax4.plot(yhat, color='b')

for ax in fig.get_axes():
    ax.label_outer()
plt.show()

peaks_x, peaks_y = find_peaks(yhat, height=0) # returns x,y https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
sub = list(peaks_y.values())[0] # peaks_y is a dict, so -> extract values -> becomes 2D array -> get first row
# in case of multiple peaks detected, combine x,y -> zip -> sort -> take first two
zipped = list(zip(sub, peaks_x)) # convert to list for this thing to work. both methods do the trick
zipped.sort(reverse=True)
# remove rest of peaks if exist
zipped = zipped[:2]

# clear peaks
peaks_x = []
for i in zipped:
    peaks_x.append(i[1])


plt.plot(yhat)
plt.plot(peaks_x, yhat[peaks_x], "x")
plt.plot(np.zeros_like(yhat), "--", color="gray")
plt.show()


arr = yhat[peaks_x[1] : peaks_x[0]] 
high_th = (int)(np.where(yhat == np.min(arr))[0]) 
almost_text = ocrf.binarize(gauss_Blur, 0, high_th)
ocrf.display_img("Threshold", almost_text)


ocrf.image_to_boxes_tes("all my work", ocrf.thin_font(almost_text))


print(pytesseract.image_to_data(almost_text))
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60: ###
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)


# verify text extraction
text = []
results = []
words = []


ocr_result = pytesseract.image_to_string(almost_text)
ocr_result = ocr_result.split("\n") # insert new element in array for every new line

for item in ocr_result:
    item = item.strip() # remove spaces
    # item = item.split(" ")[0] # deliminaet by spac and take the first word, which is the name
    if len(item) > 0: # len prevents array out of bounds error
        results.append(item) # store 

# try grabbing the named entities
for segment in results:
    segment = segment.split(" ") # removes spaces from begining and end
    for word in segment:
        words.append(word)


print(len(words))
print(words)


# check below code if relevant. copy code in a new file and add spacy 




# dialated
text = []
ocr_result = pytesseract.image_to_string(dialated_image)
ocr_result = ocr_result.split("\n") # insert new element in array for every new line
results = []

for item in ocr_result:
    item = item.strip() # remove spaces
    # item = item.split(" ")[0] # deliminaet by spac and take the first word, which is the name
    if len(item) > 0: # len prevents array out of bounds error
        results.append(item) # store 

results = list(set(results))
print(len(results))


h, w = noise_removed.shape





NOTmask = cv2.bitwise_not(edges)
image = cv2.bitwise_and(noise_removed, noise_removed, mask=NOTmask)
cv2.imshow("After", image)
cv2.waitKey(0)

# erode text for better edge detection
erosion_size = 4
blur_size = 21
element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))

eroded3 = cv2.erode(noise_removed, element)
##display_img("eroded3", eroded3)

## is picicking even tiny dots
contours, hierarchy  = cv2.findContours(eroded3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)

mask = np.ones(gray_image.shape[:2], dtype="uint8") * 255

# remove too large or too small contours
for index, box in enumerate(contours):
    x, y, w, h = cv2.boundingRect(box)
    if w*h > 1000 and w*h < 15000:
        cv2.drawContours(mask, [box], -1, 0, -1)

    #     cv2.drawContours(gray_image, contours, index, (0,255,75), 2)
    # if the contour is bad, draw it on the mask


    # if cv2.contourArea(box) > 1000 and cv2.contourArea(box) < 15000: # slightly less performance then above
    #     gray_image = cv2.drawContours(gray_image, contours, index, (0,255,75), 2)


# remove the contours from the image and show the resulting images
NOTmask = cv2.bitwise_not(mask)
image = cv2.bitwise_and(gray_image, gray_image, mask=NOTmask)
cv2.imshow("Mask", NOTmask)
cv2.imshow("After", image)
cv2.waitKey(0)

# after mask image is black and we want white background

##display_img("After White", cv2.bitwise_not(image))

# white_image = cv2.imwrite("OCR\passMasked2.jpg", image)
# white_image = cv2.imread("OCR\passMasked.jpg", cv2.IMREAD_GRAYSCALE)
white_image = image

##
orimage = image + cv2.bitwise_not(image)
##display_img("After orimage", orimage)
andimage= image - cv2.bitwise_not(image)
##display_img("After andimage", andimage)


dilated_img = dilatation(white_image)
##display_img("dilated white image", dilated_img)

# compute a "wide", "mid-range", and "tight" threshold for the edges
# using the Canny edge detector
wide = cv2.Canny(dilated_img, 10, 200)
mid = cv2.Canny(dilated_img, 30, 150)
tight = cv2.Canny(dilated_img, 240, 250)
# show the output Canny edge maps
cv2.imshow("Wide Edge Map", wide)
cv2.imshow("Mid Edge Map", mid)
cv2.imshow("Tight Edge Map", tight)
cv2.waitKey(0)

ret3,th3 = cv2.threshold(dilated_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##display_img("Gamma THRESH_OTSU", th3)

sharp = cv2.bitwise_not(th3 + tight)
#display_img("sharp", sharp)

gaussian_blur = cv2.GaussianBlur(sharp,(9,9),sigmaX=0)
#display_img("blur", gaussian_blur)

noise_removed = noise_removal(gaussian_blur)
# display_img("noise removal thresh", noise_removed)

ret3,th4 = cv2.threshold(cv2.bitwise_not(noise_removed), 160, 255, cv2.THRESH_BINARY)
##display_img("threshold", th4)


final = cv2.bitwise_and(th3, th3, mask=th4)
##display_img("final", final)




# font_thin = thin_font(font_thick)

# cv2.imwrite("OCR\passMasked.jpg",font_thin)




'''
hist_gam = cv2.calcHist([font_thin], [0], None, [256], [0, 256])
plt.plot(hist_gam, color='k')
plt.show()
'''



'''

## ROTATE AND DESKEW
# Source: https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
# Calculate skew angle of an image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours. Countours allow droawing of bounding boxes
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    print(angle)
    return rotateImage(cvImage, -1.0 * angle)

rotated = cv2.imread("OCR\page_01_rotated.jpg")
fix_rotated = deskew(rotated)
# display_img("Rotated Image", rotated)
# display_img("Fixed Rotated Image", fix_rotated)
'''