import numpy as np
import cv2
from PIL import Image
from numpy.lib.type_check import imag
import pytesseract
from matplotlib import pyplot as plt

'''
# some basic things at the begining
im_file = "OCR\page_01.jpg"
im = Image.open(im_file)
# print image metadata
print(im)
# see the picture
im.show()
'''
'''
## Invert Image (white to black - black to white)
inverted_image = cv2.bitwise_not(im_bw)
display_img("inverted image", inverted_image)

# cv2.imwrite("OCR/no_noise_image.jpg",im_bw)
'''
'''
# the open cv way
# load image to memory
image_file = "OCR\page_01.jpg"
img = cv2.imread(image_file)
'''

def display_img(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

## Binarization
# greyscaling 
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # returns a gray image

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
 
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
 
    return cv2.LUT(src, table)

## Noise Removal
def noise_removal(image):

    kernel = np.ones((3,3), np.uint8)
    image = cv2.erode(image, kernel, iterations=1) # helps with adding missing pixels to characters

    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    # remove background noise
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # image = cv2.medianBlur(image, 3)
    return(image)

## Dialasion and Erosion
# help with embolding small letters or adjust font - thick / thin

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image =cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return(image)


def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image =cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return(image)



## Rescaling
image_file = "OCR\servebolt-invoice.png"
image_file = "OCR\page_01.jpg"

img = cv2.imread(image_file)

scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

image_file = "OCR\page_01.jpg"
# image_file = "OCR\invoice.png"
# image_file = "OCR\pass1.jpg"
# image_file = "OCR\pass2.jpg"
# image_file = "OCR\pass3.jpg"

resized = cv2.imread(image_file)

# display_img("original image", resized)

gray_image = grayscale(resized)
display_img("grayscale", gray_image) 

hist_gry = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

plt.plot(hist_gry, color='k')
# plt.show()

gammaImg = gammaCorrection(gray_image, 1.5)
# display_img("gamma corr", gammaImg) 

# removes noise and random colors
thresh, im_bw = cv2.threshold(gammaImg, 220, 255, cv2.THRESH_BINARY)
display_img("threshold black&white", im_bw)




font_thick = thick_font(im_bw)
display_img("font thick after thresh", font_thick)

font_thin = thin_font(font_thick)
display_img("font thin after thick", font_thin)

# verify text extraction
text = []
results = []
words = []


ocr_result = pytesseract.image_to_string(font_thin)
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

noise_removed = noise_removal(im_bw)
display_img("noise removal thresh", noise_removed)

eroded_image = thin_font(noise_removed)
display_img("eroded image", eroded_image)




dialated_image = thick_font(noise_removed)
display_img("dilated image", dialated_image)



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