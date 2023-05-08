import numpy as np
import cv2
from PIL import Image
from numpy.lib.type_check import imag
import pytesseract
from matplotlib import pyplot as plt

def resize_image(image, scale):
    scale_percent = scale # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)  
    # resize image
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

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


cascPath = "OCR\haarcascade_frontalface_default.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)  # https://realpython.com/face-recognition-with-python/

image_file = "OCR\pass1.jpg"
image_file = "OCR\pass2.jpg"
image_file = "OCR\pass3.jpg"

original_img = cv2.imread(image_file)

gray_image = grayscale(original_img)
# display_img("grayscale", gray_image) 

'''
# Detect faces in the image
faces = faceCascade.detectMultiScale(  #  general function that detects objects. In this case, a face
    gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(100, 100),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print ("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(gray_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
display_img("FaceDetect", gray_image) 


hist_gry = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
plt.plot(hist_gry, color='k')
plt.show()
'''


gammaImg = gammaCorrection(gray_image, 1.5)
# display_img("gamma corr", gammaImg) 

# removes noise and random colors
thresh, im_bw = cv2.threshold(gammaImg, 70, 255, cv2.THRESH_BINARY)
display_img("threshold binary", im_bw)