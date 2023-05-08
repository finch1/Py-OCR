## sharpening
## couple of image processing


import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import os
from PIL import Image
from skimage import data, exposure, img_as_float
import skimage as sk

from PIL import Image
from PIL import ImageFilter

# Open an already existing image
imageObject = Image.open('STAMPS\stamp10.jpg');
imageObject.show();

# Apply sharp filter
sharpened1 = imageObject.filter(ImageFilter.DETAIL);
sharpened2 = sharpened1.filter(ImageFilter.DETAIL);
sharpened1 = sharpened2.filter(ImageFilter.SHARPEN);
sharpened2 = sharpened1.filter(ImageFilter.SHARPEN);
sharpened1 = sharpened2.filter(ImageFilter.SMOOTH);
sharpened2 = sharpened1.filter(ImageFilter.SMOOTH_MORE);
# Show the sharpened images

sharpened2.show();


results = []

#results = [['Original Image','Malta']]

results.append(['Original Image','Malta'])
results.append(['Original Image','Malta'])
print("ROW: ",len(results))
print("COL: ",len(results[0]))
print(results)


def display(title, pos, image):
    plt.subplot(4,4,pos),plt.imshow(image,cmap = 'gray')
    plt.title(title)
    #text = pytesseract.image_to_string(image)
    #results.append([title,text])

def plotCalcHist(title, histogram):
    # configure and draw the histogram figure
    plt.figure()
    plt.title(title)
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0, 256]) # <- named arguments do not work here

    plt.plot(histogram) # <- or here
    plt.show()

def show(img):
    # Display the image.
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(12, 3))

    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_axis_off()

    # Display the histogram.
    ax2.hist(img.ravel(), lw=0, bins=256)
    ax2.set_xlim(0, img.max())
    ax2.set_yticks([])

    plt.show()

#img = cv2.imread('STAMPS\stamp12.jpg',0)
img = cv2.imread('STAMPS\stamp10.jpg')#,0) # ,0 converts it into grey scale
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, img)

img = cv2.imread(filename)#,0) # ,0 converts it into grey scale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Max Grey: ", np.max(grey))
print("Grey shape: ", grey.shape)

# Display original image
display('Original Image',1,img)

# Perform Gaussian Blur
gauBlur = cv2.GaussianBlur(grey, (5,5), 0)
display('Gaussian Blur',2,gauBlur)
print("Max Gaus: ", np.max(gauBlur))

# create the histogram
histogram = cv2.calcHist(images = gauBlur, channels = [0], mask = None, histSize = [256], ranges = [0, 256])
print("hist shape", histogram.shape)
plotCalcHist("Grayscale Histogram", histogram)


# Perform Countours
image, contours, hierarchy = cv2.findContours(gauBlur,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
imgcont = cv2.drawContours(gauBlur, contours, -1, (0,255,0), 3)
display('Contours',4,imgcont)

Equ_hist = exposure.equalize_hist(grey)
display('Equ Hist',5,Equ_hist)
norm_image = cv2.normalize(Equ_hist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# create the histogram
E_histogram = cv2.calcHist(images = norm_image, channels = [0], mask = None, histSize = [256], ranges = [0, 256])
plotCalcHist('Hist Histogram',E_histogram)

''' The rescale_intensity() function stretches or shrinks the intensity levels of the image. One use case is to ensure that the whole range of values allowed by the data type is used by the image.'''
show(exposure.rescale_intensity(gauBlur, in_range=(0.4, .95), out_range=(0, 1)))
''' The equalize_adapthist() function works by splitting the image into rectangular sections and computing the histogram for each section. Then, the intensity values of the pixels are redistributed to improve the contrast and enhance the details. '''
show(exposure.equalize_adapthist(img))
#exposure.histogram(image, nbins=2)


# write the grayscale image to disk as a temporary file so we can apply OCR to it
os.remove(filename)

plt.show()

with open('results.txt', 'w') as f:
    for item in results:
        f.write("%s\n" % item)
'''
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(gauBlur,kernel,iterations = 1)
display('Dilation',6,dilation)
'''


'''
histg = cv2.calcHist([img],[0],None,[256],[0,256])
print("hist type: ", type(histg))
print("hist shape ", histg.shape)
print("img max: ", np.amax(img))
print("img type: ", type(img))
print("img shape: ", img.shape)
print("hist avg: ", np.average(histg))
print("hist max: ", np.amax(histg))
print("hist max i: ", np.argmax(histg))
plt.subplot(4,4,13),plt.hist(img.ravel(),256,[0,256])
plt.title('HIST'), plt.xticks([]), plt.yticks([])

imgo = cv2.imread('STAMPS\stamp10.jpg') # ,0 converts it into grey scale
img = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)
plt.subplot(4,4,14),plt.imshow(img)
plt.title('COLOR'), plt.xticks([]), plt.yticks([])

plt.show()
'''