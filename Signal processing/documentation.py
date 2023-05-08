## import
import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pytesseract
import os
import skimage as sk

from skimage import data, exposure, img_as_float
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
from PIL import Image
from PIL import ImageFilter


## read an image
img = io.imread('STAMPS\stamp10.jpg')
img = cv2.imread('STAMPS\stamp10.jpg')
oriimg = cv2.imread(filename,cv2.CV_LOAD_IMAGE_COLOR)

## image props
height, width, depth = oriimg.shape

## image array
red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]
r, g, b = cv2.split(stampRGB)

## clear image color
img[:, :, 2] = 255

## ploting side bars next to images
fig, axs = plt.subplots(2,2)

cax_00 = axs[0,0].imshow(img)
axs[0,0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
axs[0,0].yaxis.set_major_formatter(plt.NullFormatter())  # kill ylabels

cax_01 = axs[0,1].imshow(red, cmap='Reds')
fig.colorbar(cax_01, ax=axs[0,1])
axs[0,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[0,1].yaxis.set_major_formatter(plt.NullFormatter())

cax_10 = axs[1,0].imshow(green, cmap='Greens')
fig.colorbar(cax_10, ax=axs[1,0])
axs[1,0].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,0].yaxis.set_major_formatter(plt.NullFormatter())

cax_11 = axs[1,1].imshow(blue, cmap='Blues')
fig.colorbar(cax_11, ax=axs[1,1])
axs[1,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,1].yaxis.set_major_formatter(plt.NullFormatter())
plt.show()

## histograms and ploting historgrams
fig, axs = plt.subplots(3, sharex=True, sharey=True)

axs[0].hist(red.ravel(), bins=10)
axs[0].set_title('Red')
axs[1].hist(green.ravel(), bins=10)
axs[1].set_title('Green')
axs[2].hist(blue.ravel(), bins=10)
axs[2].set_title('Blue')
plt.show()
histogram = cv2.calcHist(images = gauBlur, channels = [0], mask = None, histSize = [256], ranges = [0, 256])

# tuple to select colors of each channel line
colors = ("b", "g", "r") 
plt.xlim([0, 256])
for(channel, c) in zip(channels, colors):
    histogram = cv2.calcHist(
        images = [channel], 
        channels = [0], 
        mask = None, 
        histSize = [256], 
        ranges = [0, 256])

    plt.plot(histogram, color = c)

# Perform Countours
image, contours, hierarchy = cv2.findContours(gauBlur,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
imgcont = cv2.drawContours(gauBlur, contours, -1, (0,255,0), 3)


## thresholding
_, threshold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) # removed biro from stamp
gaus = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)
mean_c = cv2.adaptiveThreshold(gaus, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12) # removed bold
_, otsu = cv2.threshold(gaus,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

## normalize
norm_image = cv2.normalize(Equ_hist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


## Perform Gaussian Blur
gauBlur = cv2.GaussianBlur(grey, (5,5), 0)

## color conversion
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)

## show image
cv2.imshow("Otsu", otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

## Equalization
img_eq = exposure.equalize_hist(img)

## Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

## The rescale_intensity() function stretches or shrinks the intensity levels of the image. One use case is to ensure that the whole range of values allowed by the data type is used by the image.
exposure.rescale_intensity(gauBlur, in_range=(0.4, .95), out_range=(0, 1))

## Dilation
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(gauBlur,kernel,iterations = 1)

## functions
def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram. """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)#, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

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