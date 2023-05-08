## displays color components
## displays histograms
## displays thresholds

import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

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

# Read
img = io.imread('page_2.jpg')
# img = cv2.imread('STAMPS\stamp10.jpg')
resized = Resize(img, 20)
img = resized

# Split
red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]


# Plot
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

# Plot histograms
fig, axs = plt.subplots(3, sharex=True, sharey=True)

axs[0].hist(red.ravel(), bins=10)
axs[0].set_title('Red')
axs[1].hist(green.ravel(), bins=10)
axs[1].set_title('Green')
axs[2].hist(blue.ravel(), bins=10)
axs[2].set_title('Blue')

plt.show()

_, threshold = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY) # removed biro from stamp
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gaus = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)
mean_c = cv2.adaptiveThreshold(gaus, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12) # removed bold
_, otsu = cv2.threshold(gaus,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Img", img)
cv2.imshow("Binary threshold", threshold)
cv2.imshow("Mean C", mean_c)
cv2.imshow("Gaussian", gaus)
cv2.imshow("Otsu", otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()