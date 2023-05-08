#https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

from  skimage import measure
import matplotlib.pyplot as plt 
import numpy as np 
import cv2

def mse(imgA, imgB):
    err = np.sum((imgA.astype("float") - imgB.astype("float") **2))
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err

def compare_images(imgA, imgB, title):
    m = mse(imgA, imgB)
    s = measure.compare_ssim(imgA, imgB, multichannel=True)

    gif = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m,s))

    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imgA, cmap=plt.cm.gray)
    plt.axis("off")

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imgB, cmap=plt.cm.gray)
    plt.axis("off")

    plt.show()

original = cv2.imread("stamp9.jpg")
contrast = cv2.imread("stamp10.jpg")
shopped = cv2.imread("stamp11.jpg")

fig = plt.figure("Images")
images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)

for(i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap = plt.cm.gray)
    plt.axis("off")

plt.show()

compare_images(original, original, "Original VS. Original")
compare_images(original, contrast, "Original VS. Contrast")
compare_images(original, shopped, "Original VS. Shopped")