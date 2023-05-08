# https://realpython.com/python-opencv-color-spaces/
# https://github.com/tody411/ColorHistogram

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

# https://www.geeksforgeeks.org/python-os-access-method/
print(os.access('STAMPS\stamp10.jpg', os.R_OK)) 

stampBGR = cv2.imread('STAMPS\stamp10.jpg')
print(stampBGR.shape)
##plt.imshow(stampBGR)
##plt.show()

stampRGB = cv2.cvtColor(stampBGR, cv2.COLOR_BGR2RGB)
plt.imshow(stampRGB)
#plt.show()

r, g, b = cv2.split(stampRGB)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = stampRGB.reshape((np.shape(stampRGB)[0]*np.shape(stampRGB)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

'''
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")

plt.show()
'''

hsv_stampRGB = cv2.cvtColor(stampRGB, cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(hsv_stampRGB)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

print("H. Shape: ", h.flatten().shape, "Size: ", h.flatten().size)
print("S. Shape: ", s.flatten().shape, "Size: ", s.flatten().size)
print("V. Shape: ", v.flatten().shape, "Size: ", v.flatten().size)

hf = h.flatten().astype(int)
sf = s.flatten().astype(int)
print(hf[30054])
np.savetxt('testh.txt', hf.astype(int), delimiter=',')   # x,y,z equal sized 1D arrays
np.savetxt('tests.txt', sf, delimiter=',')   # x,y,z equal sized 1D arrays

print(type(hf))
print(type(sf))

plt.scatter(hf,sf)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print("upper - hue: ", h[134,54],"sat: ", s[134,54],"val: ", v[134,54])
print("lower - hue: ", h[138,58],"sat: ", s[138,58],"val: ", v[138,58])

'''
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
'''

light_orange = (90, 18, 255)
dark_orange = (109, 111, 203)

lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()

mask = cv2.inRange(hsv_stampRGB, light_orange, dark_orange)

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(hsv_stampRGB))
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.show()

result = cv2.bitwise_and(stampRGB, stampRGB, mask=mask)

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()