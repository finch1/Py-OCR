import numpy as np
import cv2
from PIL import Image
from numpy.lib.type_check import imag
import pytesseract
from matplotlib import pyplot as plt
from pytesseract.pytesseract import image_to_alto_xml

from scipy.misc import electrocardiogram
from scipy.signal import find_peaks


fixed_Height = 860

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
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
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
    kernel = np.ones((3,3), np.uint8)
    image =cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return(image)

def add_background_border(img):
    # add a white background to help countouring in case the image does not have
    background = cv2.bitwise_not(np.zeros((img.shape[0]+50, img.shape[1]+50), dtype=np.uint8))
    x,y = 25,25
    replace = background.copy()
    replace[y: y + img.shape[0], x: x + img.shape[1]] = img
    return replace

def remove_border(image):

    hist_gry = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist_gry, color='k')
    plt.show()

    kernel = np.ones((9,9), np.uint8)
    img = cv2.dilate(image, kernel, iterations=1) # helps with adding missing pixels to characters
    display_img("dialated", img)

    
    # thresh = point_highest_peaks(img)
    im_bw, thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
    display_img("threshold binary", thresh)

    #find the contours in the image
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    original = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(original, contours, 1, (0,0,255), 2)
    #show the image
    display_img("contours", original)

    x_coord = 0 # counters
    y_coord = 0
    x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(contours[1])
    cropped_image = np.zeros((h_bb,w_bb), dtype=np.uint8)

    for x in range(image.shape[1]):
        if (x >= x_bb and x_coord < w_bb): # lower limit bound with for loop. Upper limit bound with counter
            
            for y in range(image.shape[0]):
                if (y >= y_bb and y_coord < h_bb):                    
                    cropped_image[y_coord][x_coord] = image[y][x]
                    y_coord = y_coord+1 # increment y
            y_coord = 0 # reset coordinates
            x_coord = x_coord+1 # increment x and start looping y

    # if picture of passport has two pages 
    if (h_bb/w_bb) >= 1:
        cropped_image = cropped_image[int(cropped_image.shape[0]/2):]
        # rescale
        height = cropped_image.shape[0]
        scale_Value = round((fixed_Height / height), 1)
        cropped_image = cv2.resize(cropped_image, (0,0),fx=scale_Value, fy=scale_Value)

    #now we just return the new image
    return cropped_image

def remove_face(image):
    
    cascPath = "OCR\haarcascade_frontalface_default.xml"
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)  # https://realpython.com/face-recognition-with-python/

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(  #  general function that detects objects. In this case, a face
        image,
        scaleFactor=1.1,    
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print ("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), -1)
    
    return image

def is_contour_bad(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4

def dilatation(src):
    dilatation_size = 1
    dilation_shape =  cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    return cv2.dilate(src, element)
## Noise Removal
def noise_removal_1(image):

    kernel = np.ones((4,4), np.uint8)
    image = cv2.erode(image, kernel, iterations=1) # helps with adding missing pixels to characters

    kernel = np.ones((4,4), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    # remove background noise
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return(image)

def binarize(image_to_filter, low_th, high_th):

    # the threshold value is usually provided as a number between 0 and 255.
    # This process will be done manually as every ID has light background and dark text
    # the algorithm for the binarization is pretty simple, go through every pixel in the
    # image and, if it's greater than the threshold, turn it all the way up (255), and
    # if it's lower than the threshold, turn it all the way down (0).
    # so lets write this in code. First, we need to iterate over all of the pixels in the
    # image we want to work with
    for x in range(image_to_filter.shape[1]):
        for y in range(image_to_filter.shape[0]):
            # for the given pixel at w,h, lets check its value against the threshold
            if image_to_filter[y][x] >= low_th and image_to_filter[y][x] <= high_th: #note that the first parameter is actually a tuple object
                # lets set this to zero
                image_to_filter[y][x] = 0
            else:
                # otherwise lets set this to 255
                image_to_filter[y][x] = 255
    #now we just return the new image
    return image_to_filter
 
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def point_between_peaks(img):

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    y = hist.flatten()
    yhat = savitzky_golay(y, 51, 3) # window size 51, polynomial order 3

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

    # get hist section between two peaks
    arr = yhat[peaks_x[1] : peaks_x[0]] 
    high_th = (int)(np.where(yhat == np.min(arr))[0]) 
    return binarize(img, 0, high_th)

def point_highest_peaks(img):

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    y = hist.flatten()
    yhat = savitzky_golay(y, 51, 3) # window size 51, polynomial order 3

    peaks_x, peaks_y = find_peaks(yhat, height=0) # returns x,y https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    sub = list(peaks_y.values())[0] # peaks_y is a dict, so -> extract values -> becomes 2D array -> get first row
    # in case of multiple peaks detected, combine x,y -> zip -> sort -> take first two
    zipped = list(zip(sub, peaks_x)) # convert to list for this thing to work. both methods do the trick
    zipped.sort(reverse=True)
    # remove rest of peaks if exist
    zipped = zipped[:1]

    # clear peaks
    high_th = zipped[0][0]

    # get hist section between two peaks
    return binarize(img, 0, high_th)

def image_to_boxes_tes(imageName, img):
    letters = pytesseract.image_to_boxes(img)
    letters = letters.split('\n')
    letters = [letter.split() for letter in letters]
    h, w = img.shape

    delete = []
    # find average area
    for index, letter in enumerate(letters):    
        if len(letter) != 0 and (letter[0].isupper() or letter[0].isnumeric()):
            letters[index].append((int(letter[3]) - int(letter[1])) * (int(letter[4]) - int(letter[2])))
        else:
            delete.append(index)
    
    letters = np.delete(letters,delete,axis=0)

    from scipy import stats
    area_array = [row[6] for row in letters] # returns the seventh columm
    #mode = stats.median_abs_deviation(area_array)
   
    plt.hist(area_array, bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 2000, 3000]) 
    plt.title("histogram") 
    plt.show()
        

    for letter in letters:    
        if len(letter) != 0 and (letter[0].isupper() or letter[0].isnumeric()):
            # remove probale bad data
            #if letter[6] >= 250 and letter[6] <= 600: # add a range to mode +/-10
                print(letter[0] + " - " + letter[1])
                cv2.rectangle(img, (int(letter[1]), h - int(letter[2])), (int(letter[3]), h - int(letter[4])), (0,0,255), 1)

    cv2.imshow(imageName, img)
    cv2.waitKey(0)
