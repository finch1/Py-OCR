import pytesseract
import cv2

def display_img(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

image = cv2.imread("OCR\index_02.jpg")
# display_img("Original", image)

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray scale image
# display_img("Greyed", grey)

# bluring helps with identifying structure not text. 
blur = cv2.GaussianBlur(grey, (7,7), 0)
# display_img("blured", blur)

# bluring text will help image proccessing figure out columns
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )[1]
# display_img("inverse threshold", thresh)

# identiy structure in text to identify columns for bounding boxes or contours 
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,13))
dialate = cv2.dilate(thresh, kernal, iterations=1)
# display_img("dialated", dialate)

text = []
cnts = cv2.findContours(dialate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    # to avoid drawing the small boxes around words
    if h > 200 and w > 20:
        # we can then save the text in the boxes as images
        roi = image[y:y+h, x:x+h]
        cv2.imwrite("OCR\index_roi.png",roi)
        ocr_result = pytesseract.image_to_string(roi)
        ocr_result = ocr_result.split("\n") # insert new element in array for every new line
        # for item in ocr_result:
        #     text.append(item)
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,12,255), 2) # where to draw bounding boxes

# display_img("bounded", image)

results = []

text = ocr_result
for item in text:
    item = item.strip() # remove spaces
    item = item.split(" ")[0] # deliminaet by spac and take the first word, which is the name
    if len(item) > 0 and item[0].isupper(): # len prevents array out of bounds error
        results.append(item) # store 

results = list(set(results))
print(results)