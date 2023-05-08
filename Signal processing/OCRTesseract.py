import pytesseract
from PIL import Image

img_file = "OCR\page_01.jpg"
img_file = "OCR/no_noise_image.jpg"
img_file = "OCR\pass1.jpg"

img = Image.open(img_file)

ocr_result = pytesseract.image_to_string(img)
print(ocr_result)
