import cv2
import pytesseract

def textRecognation(img):
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    img_text = pytesseract.image_to_string(img, lang='rus+eng')
    print(img_text)
