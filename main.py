import cv2
import numpy as np
from Work.rotateDocument import CalcDegree
from Work.tableDetection import tableDetect

file = 'pictures/...'
img = cv2.imread(file)
cv2.imshow("img", img)
cv2.waitKey(0)

rotate_img = CalcDegree(img)
# cv2.imshow("rotate", rotate_img)
# cv2.waitKey(0)

rotate_img = tableDetect(rotate_img)
cv2.imshow("tableDetect", rotate_img)
cv2.waitKey(0)
