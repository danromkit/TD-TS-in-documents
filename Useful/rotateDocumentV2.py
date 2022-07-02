import cv2
import numpy as np

# Степень конверсии
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res

# Рассчитать угол с помощью преобразования Хафа
def CalcDegreeV2(srcImage):
    gray = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow("gray", gray)
    # # cv2.waitKey(0)
    # gray = cv2.bitwise_not(gray)
    # # cv2.imshow("bitwise_not", gray)
    # # cv2.waitKey(0)
    #
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # # cv2.imshow("thresh", thresh)
    # # cv2.waitKey(0)
    #
    # coords = np.column_stack(np.where(thresh > 0))
    #
    # angle = cv2.minAreaRect(coords)[-1]
    # print("angle_1", angle)
    #
    # if angle < 1:
    #     angle = -angle
    # elif angle > 1:
    #     angle = 90 - angle
    # print("angle_2", angle)
    # # rotate the image to deskew it
    # (h, w) = srcImage.shape[:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    #
    # rotated_img = cv2.warpAffine(srcImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #
    # # cv2.imshow("rotated_img", rotated_img)
    # # cv2.waitKey(0)
    # return rotated_img


    # gray = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)
    #
    # for i in range(gray.shape[0]):
    #     for j in range(2):
    #         gray[i][j] = 0
    #
    # for i in range(2):
    #     for j in range(gray.shape[1]):
    #         gray[i][j] = 0
    #
    # for i in range(gray.shape[0] - 1, gray.shape[0] - 3, -1):
    #     for j in range(gray.shape[1]):
    #         gray[i][j] = 0
    #
    # for i in range(gray.shape[0]):
    #     for j in range(gray.shape[1] - 1, gray.shape[1] - 3, -1):
    #         gray[i][j] = 0
    #
    # angle = cv2.minAreaRect(gray)[-1]
    # print(angle)
    #
    #
    #
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)
