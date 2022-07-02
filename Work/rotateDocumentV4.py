import math

import cv2
import numpy as np


# Степень конверсии
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res


# Рассчитать угол с помощью преобразования Хафа
def rotateDocumentV4(srcImage):
    img_gray = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("rotate_gray", img_gray)
    # cv2.waitKey(0)

    _, thresh = cv2.threshold(img_gray, 195, 255, cv2.THRESH_BINARY)
    # cv2.imshow("img_bin", thresh)
    # cv2.waitKey(0)

    thresh1 = cv2.bitwise_not(thresh)
    # cv2.imshow("thresh_1", thresh1)
    # cv2.waitKey(0)

    hor_kernel = np.ones((1, 25), np.uint8)
    erosion_hor = cv2.erode(thresh1, hor_kernel, iterations=1)
    # cv2.imshow("erosion_hor", erosion_hor)
    # cv2.waitKey(0)

    # Очистка от случайных горизонтальных линий (можно улучшить)
    for i in range(erosion_hor.shape[0]):
        count = 0
        for j in range(erosion_hor.shape[1]):
            if erosion_hor[i][j] == 255:
                count += 1
        if count <= 25:
            for j in range(erosion_hor.shape[1]):
                erosion_hor[i][j] = 0

    # cv2.imshow("erosion_hor_clean", erosion_hor)
    # cv2.waitKey(0)
    dilation_hor = cv2.dilate(erosion_hor, hor_kernel, iterations=1)
    # cv2.imshow("dilation_hor", dilation_hor)
    # cv2.waitKey(0)

    ready_img = cv2.bitwise_not(dilation_hor)
    # cv2.imshow("ready_img", ready_img)
    # cv2.waitKey(0)

    dstImage = cv2.Canny(ready_img, 50, 200, 3)
    # cv2.imshow("dstImage", dstImage)
    # cv2.waitKey(0)
    lineimage = srcImage.copy()

    # Обнаружение прямых линий по преобразованию Хафа
    # Четвертый параметр - это порог, чем больше порог, тем выше точность обнаружения
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 160, srn=0, stn=0, min_theta=1.39626, max_theta=1.74533)  # 80 100
    # lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 160, srn=0, stn=0,  min_theta=1.309, max_theta=1.8326) #75 105
    # lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 140, srn=0, stn=0,  min_theta=1.5708, max_theta=1.65806) #90 95 право
    # lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 140, srn=0, stn=0, min_theta=1.48353, max_theta=1.5708)  #85 90
    # lines = cv2.HoughLines(dstImage, 2, np.pi / 180, 160)  # 1.562
    sum = 0
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # print("theta:", theta, " rho:", rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            sum += theta
            # cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.imshow("Imagelines", lineimage)
            # cv2.waitKey(0)

    average = sum / len(lines)
    angle = DegreeTrans(average) - 90
    print("angle", angle)
    # Центр вращения является центром изображения
    h, w = srcImage.shape[:2]
    # Рассчитать 2D повернутую матрицу аффинного преобразования
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1)
    # Аффинное преобразование, цвет фона заполнен белым
    rotate = cv2.warpAffine(srcImage, RotateMatrix, (w, h), borderValue=(255, 255, 255))

    return rotate
