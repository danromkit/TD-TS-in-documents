import cv2
import numpy as np


def tableDetect(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    # Очистка от случайных горизонтальных линий (можно улучшить)
    for i in range(erosion_hor.shape[0]):
        count = 0
        for j in range(erosion_hor.shape[1]):
            if erosion_hor[i][j] == 255:
                count += 1
        if count <= 15:
            for j in range(erosion_hor.shape[1]):
                erosion_hor[i][j] = 0
    # cv2.imshow("erosion_hor", erosion_hor)
    # cv2.waitKey(0)
    dilation_hor = cv2.dilate(erosion_hor, hor_kernel, iterations=1)
    # cv2.imshow("dilation_hor", dilation_hor)
    # cv2.waitKey(0)

    ver_kernel = np.ones((25, 1), np.uint8)
    erosion_ver = cv2.erode(thresh1, ver_kernel, iterations=1)
    # cv2.imshow("erosion_ver", erosion_ver)
    # cv2.waitKey(0)
    dilation_ver = cv2.dilate(erosion_ver, ver_kernel, iterations=1)
    # cv2.imshow("dilation_ver", dilation_ver)
    # cv2.waitKey(0)

    img_vh = cv2.addWeighted(dilation_ver, 0.5, dilation_hor, 0.5, 0.0)
    # cv2.imshow("img_vh", img_vh)
    # cv2.waitKey(0)

    ready_img = cv2.bitwise_not(img_vh)
    # cv2.imshow("ready_img", ready_img)
    # cv2.waitKey(0)

    for i in range(ready_img.shape[0]):
        for j in range(ready_img.shape[1]):
            if ready_img[i, j] < 140:
                ready_img[i, j] = 0

    contours, hierarchy = cv2.findContours(ready_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # CHAIN_APPROX_TC89_L1

    ogr = round(max(img.shape[0], img.shape[1]) * 0.05)
    delta = round(ogr / 2 + 0.5)
    table_list = []

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if h > delta / 1.1 and w > delta and (hierarchy[0, i, 3] == 0 or hierarchy[0, i, 2] == 0):
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            table_list.append(img[int(y):int(y + h), int(x):int(x + w)])


    return img, table_list
