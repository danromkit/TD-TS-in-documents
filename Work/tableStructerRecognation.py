import cv2
import numpy as np


def recognizeStructer(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("rotate_gray", img_gray)
    cv2.waitKey(0)

    _, thresh = cv2.threshold(img_gray, 195, 255, cv2.THRESH_BINARY)
    cv2.imshow("img_bin", thresh)
    cv2.waitKey(0)

    thresh1 = cv2.bitwise_not(thresh)
    cv2.imshow("thresh_1", thresh1)
    cv2.waitKey(0)

    hor_kernel = np.ones((1, 25), np.uint8)
    erosion_hor = cv2.erode(thresh1, hor_kernel, iterations=1)
    cv2.imshow("erosion_hor", erosion_hor)
    cv2.waitKey(0)
    dilation_hor = cv2.dilate(erosion_hor, hor_kernel, iterations=1)

    # max_hor_1 = 0
    # max_hor_2 = 0
    # max_hor_3 = 0
    # count = -1
    # for i in range(dilation_hor.shape[0]):
    #     count += 1
    #     for j in range(dilation_hor.shape[1]):
    #         if count < 3:
    #             if count == 0:
    #                 if dilation_hor[i][j] == 255:
    #                     max_hor_1 += 1
    #             if count == 1:
    #                 if dilation_hor[i][j] == 255:
    #                     max_hor_2 += 1
    #             if count == 2:
    #                 if dilation_hor[i][j] == 255:
    #                     max_hor_3 += 1
    #         else:
    #             if max_hor_1 > max_hor_2 and max_hor_1 > max_hor_3:
    #                 largest = max_hor_1
    #             if max_hor_2 > max_hor_1 and max_hor_2 > max_hor_3:
    #                 largest = max_hor_2
    #             if max_hor_3 > max_hor_1 and max_hor_3 > max_hor_2:
    #                 largest = max_hor_3
    cv2.imshow("dilation_hor", dilation_hor)
    cv2.waitKey(0)

    ver_kernel = np.ones((25, 1), np.uint8)
    erosion_ver = cv2.erode(thresh1, ver_kernel, iterations=1)
    cv2.imshow("erosion_ver", erosion_ver)
    cv2.waitKey(0)
    dilation_ver = cv2.dilate(erosion_ver, ver_kernel, iterations=1)
    cv2.imshow("dilation_ver", dilation_ver)
    cv2.waitKey(0)

    crossing_lines = cv2.bitwise_and(dilation_hor, dilation_ver)
    cv2.imshow("crossing_lines", crossing_lines)
    cv2.waitKey(0)

    for i in range(crossing_lines.shape[0] - 2):
        for j in range(crossing_lines.shape[1] - 2):
            if crossing_lines[i][j] == 0:
                continue
            else:
                if crossing_lines[i][j] == crossing_lines[i][j + 1]:
                    crossing_lines[i][j + 1] = 0
                if crossing_lines[i][j] == crossing_lines[i][j + 2]:
                    crossing_lines[i][j + 2] = 0
                if crossing_lines[i][j] == crossing_lines[i + 1][j]:
                    crossing_lines[i + 1][j] = 0
                if crossing_lines[i][j] == crossing_lines[i + 2][j]:
                    crossing_lines[i + 2][j] = 0
                if crossing_lines[i][j] == crossing_lines[i + 1][j + 1]:
                    crossing_lines[i + 1][j + 1] = 0
                if crossing_lines[i][j] == crossing_lines[i + 2][j + 2]:
                    crossing_lines[i + 2][j + 2] = 0


    # crossing_lines
    cv2.imshow("crossing_lines_1", crossing_lines)
    cv2.waitKey(0)



