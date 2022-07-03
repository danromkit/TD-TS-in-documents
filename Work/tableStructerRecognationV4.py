import copy
import functools

import cv2
import numpy as np

from Work.textRecognation import textRecognation


def custom_tuple_sorting(s, t, offset=4):
    x0, y0, _, _ = s
    x1, y1, _, _ = t
    if abs(y0 - y1) > offset:
        if y0 < y1:
            return -1
        else:
            return 1
    else:
        if x0 < x1:
            return -1

        elif x0 == x1:
            return 0

        else:
            return 1


def recognizeStructerV4(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img_gray", img_gray)
    # cv2.waitKey(0)

    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)

    thresh1 = cv2.bitwise_not(thresh)
    # cv2.imshow("thresh_1", thresh1)
    # cv2.waitKey(0)

    hor_kernel = np.ones((1, 40), np.uint8)  # (1, 27)
    erosion_hor = cv2.erode(thresh1, hor_kernel, iterations=1)
    # cv2.imshow("erosion_hor", erosion_hor)
    # cv2.waitKey(0)

    # Очистка от случайных горизонтальных линий (можно улучшить)
    for i in range(erosion_hor.shape[0]):
        count = 0
        for j in range(erosion_hor.shape[1]):
            if erosion_hor[i][j] == 255:
                count += 1
        if count <= 15:
            for j in range(erosion_hor.shape[1]):
                erosion_hor[i][j] = 0

    dilation_hor = cv2.dilate(erosion_hor, hor_kernel, iterations=1)

    # # Утолщение горизонтальных линий
    # count_hor_start_end_lines_i = []
    # count_hor_start_lines_j = []
    # count_hor_end_lines_i = []
    # count_hor_end_lines_j = []
    #
    # for i in range(dilation_hor.shape[0]):
    #     count = 0
    #     for j in range(dilation_hor.shape[1]):
    #         if dilation_hor[i][j] == 255:
    #             count += 1
    #     if count >= 10:
    #         for j in range(dilation_hor.shape[1]):
    #             if dilation_hor[i][j] == 255:
    #                 count_hor_start_end_lines_i.append(i)
    #                 count_hor_start_lines_j.append(j)
    #                 break
    #         for k in range(dilation_hor.shape[1] - 1, 0, -1):
    #             if dilation_hor[i][k] == 255:
    #                 count_hor_end_lines_j.append(k)
    #                 break
    #     else:
    #         if len(count_hor_start_lines_j) != 0 and len(count_hor_end_lines_j) != 0:
    #             min_start_hor_lines = min(count_hor_start_lines_j)
    #             max_end_hor_lines = max(count_hor_end_lines_j)
    #             for p in count_hor_start_end_lines_i:
    #                 for j in range(min_start_hor_lines, max_end_hor_lines + 1):
    #                     dilation_hor[p][j] = 255
    #             count_hor_start_end_lines_i = []
    #             count_hor_start_lines_j = []
    #             count_hor_end_lines_j = []

    # cv2.imshow("dilation_hor", dilation_hor)
    # cv2.waitKey(0)

    ver_kernel = np.ones((21, 1), np.uint8)  # 15, 1
    erosion_ver = cv2.erode(thresh1, ver_kernel, iterations=1)
    # cv2.imshow("erosion_ver", erosion_ver)
    # cv2.waitKey(0)
    dilation_ver = cv2.dilate(erosion_ver, ver_kernel, iterations=1)
    # cv2.imshow("dilation_ver", dilation_ver)
    # cv2.waitKey(0)

    # # Утолщение вертикальных линий
    # w = dilation_ver.shape[1]
    # h = dilation_ver.shape[0]
    # count_ver_start_end_lines_j = []
    # count_ver_start_lines_i = []
    # count_ver_end_lines_i = []
    # count_ver_lines = 0
    # for j in range(dilation_ver.shape[1]):
    #     count = 0
    #     for i in range(dilation_ver.shape[0]):
    #         if dilation_ver[i][j] == 255:
    #             count += 1
    #     if count >= 10:
    #         for i in range(dilation_ver.shape[0]):
    #             if dilation_ver[i][j] == 255:
    #                 count_ver_start_end_lines_j.append(j)
    #                 count_ver_start_lines_i.append(i)
    #                 break
    #         for k in range(dilation_ver.shape[0] - 1, 0, -1):
    #             if dilation_ver[k][j] == 255:
    #                 count_ver_end_lines_i.append(k)
    #                 break
    #     else:
    #         if len(count_ver_start_lines_i) != 0 and len(count_ver_end_lines_i) != 0:
    #             min_start_ver_lines = min(count_ver_start_lines_i)
    #             max_end_ver_lines = max(count_ver_end_lines_i)
    #             for i in range(min_start_ver_lines, max_end_ver_lines + 1):
    #                 for p in count_ver_start_end_lines_j:
    #                     dilation_ver[i][p] = 255
    #             count_ver_start_end_lines_j = []
    #             count_ver_start_lines_i = []
    #             count_ver_end_lines_i = []

    # cv2.imshow("dilation_ver", dilation_ver)
    # cv2.waitKey(0)

    img_vh = cv2.addWeighted(dilation_ver, 0.5, dilation_hor, 0.5, 0.0)
    _, table = cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("table", table)
    # cv2.waitKey(0)
    # vertical_kernel_for_table = np.ones((21, 1), np.uint8)
    # table_erosian_ver = cv2.erode(table, vertical_kernel_for_table, iterations=1)
    # cv2.imshow("table_erosian_ver", table_erosian_ver)
    # cv2.waitKey(0)
    # table_dilation_ver = cv2.dilate(table_erosian_ver, vertical_kernel_for_table, iterations=1)
    # cv2.imshow("table_dilation_ver", table_dilation_ver)
    # cv2.waitKey(0)

    # table = cv2.bitwise_not(table)
    # cv2.imshow("table", table)
    # cv2.waitKey(0)

    bitor = cv2.bitwise_or(table, thresh)
    # cv2.imshow("bitor", bitor)
    # cv2.waitKey(0)

    bitor = cv2.bitwise_not(bitor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11))  # 7, 6  (7, 10)
    closed = cv2.morphologyEx(bitor, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closed", closed)
    cv2.waitKey(0)

    # Поиск крайней левой и крайней правой точки (j)
    count_hor_start_end_lines_i = []
    count_hor_start_lines_j = []
    count_hor_end_lines_i = []
    count_hor_end_lines_j = []
    all_hor_start_end_lines = []
    for i in range(dilation_hor.shape[0]):
        count = 0
        for j in range(dilation_hor.shape[1]):
            if dilation_hor[i][j] == 255:
                count += 1
        if count >= 10:
            for j in range(dilation_hor.shape[1]):
                if dilation_hor[i][j] == 255:
                    count_hor_start_end_lines_i.append(i)
                    count_hor_start_lines_j.append(j)
                    break
            for k in range(dilation_hor.shape[1] - 1, 0, -1):
                if dilation_hor[i][k] == 255:
                    count_hor_end_lines_j.append(k)
                    break
        else:
            if len(count_hor_start_lines_j) != 0 and len(count_hor_end_lines_j) != 0:
                min_start_hor_lines = min(count_hor_start_lines_j)
                max_end_hor_lines = max(count_hor_end_lines_j)

                all_hor_start_end_lines.append(min_start_hor_lines)
                all_hor_start_end_lines.append(max_end_hor_lines)

                count_hor_start_end_lines_i = []
                count_hor_start_lines_j = []
                count_hor_end_lines_j = []

    min_start_hor_elem = min(all_hor_start_end_lines)
    max_start_hor_elem = max(all_hor_start_end_lines)
    # print("min_start_hor_elem: ", min_start_hor_elem)
    # print("max_start_hor_elem: ", max_start_hor_elem)

    # Поиск самой нижней линии, которая начинается с минимального индекса по горизонтали (i)
    for i in range(dilation_hor.shape[0] - 1, 0, -1):
        if dilation_hor[i][min_start_hor_elem] == 255 or dilation_hor[i][min_start_hor_elem + 1] == 255 or \
                dilation_hor[i][min_start_hor_elem + 2] == 255 or dilation_hor[i][min_start_hor_elem + 3] == 255:
            max_start_ver_elem = i
            break

    for i in range(max_start_ver_elem, dilation_hor.shape[0]):
        count = 0
        for j in range(min_start_hor_elem, max_start_hor_elem):
            if dilation_hor[i][j] == 255:
                count += 1
        if count != 0:
            max_start_ver_elem_1 = i
        else:
            break

    # print("max_start_ver_elem: ", max_start_ver_elem)
    # print("max_start_ver_elem_1: ", max_start_ver_elem_1)

    # Дорисовка линий, которые находятся снизу
    one_hor_line = []
    for i in range(dilation_hor.shape[0] - 1, max_start_ver_elem_1 + 1, -1):
        count = 0
        for j in range(min_start_hor_elem, max_start_hor_elem):
            if dilation_hor[i][j] == 255:
                count += 1
        if count != 0:
            one_hor_line.append(i)
        if count == 0 and len(one_hor_line) != 0:
            max_one_hor_line = max(one_hor_line)
            for k in range(min_start_hor_elem, max_start_hor_elem):
                dilation_hor[max_one_hor_line][k] = 255
            one_hor_line = []

    # cv2.imshow("dilation_hor", dilation_hor)
    # cv2.waitKey(0)

    # Поиск самой верхней линии, которая начинается с минимального индекса по горизонтали (i)
    for i in range(dilation_hor.shape[0]):
        if dilation_hor[i][min_start_hor_elem] == 255 or dilation_hor[i][min_start_hor_elem + 1] == 255 or \
                dilation_hor[i][min_start_hor_elem + 2] == 255 or dilation_hor[i][min_start_hor_elem + 3] == 255:
            max_start_ver_elem_up = i
            break

    for i in range(max_start_ver_elem_up, 0, -1):
        count = 0
        for j in range(min_start_hor_elem, max_start_hor_elem):
            if dilation_hor[i][j] == 255:
                count += 1
        if count != 0:
            max_start_ver_elem_up_1 = i
        else:
            break

    # print("max_start_ver_elem_up: ", max_start_ver_elem_up)
    # print("max_start_ver_elem_up_1: ", max_start_ver_elem_up_1)

    # Дорисовка линий, которые находятся сверху
    one_hor_line = []
    for i in range(0, max_start_ver_elem_up_1 - 1):  # range(max_start_ver_elem_up, -1, -1)
        count = 0
        for j in range(min_start_hor_elem, max_start_hor_elem):
            if dilation_hor[i][j] == 255:
                count += 1
        if count != 0:
            one_hor_line.append(i)
        if count == 0 and len(one_hor_line) != 0:
            max_one_hor_line = max(one_hor_line)  # max(one_hor_line)
            for k in range(min_start_hor_elem, max_start_hor_elem):
                dilation_hor[max_one_hor_line][k] = 255
            one_hor_line = []

    # Дорисовка линий, которые находятся между двумя длинными линиями
    for i in range(48, max_start_ver_elem_1):
        count = 0
        for j in range(dilation_hor.shape[1]):
            if dilation_hor[i][j] == 255:
                count += 1
        for j in range(dilation_hor.shape[1]):
            if dilation_hor[i][j] == 255:
                first_j = j
        for j in range(dilation_hor.shape[1]):
            if dilation_hor[i][j] == 255:
                last_j = j
        if count != 0:
            count_closed_before = 0
            count_closed_after = 0
            for j in range(first_j + 1):
                if closed[i][j] == 255:
                    count_closed_before += 1
            for j in range(closed.shape[1] - 1, last_j, -1):
                if closed[i][j] == 255:
                    count_closed_after += 1
            if count_closed_before == 0:
                for j in range(first_j + 1):
                    dilation_hor[i][j] = 255
            if count_closed_after == 0:
                for j in range(dilation_hor.shape[1] - 1, last_j, -1):
                    dilation_hor[i][j] = 255

    # cv2.imshow("dilation_hor_result", dilation_hor)
    # cv2.waitKey(0)

    # Поиск крайней верхней и крайней нижней точки (i)
    ver_start_lines_i = []
    ver_end_lines_i = []
    ver_start_end_lines_j = []
    all_ver_start_end_lines = []

    for j in range(dilation_ver.shape[1]):
        count = 0
        for i in range(dilation_ver.shape[0]):
            if dilation_ver[i][j] == 255:
                count += 1
        if count >= 10:
            for i in range(dilation_ver.shape[0]):
                if dilation_ver[i][j] == 255:
                    ver_start_lines_i.append(i)  # Здесь хранится значения i
                    ver_start_end_lines_j.append(j)
                    break
            for k in range(dilation_ver.shape[0] - 1, 0, -1):
                if dilation_ver[k][j] == 255:
                    ver_end_lines_i.append(k)
                    break
        else:
            if len(ver_start_lines_i) != 0 and len(ver_end_lines_i) != 0:
                min_start_ver_lines = min(ver_start_lines_i)
                max_end_ver_lines = max(ver_end_lines_i)

                all_ver_start_end_lines.append(min_start_ver_lines)
                all_ver_start_end_lines.append(max_end_ver_lines)

                ver_start_lines_i = []
                ver_end_lines_i = []
                ver_start_end_lines_j = []

    min_start_ver_elem_vertical = min(all_ver_start_end_lines)
    max_start_ver_elem_vertical = max(all_ver_start_end_lines)
    # print("min_start_ver_elem_vertical: ", min_start_ver_elem_vertical)
    # print("max_start_ver_elem_vertical: ", max_start_ver_elem_vertical)

    # Поиск пикселя начала первой вертикальной линии
    first_ver_j = []
    for j in range(dilation_ver.shape[1]):
        count = 0
        for i in range(dilation_ver.shape[0]):
            if dilation_ver[i][j] == 255:
                count += 1
        if count != 0:
            first_ver_j.append(j)
        else:
            if count == 0 and len(first_ver_j) != 0:
                break

    # print("first_ver_j", first_ver_j)

    # Поиск пикселя начала последней вертикальной линии
    last_ver_j = []
    for j in range(dilation_ver.shape[1] - 1, 0, -1):
        count = 0
        for i in range(dilation_ver.shape[0]):
            if dilation_ver[i][j] == 255:
                count += 1
        if count != 0:
            last_ver_j.append(j)
        else:
            if count == 0 and len(last_ver_j) != 0:
                break

    # print("last_ver_j", last_ver_j)

    # first_flag = False
    # for j in first_ver_j:
    #     if dilation_ver[max_start_ver_elem_vertical][j] == 255:
    #         first_flag = True
    #
    # second_flag = False
    # for j in last_ver_j:
    #     if dilation_ver[max_start_ver_elem_vertical][j] == 255:
    #         second_flag = True

    min_last_ver_j = min(last_ver_j)
    max_last_ver_j = max(first_ver_j)
    for i in range(min_start_ver_elem_vertical, max_start_ver_elem_vertical + 1):
        dilation_ver[i][min_last_ver_j] = 255
    for i in range(min_start_ver_elem_vertical, max_start_ver_elem_vertical + 1):
        dilation_ver[i][max_last_ver_j] = 255

    # cv2.imshow("dilation_ver_result", dilation_ver)
    # cv2.waitKey(0)

    img_vh = cv2.addWeighted(dilation_ver, 0.5, dilation_hor, 0.5, 0.0)
    # cv2.imshow("table", img_vh)
    # cv2.waitKey(0)

    for i in range(img_vh.shape[0]):
        for j in range(max_last_ver_j):
            img_vh[i][j] = 0
    for i in range(img_vh.shape[0]):
        for j in range(img_vh.shape[1] - 1, min_last_ver_j, -1):
            img_vh[i][j] = 0

    # cv2.imshow("table", img_vh)
    # cv2.waitKey(0)

    _, img_vh = cv2.threshold(img_vh, 100, 255, cv2.THRESH_BINARY)

    img_vh = cv2.bitwise_not(img_vh)
    cv2.imshow("img_vh", img_vh)
    cv2.waitKey(0)

    # text_vh = cv2.addWeighted(img_vh, 0.5, closed, 0.5, 0.0)
    # cv2.imshow("text_vh", text_vh)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    box = []
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w < 0.9 * img_vh.shape[1] and h < 0.9 * img_vh.shape[0] and h > 0.03 * img_vh.shape[0]:  # 0.03 0.073
            # image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            box.append([x, y, w, h])
            # cv2.imshow("image", img)
            # cv2.waitKey(0)

    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    box.sort(key=functools.cmp_to_key(lambda s, t: custom_tuple_sorting(s, t, 4)))

    for i in box:
        # pass
        # cv2.imshow(f"box{i}", img[i[1]:(i[1] + i[3]), i[0]: (i[0] + i[2])])
        # cv2.waitKey(0)
        # texts = textRecognation(img[i[1]:(i[1] + i[3]), i[0]: (i[0] + i[2])])
        textRecognation(img[i[1]:(i[1] + i[3]), i[0]: (i[0] + i[2])])
        # cv2.imshow(f"box{i}", img[i[1]:(i[1] + i[3]), i[0] : (i[0] + i[2])])
        # cv2.waitKey(0)

    # for text in texts:
    #     print(text)
