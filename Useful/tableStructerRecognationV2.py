import copy
import cv2
import numpy as np


def recognizeStructerV2(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img_gray", img_gray)
    # cv2.waitKey(0)

    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow("img_bin", thresh)
    # cv2.waitKey(0)

    thresh1 = cv2.bitwise_not(thresh)
    # cv2.imshow("thresh_1", thresh1)
    # cv2.waitKey(0)

    hor_kernel = np.ones((1, 40), np.uint8)  # (1, 27)
    erosion_hor = cv2.erode(thresh1, hor_kernel, iterations=1)
    cv2.imshow("erosion_hor", erosion_hor)
    cv2.waitKey(0)

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
    # cv2.imshow("dilation_hor", dilation_hor)
    # cv2.waitKey(0)

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

    cv2.imshow("dilation_hor", dilation_hor)
    cv2.waitKey(0)

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
    for i in range(0, max_start_ver_elem_up_1 - 1):
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

    # Удаление линий, у которых меньше всего пикселей со значением 255
    i_max_hor_in_line = []
    for i in range(dilation_hor.shape[0]):
        count = 0
        for j in range(dilation_hor.shape[1]):
            if dilation_hor[i][j] == 255:
                count += 1
        if count != 0:
            index_value = [i, count]
            # count_max_hor_in_line.append(count)
            i_max_hor_in_line.append(index_value)
        elif count == 0 and len(i_max_hor_in_line) != 0:
            maximum = 0
            for k in range(len(i_max_hor_in_line)):
                if i_max_hor_in_line[k][1] >= maximum:
                    maximum = i_max_hor_in_line[k][1]

            for k in range(len(i_max_hor_in_line)):
                if maximum in i_max_hor_in_line[k]:
                    i_max_hor_in_line[k][0] = 0
                    i_max_hor_in_line[k][1] = 0

            for i in range(len(i_max_hor_in_line)):
                for j in range(dilation_hor.shape[1]):
                    dilation_hor[i_max_hor_in_line[i][0]][j] = 0
            i_max_hor_in_line = []


    cv2.imshow("dilation_hor_1_up", dilation_hor)
    cv2.waitKey(0)

    ver_kernel = np.ones((15, 1), np.uint8)
    erosion_ver = cv2.erode(thresh1, ver_kernel, iterations=1)
    # cv2.imshow("erosion_ver", erosion_ver)
    # cv2.waitKey(0)
    dilation_ver = cv2.dilate(erosion_ver, ver_kernel, iterations=1)
    cv2.imshow("dilation_ver", dilation_ver)
    cv2.waitKey(0)

    # # Поиск пикселя начала первой вертикальной линии
    # first_ver_j = []
    # for j in range(dilation_ver.shape[1]):
    #     count = 0
    #     for i in range(dilation_ver.shape[0]):
    #         if dilation_ver[i][j] == 255:
    #             count += 1
    #             # first_ver_j = j
    #             # break
    #     if count != 0:
    #         first_ver_j.append(j)
    #     else:
    #         if count == 0 and len(first_ver_j) != 0:
    #             max_first_ver_j = max(first_ver_j)
    #             break
    #
    #
    # # Поиск последнего пикселя в найденной вертикальной линии, чье значение равно 255
    # for i in range(dilation_ver.shape[0] - 1, 0, -1):
    #     if dilation_ver[i][max_first_ver_j] == 255:
    #         last_ver_i = i
    #         break
    #
    # # Поиск пикселя начала последней вертикальной линии
    # last_ver_j = []
    # count_ver = []
    #
    # for j in range(dilation_ver.shape[1] - 1, 0, -1):
    #     count = 0
    #     for i in range(dilation_ver.shape[0]):
    #         if dilation_ver[i][j] == 255:
    #             count += 1
    #     if count != 0:
    #
    #         last_ver_j.append(j)
    #     else:
    #         if len(last_ver_j) != 0:
    #             break
    #
    # # Поиск максимального значения пикселя последней вертикальной линии
    # max_last_ver_j = []
    # for j in last_ver_j:
    #     for i in range(dilation_ver.shape[0] - 1, 0, -1):
    #         if dilation_ver[i][j] == 255:
    #             max_last_ver_j.append(i)
    #             break
    #
    # max_last_ver_i = max(max_last_ver_j)
    #
    # # Достраиваем первую вертикальную линию
    # for i in range(last_ver_i, max_last_ver_i + 1):
    #     dilation_ver[i][max_first_ver_j] = 255
    #
    # # print("max_last_ver_i", max_last_ver_i)
    # # print("first_ver_j", first_ver_j)
    # # print("last_ver_i", last_ver_i)
    # # print("last_ver_j", last_ver_j)

    # cv2.imshow("dilation_ver", dilation_ver)
    # cv2.waitKey(0)
    #
    # img_vh = cv2.addWeighted(dilation_ver, 0.5, dilation_hor, 0.5, 0.0)
    # cv2.imshow("table", img_vh)
    # cv2.waitKey(0)
    #
    # _, img_vh = cv2.threshold(img_vh, 100, 255, cv2.THRESH_BINARY)
    #
    # img_vh = cv2.bitwise_not(img_vh)
    # cv2.imshow("img_vh", img_vh)
    # cv2.waitKey(0)
    #
    #
    # contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # for i in range(0, len(contours)):
    #     x, y, w, h = cv2.boundingRect(contours[i])
    #     if w < 0.9 * img_vh.shape[1] and h < 0.9 * img_vh.shape[0]:
    #         image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #         # cv2.imshow("image", img)
    #         # cv2.waitKey(0)
    #
    # cv2.imshow("image", image)
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
    print("min_start_ver_elem_vertical: ", min_start_ver_elem_vertical)
    print("max_start_ver_elem_vertical: ", max_start_ver_elem_vertical)

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

    print("first_ver_j", first_ver_j)

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

    print("last_ver_j", last_ver_j)

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
    # if first_flag > second_flag:
    #     for i in range(min_start_ver_elem_vertical, max_start_ver_elem_vertical + 1):
    #         dilation_ver[i][min_last_ver_j] = 255
    #
    # if first_flag < second_flag:
    #     for i in range(min_start_ver_elem_vertical, max_start_ver_elem_vertical + 1):
    #         dilation_ver[i][max_last_ver_j] = 255
    #
    # if first_flag == second_flag:
    #     for i in range(min_start_ver_elem_vertical, max_start_ver_elem_vertical + 1):
    #         dilation_ver[i][min_last_ver_j] = 255
    #     for i in range(min_start_ver_elem_vertical, max_start_ver_elem_vertical + 1):
    #         dilation_ver[i][max_last_ver_j] = 255

    cv2.imshow("dilation_ver", dilation_ver)
    cv2.waitKey(0)

    # erode_img_vh = cv2.addWeighted(erosion_ver, 0.5, erosion_hor, 0.5, 0.0)
    # cv2.imshow("erode_img_vh", erode_img_vh)
    # cv2.waitKey(0)
    img_vh = cv2.addWeighted(dilation_ver, 0.5, dilation_hor, 0.5, 0.0)

    for i in range(img_vh.shape[0]):
        for j in range(max_last_ver_j):
            img_vh[i][j] = 0
    for i in range(img_vh.shape[0]):
        for j in range(img_vh.shape[1] - 1, min_last_ver_j, -1):
            img_vh[i][j] = 0

    cv2.imshow("table", img_vh)
    cv2.waitKey(0)

    _, img_vh = cv2.threshold(img_vh, 100, 255, cv2.THRESH_BINARY)

    img_vh = cv2.bitwise_not(img_vh)
    cv2.imshow("img_vh", img_vh)
    cv2.waitKey(0)

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(len(contours))
    # print(contours[0])
    # print(len(contours[0]))
    # print(cv2.boundingRect(contours[0]))

    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    # Get mean of heights
    mean = np.mean(heights)

    # Create list box to store all boxes in
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    # print("lencontours", len(contours))
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # print("x", x, "y", y, "w", w, "h", h)
        if (w < 0.9 * img_vh.shape[1] and h < 0.9 * img_vh.shape[0]):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    cv2.imshow("image", image)
    cv2.waitKey(0)

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0

    # print("len box", len(box))
    # Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]

        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]

                if (i == len(box) - 1):
                    row.append(column)

            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    # print(column)
    # print(row)

    # calculating maximum number of cells
    countcol = 0
    index = 0
    for i in range(len(row)):
        current = len(row[i])
        # print("len",len(row[i]))
        if current > countcol:
            countcol = current
            index = i

    # print("countcol", countcol)

    # Retrieving the center of each column
    # center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]
    # print("center",center)

    center = np.array(center)
    center.sort()
    # print("center.sort()", center)
    # Regarding the distance to the columns center, the boxes are arranged in respective order

    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)



    # return finalboxes, img_bin
