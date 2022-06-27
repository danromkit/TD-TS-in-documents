import copy
import cv2
import numpy as np


def recognizeStructer(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img_gray", img_gray)
    # cv2.waitKey(0)

    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow("img_bin", thresh)
    # cv2.waitKey(0)

    thresh1 = cv2.bitwise_not(thresh)
    # cv2.imshow("thresh_1", thresh1)
    # cv2.waitKey(0)

    hor_kernel = np.ones((1, 40), np.uint8)
    erosion_hor = cv2.erode(thresh1, hor_kernel, iterations=1)
    cv2.imshow("erosion_hor", erosion_hor)
    cv2.waitKey(0)

    # Можно улучшить
    for i in range(erosion_hor.shape[0]):
        count = 0
        for j in range(erosion_hor.shape[1]):
            if erosion_hor[i][j] == 255:
                count += 1
        if count <= 35:
            for j in range(erosion_hor.shape[1]):
                erosion_hor[i][j] = 0

    dilation_hor = cv2.dilate(erosion_hor, hor_kernel, iterations=1)
    cv2.imshow("dilation_hor", dilation_hor)
    cv2.waitKey(0)
    thick_dilation_hor = copy.deepcopy(dilation_hor)


    # Поиск самой нижней и длинной горизонтальной линии
    count_hor = []
    i_hor = []

    for i in range(dilation_hor.shape[0]):
        count = 0
        for j in range(dilation_hor.shape[1]):
            if dilation_hor[i][j] == 255:
                count += 1
        count_hor.append(count)
        i_hor.append(i)

    max_count_hor = count_hor[0]

    for i, x in enumerate(count_hor):
            if x >= max_count_hor:
                max_count_hor = x
                max_i_hor = i
    #
    # # Поиск первого пикселя в найденной строке, чье значение равно 255
    # for j in range(dilation_hor.shape[1]):
    #     if dilation_hor[max_i_hor][j] == 255:
    #         j_hor_start = j
    #         break
    # # Поиск последнего пикселя в найденной строке, чье значение равно 255
    # for j in range(dilation_hor.shape[1] - 1, 0, -1):
    #     if dilation_hor[max_i_hor][j] == 255:
    #         j_hor_end = j
    #         break
    #
    #
    # print("max_i_hor", max_i_hor)
    # # print("max_count_hor", max_count_hor)
    # # print("j_hor_start", j_hor_start)
    # # print("j_hor_end", j_hor_end)
    #
    # # Достраиваем линии, которые находятся под самой последней длинной линией
    # for i in range(max_i_hor + 1, dilation_hor.shape[0]):
    #     for j in range(dilation_hor.shape[1]):
    #         if dilation_hor[i][j] == 255:
    #             for j in range(j_hor_start, j_hor_end):
    #                 dilation_hor[i][j] = 255

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
    print("all_hor_start_end_lines", all_hor_start_end_lines)
    # print("count_hor_start_end_lines_i", count_hor_start_end_lines_i)
    print("min_start_hor_elem", min_start_hor_elem)
    print("max_start_hor_elem", max_start_hor_elem)
    print("max_i_hor", max_i_hor)

    # # Достраиваем линии, которые находятся под самой последней длинной линией
    # for i in range(max_i_hor + 1, dilation_hor.shape[0]):
    #     for j in range(dilation_hor.shape[1]):
    #         if dilation_hor[i][j] == 255:
    #             for j in range(min_start_hor_elem, max_start_hor_elem):
    #                 dilation_hor[i][j] = 255


    # Утолщение горизонтальных линий
    count_hor_start_end_lines_i = []
    count_hor_start_lines_j = []
    count_hor_end_lines_i = []
    count_hor_end_lines_j = []

    for i in range(thick_dilation_hor.shape[0]):
        count = 0
        for j in range(thick_dilation_hor.shape[1]):
            if thick_dilation_hor[i][j] == 255:
                count += 1
        if count >= 10:
            for j in range(thick_dilation_hor.shape[1]):
                if thick_dilation_hor[i][j] == 255:
                    count_hor_start_end_lines_i.append(i)
                    count_hor_start_lines_j.append(j)
                    break
            for k in range(thick_dilation_hor.shape[1] - 1, 0, -1):
                if thick_dilation_hor[i][k] == 255:
                    count_hor_end_lines_j.append(k)
                    break
        else:
            if len(count_hor_start_lines_j) != 0 and len(count_hor_end_lines_j) != 0:
                min_start_hor_lines = min(count_hor_start_lines_j)
                max_end_hor_lines = max(count_hor_end_lines_j)
                for p in count_hor_start_end_lines_i:
                    for j in range(min_start_hor_lines, max_end_hor_lines + 1):
                        thick_dilation_hor[p][j] = 255
                count_hor_start_end_lines_i = []
                count_hor_start_lines_j = []
                count_hor_end_lines_j = []


    print("stop")
    # # Утолщение горизонтальных линий V.2
    # count_hor_start_end_lines_i = []
    # count_hor_start_lines_j = []
    # count_hor_end_lines_i = []
    # count_hor_end_lines_j = []
    #
    # for i in range(85, dilation_hor.shape[0]):
    #     count = 0
    #     for j in range(dilation_hor.shape[1]):
    #         if dilation_hor[i][j] == 255:
    #             count += 1
    #     if count >= 10:
    #         for j in range(dilation_hor.shape[1]):
    #             if dilation_hor[i][j] == 255:
    #                 count_hor_start_end_lines_i.append(i)
    #                 count_hor_start_lines_j.append(j)
    #                  break
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

    cv2.imshow("thick_dilation_hor", thick_dilation_hor)
    cv2.waitKey(0)

    # Удаление лишних
    # max_hor_1 = 0
    # max_hor_2 = 0
    # count = 0
    # flag1 = False
    # flag2 = False
    # for i in range(dilation_hor.shape[0]):
    #         if i % 2 == 0:
    #             max_hor_1 = 0
    #             count += 1
    #             flag1 = True
    #             for j in range(dilation_hor.shape[1]):
    #                 if dilation_hor[i][j] == 255:
    #                     max_hor_1 += 1
    #             if max_hor_1 > 0:
    #                 max_hor_1i = i
    #             else:
    #                 max_hor_1 = 0
    #                 max_hor_1i = i
    #                 flag1 = False
    #
    #         if i % 2 == 1:
    #             max_hor_2 = 0
    #             count += 1
    #             flag2 = True
    #             for j in range(dilation_hor.shape[1]):
    #                 if dilation_hor[i][j] == 255:
    #                     max_hor_2 += 1
    #             if max_hor_2 > 0:
    #                 max_hor_2i = i
    #             else:
    #                 max_hor_2 = 0
    #                 max_hor_2i = i
    #                 flag2 = False
    #
    #
    #         if flag1 and flag2:
    #             if max_hor_1 > max_hor_2:
    #                 for j in range(dilation_hor.shape[1]):
    #                     dilation_hor[max_hor_2i][j] = 0
    #                 max_hor_2 = 0
    #                 flag2 = False
    #
    #             if max_hor_2 > max_hor_1:
    #                 for j in range(dilation_hor.shape[1]):
    #                     dilation_hor[max_hor_1i][j] = 0
    #                 max_hor_1 = 0
    #                 flag1 = False
    #
    #         # if count == 2:
    #         #     count = 0
    #         #     # max_hor_1 = 0
    #         #     # max_hor_2 = 0
    #         #     # max_hor_1i = 0
    #         #     # max_hor_2i = 0
    #         #     # flag1 = False
    #         #     # flag2 = False
    #
    #
    # # dilation_hor
    # cv2.imshow("dilation_hor_2", dilation_hor)
    # cv2.waitKey(0)

    ver_kernel = np.ones((15, 1), np.uint8)
    erosion_ver = cv2.erode(thresh1, ver_kernel, iterations=1)
    # cv2.imshow("erosion_ver", erosion_ver)
    # cv2.waitKey(0)
    dilation_ver = cv2.dilate(erosion_ver, ver_kernel, iterations=1)
    cv2.imshow("dilation_ver", dilation_ver)
    cv2.waitKey(0)
    thick_dilation_ver = copy.deepcopy(dilation_ver)

    # # Поиск пикселя начала первой вертикальной линии
    # first_ver_j = 0
    # for j in range(dilation_ver.shape[1]):
    #     for i in range(dilation_ver.shape[0]):
    #         if dilation_ver[i][j] == 255:
    #             first_ver_j = j
    #             break
    #     if first_ver_j != 0:
    #         break
    #
    # # Поиск последнего пикселя в найденной вертикальной линии, чье значение равно 255
    # for i in range(dilation_ver.shape[0] - 1, 0, -1):
    #     if dilation_ver[i][first_ver_j] == 255:
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
    # for i in range(last_ver_i, max_last_ver_i):
    #     dilation_ver[i][first_ver_j] = 255
    #
    #
    # # print("max_last_ver_i", max_last_ver_i)
    # # print("first_ver_j", first_ver_j)
    # # print("last_ver_i", last_ver_i)
    # # print("last_ver_j", last_ver_j)


    # # Утолщение вертикальных линий
    # w = dilation_ver.shape[1]
    # h = dilation_ver.shape[0]
    # count_ver_start_end_lines_j = []
    # count_ver_start_lines_i = []
    # count_ver_end_lines_i = []
    # count_ver_lines = 0
    # for j in range(thick_dilation_ver.shape[1]):
    #     count = 0
    #     for i in range(thick_dilation_ver.shape[0]):
    #         if thick_dilation_ver[i][j] == 255:
    #             count += 1
    #     if count >= 10:
    #         for i in range(thick_dilation_ver.shape[0]):
    #             if thick_dilation_ver[i][j] == 255:
    #                 count_ver_start_end_lines_j.append(j)
    #                 count_ver_start_lines_i.append(i)
    #                 break
    #         for k in range(thick_dilation_ver.shape[0] - 1, 0, -1):
    #             if thick_dilation_ver[k][j] == 255:
    #                 count_ver_end_lines_i.append(k)
    #                 break
    #     else:
    #         if len(count_ver_start_lines_i) != 0 and len(count_ver_end_lines_i) != 0:
    #             min_start_ver_lines = min(count_ver_start_lines_i)
    #             max_end_ver_lines = max(count_ver_end_lines_i)
    #             for i in range(min_start_ver_lines, max_end_ver_lines + 1):
    #                 for p in count_ver_start_end_lines_j:
    #                     thick_dilation_ver[i][p] = 255
    #             count_ver_start_end_lines_j = []
    #             count_ver_start_lines_i = []
    #             count_ver_end_lines_i = []
    # print("stop")
    # # dilation_ver
    # cv2.imshow("thick_dilation_ver", thick_dilation_ver)
    # cv2.waitKey(0)

    img_vh = cv2.addWeighted(dilation_ver, 0.5, dilation_hor, 0.5, 0.0)
    cv2.imshow("table", img_vh)
    cv2.waitKey(0)

    # # Удаление лишних линий по горизонтали
    # cout_pixels_hor = []
    # number_lines = []
    # for i in range(img_vh.shape[0]):
    #     count = 0
    #     for j in range(img_vh.shape[1]):
    #         if img_vh[i][j] == 255:
    #             count += 1
    #
    #     if count != 0:
    #         number_lines.append(i)
    #         cout_pixels_hor.append(count)
    #
    #     if count == 0 and len(cout_pixels_hor) != 0:
    #         count_max_pixels_hor = max(cout_pixels_hor)
    #         for m in range(len(cout_pixels_hor)):
    #             if count_max_pixels_hor > cout_pixels_hor[m]:
    #                 for j in range(img_vh.shape[1]):
    #                     img_vh[number_lines[m]][j] = 0
    #         cout_pixels_hor = []
    #         number_lines = []
    #
    # # cv2.imshow("img_vh_1", img_vh)
    # # cv2.waitKey(0)
    #
    # # Удаление лишних линий по вертикали
    # cout_pixels_ver = []
    # number_columns = []
    # for j in range(img_vh.shape[1]):
    #     count = 0
    #     for i in range(img_vh.shape[0]):
    #         if img_vh[i][j] == 255:
    #             count += 1
    #     if count != 0:
    #         number_columns.append(j)
    #         cout_pixels_ver.append(count)
    #
    #     if count == 0 and len(cout_pixels_ver) != 0:
    #         count_max_pixels_ver = max(cout_pixels_ver)
    #         for m in range(len(cout_pixels_ver)):
    #             if count_max_pixels_ver > cout_pixels_ver[m]:
    #                 for i in range(img_vh.shape[0]):
    #                     img_vh[i][number_columns[m]] = 0
    #         cout_pixels_ver = []
    #         number_columns = []
    #
    # for i in range(img_vh.shape[0]):
    #     for j in range(img_vh.shape[1]):
    #         if img_vh[i][j] != 255:
    #             img_vh[i][j] = 0
    print("stop")
    # img_vh
    _, img_vh = cv2.threshold(img_vh, 100, 255, cv2.THRESH_BINARY)

    img_vh = cv2.bitwise_not(img_vh)
    cv2.imshow("img_vh", img_vh)
    cv2.waitKey(0)

    # crossing_lines = cv2.bitwise_and(dilation_hor, dilation_ver)
    # cv2.imshow("crossing_lines", crossing_lines)
    # cv2.waitKey(0)

    # for i in range(img_vh.shape[0] - 2):
    #     for j in range(img_vh.shape[1] - 2):
    #         if img_vh[i][j] == 0:
    #             continue
    #         else:
    #             if img_vh[i][j] == img_vh[i][j + 1]:
    #                 img_vh[i][j + 1] = 0
    #             if img_vh[i][j] == img_vh[i][j + 2]:
    #                 img_vh[i][j + 2] = 0
    #             if img_vh[i][j] == img_vh[i + 1][j]:
    #                 img_vh[i + 1][j] = 0
    #             if img_vh[i][j] == img_vh[i + 2][j]:
    #                 img_vh[i + 2][j] = 0
    #             if img_vh[i][j] == img_vh[i + 1][j + 1]:
    #                 img_vh[i + 1][j + 1] = 0
    #             if img_vh[i][j] == img_vh[i + 1][j + 2]:
    #                 img_vh[i + 1][j + 2] = 0
    #             if img_vh[i][j] == img_vh[i + 2][j + 1]:
    #                 img_vh[i + 2][j + 1] = 0
    #             if img_vh[i][j] == img_vh[i + 2][j + 2]:
    #                 img_vh[i + 2][j + 2] = 0

    # img_vh
    # crossing_lines
    # cv2.imshow("crossing_lines_1", img_vh)
    # cv2.waitKey(0)

    # # Выньте логотип focus
    # ys, xs = np.where(crossing_lines > 0)
    # print("ys: ", ys)
    # print("xs: ", xs)
    #
    # # Массив горизонтальных и вертикальных координат
    # y_point_arr = []
    # x_point_arr = []
    # # При сортировке похожие пиксели исключаются, и берется только последняя точка с похожими значениями
    # # Это 10 - это расстояние между двумя пикселями. Оно не фиксировано. Оно будет регулироваться в соответствии с разными изображениями. Это в основном высота (переходы по координате y) и длина (переходы по координате x) таблицы ячеек
    # i = 0
    # sort_x_point = np.sort(xs)
    # for i in range(len(sort_x_point) - 1):
    #     if sort_x_point[i + 1] - sort_x_point[i] > 10:
    #         x_point_arr.append(sort_x_point[i])
    #     i = i + 1
    # # Чтобы добавить последнюю точку
    # x_point_arr.append(sort_x_point[i])
    #
    # i = 0
    # sort_y_point = np.sort(ys)
    # # print(np.sort(ys))
    # for i in range(len(sort_y_point) - 1):
    #     if (sort_y_point[i + 1] - sort_y_point[i] > 10):
    #         y_point_arr.append(sort_y_point[i])
    #     i = i + 1
    # y_point_arr.append(sort_y_point[i])
    #
    # # Цикл координаты y, таблица разделения координат x
    # data = [[] for i in range(len(y_point_arr))]
    # for i in range(len(y_point_arr) - 1):
    #     for j in range(len(x_point_arr) - 1):
    #         # При делении первый параметр - это координата y, а второй параметр - координата x
    #         cell = img[y_point_arr[i]:y_point_arr[i + 1], x_point_arr[j]:x_point_arr[j + 1]]
    #         cv2.imshow("sub_pic" + str(i) + str(j), cell)
    #         cv2.waitKey(0)

    print("stop")
    # # bitxor = cv2.bitwise_xor(img_bin, img_vh)
    # #     cv2.imshow("bitxor", bitxor)
    # #     cv2.waitKey(0)
    # #
    # #     bitnot = cv2.bitwise_not(bitxor)
    # #     # Plotting the generated image
    # #     cv2.imshow("bitnot", bitnot)
    # #     cv2.waitKey(0)
    # #
    # #
    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w < 0.9 * img_vh.shape[1] and h < 0.9 * img_vh.shape[0]:
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # cv2.imshow("image", img)
            # cv2.waitKey(0)

    # # # print(len(contours))
    # # # print(contours[0])
    # # # print(len(contours[0]))
    # # # print(cv2.boundingRect(contours[0]))
    # #
    # # def sort_contours(cnts, method="left-to-right"):
    # #     # initialize the reverse flag and sort index
    # #     reverse = False
    # #     i = 0
    # #     # handle if we need to sort in reverse
    # #     if method == "right-to-left" or method == "bottom-to-top":
    # #         reverse = True
    # #     # handle if we are sorting against the y-coordinate rather than
    # #     # the x-coordinate of the bounding box
    # #     if method == "top-to-bottom" or method == "bottom-to-top":
    # #         i = 1
    # #     # construct the list of bounding boxes and sort them from top to
    # #     # bottom
    # #     boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # #     cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # #     # return the list of sorted contours and bounding boxes
    # #     return cnts, boundingBoxes
    # #
    # # # Sort all the contours by top to bottom.
    # # contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    # #
    # # # Creating a list of heights for all detected boxes
    # # heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    # #
    # # # Get mean of heights
    # # mean = np.mean(heights)
    # #
    # # # Create list box to store all boxes in
    # # box = []
    # # # Get position (x,y), width and height for every contour and show the contour on image
    # # # print("lencontours", len(contours))
    # # for c in contours:
    # #     x, y, w, h = cv2.boundingRect(c)
    # #     # print("x", x, "y", y, "w", w, "h", h)
    # #     if (w < 0.9 * img_vh.shape[1] and h < 0.9 * img_vh.shape[0]):
    # #         image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # #         box.append([x, y, w, h])
    # #
    cv2.imshow("image", image)
    cv2.waitKey(0)
    # #
    # # # Creating two lists to define row and column in which cell is located
    # # row = []
    # # column = []
    # # j = 0
    # #
    # # # print("len box", len(box))
    # # # Sorting the boxes to their respective row and column
    # # for i in range(len(box)):
    # #     if (i == 0):
    # #         column.append(box[i])
    # #         previous = box[i]
    # #
    # #     else:
    # #         if (box[i][1] <= previous[1] + mean / 2):
    # #             column.append(box[i])
    # #             previous = box[i]
    # #
    # #             if (i == len(box) - 1):
    # #                 row.append(column)
    # #
    # #         else:
    # #             row.append(column)
    # #             column = []
    # #             previous = box[i]
    # #             column.append(box[i])
    # #
    # # # print(column)
    # # # print(row)
    # #
    # # # calculating maximum number of cells
    # # countcol = 0
    # # index = 0
    # # for i in range(len(row)):
    # #     current = len(row[i])
    # #     # print("len",len(row[i]))
    # #     if current > countcol:
    # #         countcol = current
    # #         index = i
    # #
    # # # print("countcol", countcol)
    # #
    # # # Retrieving the center of each column
    # # # center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    # # center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]
    # # # print("center",center)
    # #
    # # center = np.array(center)
    # # center.sort()
    # # # print("center.sort()", center)
    # # # Regarding the distance to the columns center, the boxes are arranged in respective order
    # #
    # # finalboxes = []
    # # for i in range(len(row)):
    # #     lis = []
    # #     for k in range(countcol):
    # #         lis.append([])
    # #     for j in range(len(row[i])):
    # #         diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
    # #         minimum = min(diff)
    # #         indexing = list(diff).index(minimum)
    # #         lis[indexing].append(row[i][j])
    # #     finalboxes.append(lis)
    # #
    # # # return finalboxes, img_bin
