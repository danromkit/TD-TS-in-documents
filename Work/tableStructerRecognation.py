import cv2
import numpy as np


def recognizeStructer(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("rotate_gray", img_gray)
    # cv2.waitKey(0)

    _, thresh = cv2.threshold(img_gray, 195, 255, cv2.THRESH_BINARY)
    # cv2.imshow("img_bin", thresh)
    # cv2.waitKey(0)

    thresh1 = cv2.bitwise_not(thresh)
    # cv2.imshow("thresh_1", thresh1)
    # cv2.waitKey(0)

    hor_kernel = np.ones((1, 27), np.uint8)
    erosion_hor = cv2.erode(thresh1, hor_kernel, iterations=1)
    # cv2.imshow("erosion_hor", erosion_hor)
    # cv2.waitKey(0)
    dilation_hor = cv2.dilate(erosion_hor, hor_kernel, iterations=1)
    # cv2.imshow("dilation_hor", dilation_hor)
    # cv2.waitKey(0)


    #Утолщение горизонтальных линий
    count_hor_start_end_lines_i = []
    count_hor_start_lines_j = []
    count_hor_end_lines_i = []
    count_hor_end_lines_j = []
    count_hor_lines = 0
    for i in range(dilation_hor.shape[0]):
        count = 0
        for j in range(dilation_hor.shape[1]):
            if dilation_hor[i][j] == 255:
                count += 1
        if count >= 10:
            count_hor_lines += 1
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
                for p in count_hor_start_end_lines_i:
                    for j in range(min_start_hor_lines, max_end_hor_lines + 1):
                        dilation_hor[p][j] = 255
                count_hor_start_end_lines_i = []
                count_hor_start_lines_j = []
                count_hor_end_lines_j = []

    # dilation_hor
    cv2.imshow("dilation_hor", dilation_hor)
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

    # cv2.imshow("dilation_ver", dilation_ver)
    # cv2.waitKey(0)

    # Утолщение вертикальных линий
    w = dilation_ver.shape[1]
    h = dilation_ver.shape[0]
    count_ver_start_end_lines_j = []
    count_ver_start_lines_i = []
    count_ver_end_lines_i = []
    count_ver_lines = 0
    for j in range(dilation_ver.shape[1]):
        count = 0
        for i in range(dilation_ver.shape[0]):
            if dilation_ver[i][j] == 255:
                count += 1
        if count >= 10:
            for i in range(dilation_ver.shape[0]):
                if dilation_ver[i][j] == 255:
                    count_ver_start_end_lines_j.append(j)
                    count_ver_start_lines_i.append(i)
                    break
            for k in range(dilation_ver.shape[0] - 1, 0, -1):
                if dilation_ver[k][j] == 255:
                    count_ver_end_lines_i.append(k)
                    break
        else:
            if len(count_ver_start_lines_i) != 0 and len(count_ver_end_lines_i) != 0:
                min_start_ver_lines = min(count_ver_start_lines_i)
                max_end_ver_lines = max(count_ver_end_lines_i)
                for i in range(min_start_ver_lines, max_end_ver_lines + 1):
                    for p in count_ver_start_end_lines_j:
                        dilation_ver[i][p] = 255
                count_ver_start_end_lines_j = []
                count_ver_start_lines_i = []
                count_ver_end_lines_i = []

    # dilation_ver
    cv2.imshow("dilation_ver", dilation_ver)
    cv2.waitKey(0)

    img_vh = cv2.addWeighted(dilation_ver, 0.5, dilation_hor, 0.5, 0.0)
    # cv2.imshow("img_vh", img_vh)
    # cv2.waitKey(0)

    # Удаление лишних линий по горизонтали
    cout_pixels_hor = []
    number_lines = []
    for i in range(img_vh.shape[0]):
        count = 0
        for j in range(img_vh.shape[1]):
            if img_vh[i][j] == 255:
                count += 1

        if count != 0:
            number_lines.append(i)
            cout_pixels_hor.append(count)

        if count == 0 and len(cout_pixels_hor) != 0:
            count_max_pixels_hor = max(cout_pixels_hor)
            for m in range(len(cout_pixels_hor)):
                if count_max_pixels_hor > cout_pixels_hor[m]:
                    for j in range(img_vh.shape[1]):
                        img_vh[number_lines[m]][j] = 0
            cout_pixels_hor = []
            number_lines = []

    # cv2.imshow("img_vh_1", img_vh)
    # cv2.waitKey(0)

    # Удаление лишних линий по вертикали
    cout_pixels_ver = []
    number_columns = []
    for j in range(img_vh.shape[1]):
        count = 0
        for i in range(img_vh.shape[0]):
            if img_vh[i][j] == 255:
                count += 1
        if count != 0:
            number_columns.append(j)
            cout_pixels_ver.append(count)

        if count == 0 and len(cout_pixels_ver) != 0:
            count_max_pixels_ver = max(cout_pixels_ver)
            for m in range(len(cout_pixels_ver)):
                if count_max_pixels_ver > cout_pixels_ver[m]:
                    for i in range(img_vh.shape[0]):
                        img_vh[i][number_columns[m]] = 0
            cout_pixels_ver = []
            number_columns = []

    for i in range(img_vh.shape[0]):
        for j in range(img_vh.shape[1]):
            if img_vh[i][j] != 255:
                img_vh[i][j] = 0

    # # img_vh
    cv2.imshow("img_vh", img_vh)
    cv2.waitKey(0)

    # crossing_lines = cv2.bitwise_and(dilation_hor, dilation_ver)
    # cv2.imshow("crossing_lines", crossing_lines)
    # cv2.waitKey(0)




    for i in range(img_vh.shape[0] - 2):
        for j in range(img_vh.shape[1] - 2):
            if img_vh[i][j] == 0:
                continue
            else:
                if img_vh[i][j] == img_vh[i][j + 1]:
                    img_vh[i][j + 1] = 0
                if img_vh[i][j] == img_vh[i][j + 2]:
                    img_vh[i][j + 2] = 0
                if img_vh[i][j] == img_vh[i + 1][j]:
                    img_vh[i + 1][j] = 0
                if img_vh[i][j] == img_vh[i + 2][j]:
                    img_vh[i + 2][j] = 0
                if img_vh[i][j] == img_vh[i + 1][j + 1]:
                    img_vh[i + 1][j + 1] = 0
                if img_vh[i][j] == img_vh[i + 1][j + 2]:
                    img_vh[i + 1][j + 2] = 0
                if img_vh[i][j] == img_vh[i + 2][j + 1]:
                    img_vh[i + 2][j + 1] = 0
                if img_vh[i][j] == img_vh[i + 2][j + 2]:
                    img_vh[i + 2][j + 2] = 0

    # img_vh
    # crossing_lines
    cv2.imshow("crossing_lines_1", img_vh)
    cv2.waitKey(0)




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




