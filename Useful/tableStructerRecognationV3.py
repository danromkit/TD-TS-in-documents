import numpy as np
import cv2


def recognizeStructerV3(img):
    # tess.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    # print("img_height", img_height, "img_width", img_width)
    cv2.imshow("img", img)
    cv2.waitKey(0)

    # thresholding the image to a binary image
    # thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    # thresh, img_bin = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", img_bin)
    # cv2.waitKey(0)

    # contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # invert = False
    # invert = False
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     # print("x", x, "y", y, "w", w, "h", h)
    #     if (w < 0.9 * img_width and h < 0.9 * img_height and (
    #             w > max(10, img_width / 30) and h > max(10, img_height / 30))):
    #         # if(w*h > 100 and (w < 0.9 * img_width and h < 0.9*img_height)):
    #         # invert = True
    #         invert = True
    #         # print("size", img_width * img_height)
    #         # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         img_bin[y:y + h, x:x + w] = 255 - img_bin[y:y + h, x:x + w]

    cv2.imshow("img_bin", img_bin)
    cv2.waitKey(0)

    # img_bin = 255 - img_bin if (invert) else img_bin

    # cv2.imshow("img_bin", img_bin)
    # cv2.waitKey(0)

    img_bin_inv = cv2.bitwise_not(img_bin)
    cv2.imshow("img_bin_inv", img_bin_inv)
    cv2.waitKey(0)

    ############################################################################################################################################

    kernel_len_ver = max(10, img_height // 50)
    kernel_len_hor = max(10, img_width // 50)
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))  # shape (kernel_len, 1) inverted! xD
    # print("ver", ver_kernel)
    # print(ver_kernel.shape)

    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))  # shape (1,kernel_ken) xD
    # print("hor", hor_kernel)
    # print(hor_kernel.shape)

    # A kernel of 2x2
    kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))

    # print(kernel)
    # print(kernel.shape)

    # Use vertical kernel to detect and save the vertical lines in a jpg
    erosion_ver = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(erosion_ver, ver_kernel, iterations=4)

    # Утолщение вертикальных линий
    w = vertical_lines.shape[1]
    h = vertical_lines.shape[0]
    count_ver_start_end_lines_j = []
    count_ver_start_lines_i = []
    count_ver_end_lines_i = []
    count_ver_lines = 0
    for j in range(vertical_lines.shape[1]):
        count = 0
        for i in range(vertical_lines.shape[0]):
            if vertical_lines[i][j] == 255:
                count += 1
        if count >= 10:
            for i in range(vertical_lines.shape[0]):
                if vertical_lines[i][j] == 255:
                    count_ver_start_end_lines_j.append(j)
                    count_ver_start_lines_i.append(i)
                    break
            for k in range(vertical_lines.shape[0] - 1, 0, -1):
                if vertical_lines[k][j] == 255:
                    count_ver_end_lines_i.append(k)
                    break
        else:
            if len(count_ver_start_lines_i) != 0 and len(count_ver_end_lines_i) != 0:
                min_start_ver_lines = min(count_ver_start_lines_i)
                max_end_ver_lines = max(count_ver_end_lines_i)
                for i in range(min_start_ver_lines, max_end_ver_lines + 1):
                    for p in count_ver_start_end_lines_j:
                        vertical_lines[i][p] = 255
                count_ver_start_end_lines_j = []
                count_ver_start_lines_i = []
                count_ver_end_lines_i = []

    # # Поиск крайней верхней и крайней нижней точки (i)
    # ver_start_lines_i = []
    # ver_end_lines_i = []
    # ver_start_end_lines_j = []
    # all_ver_start_end_lines = []
    #
    # for j in range(vertical_lines.shape[1]):
    #     count = 0
    #     for i in range(vertical_lines.shape[0]):
    #         if vertical_lines[i][j] == 255:
    #             count += 1
    #     if count >= 10:
    #         for i in range(vertical_lines.shape[0]):
    #             if vertical_lines[i][j] == 255:
    #                 ver_start_lines_i.append(i)  # Здесь хранится значения i
    #                 ver_start_end_lines_j.append(j)
    #                 break
    #         for k in range(vertical_lines.shape[0] - 1, 0, -1):
    #             if vertical_lines[k][j] == 255:
    #                 ver_end_lines_i.append(k)
    #                 break
    #     else:
    #         if len(ver_start_lines_i) != 0 and len(ver_end_lines_i) != 0:
    #             min_start_ver_lines = min(ver_start_lines_i)
    #             max_end_ver_lines = max(ver_end_lines_i)
    #
    #             all_ver_start_end_lines.append(min_start_ver_lines)
    #             all_ver_start_end_lines.append(max_end_ver_lines)
    #
    #             ver_start_lines_i = []
    #             ver_end_lines_i = []
    #             ver_start_end_lines_j = []
    #
    # min_start_ver_elem_vertical = min(all_ver_start_end_lines)
    # max_start_ver_elem_vertical = max(all_ver_start_end_lines)
    # print("min_start_ver_elem_vertical: ", min_start_ver_elem_vertical)
    # print("max_start_ver_elem_vertical: ", max_start_ver_elem_vertical)
    #
    # # Поиск пикселя начала первой вертикальной линии
    # first_ver_j = []
    # for j in range(vertical_lines.shape[1]):
    #     count = 0
    #     for i in range(vertical_lines.shape[0]):
    #         if vertical_lines[i][j] == 255:
    #             count += 1
    #     if count != 0:
    #         first_ver_j.append(j)
    #     else:
    #         if count == 0 and len(first_ver_j) != 0:
    #             break
    #
    # print("first_ver_j", first_ver_j)
    #
    # # Поиск пикселя начала последней вертикальной линии
    # last_ver_j = []
    # for j in range(vertical_lines.shape[1] - 1, 0, -1):
    #     count = 0
    #     for i in range(vertical_lines.shape[0]):
    #         if vertical_lines[i][j] == 255:
    #             count += 1
    #     if count != 0:
    #         last_ver_j.append(j)
    #     else:
    #         if count == 0 and len(last_ver_j) != 0:
    #             break
    #
    # min_last_ver_j = min(last_ver_j)
    # max_last_ver_j = max(first_ver_j)
    # # print("min_last_ver_j", min_last_ver_j)
    # # print("max_last_ver_j", max_last_ver_j)
    # for i in range(min_start_ver_elem_vertical, max_start_ver_elem_vertical + 1):
    #     vertical_lines[i][min_last_ver_j] = 255
    # for i in range(min_start_ver_elem_vertical, max_start_ver_elem_vertical + 1):
    #     vertical_lines[i][max_last_ver_j] = 255

    # Plot the generated image
    # cv2.imshow("erosion_ver", erosion_ver)
    # cv2.waitKey(0)

    cv2.imshow("vertical_lines", vertical_lines)
    cv2.waitKey(0)

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    erosion_hor = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(erosion_hor, hor_kernel, iterations=5)

    # # Утолщение горизонтальных линий
    # count_hor_start_end_lines_i = []
    # count_hor_start_lines_j = []
    # count_hor_end_lines_i = []
    # count_hor_end_lines_j = []
    #
    # for i in range(horizontal_lines.shape[0]):
    #     count = 0
    #     for j in range(horizontal_lines.shape[1]):
    #         if horizontal_lines[i][j] == 255:
    #             count += 1
    #     if count >= 10:
    #         for j in range(horizontal_lines.shape[1]):
    #             if horizontal_lines[i][j] == 255:
    #                 count_hor_start_end_lines_i.append(i)
    #                 count_hor_start_lines_j.append(j)
    #                 break
    #         for k in range(horizontal_lines.shape[1] - 1, 0, -1):
    #             if horizontal_lines[i][k] == 255:
    #                 count_hor_end_lines_j.append(k)
    #                 break
    #     else:
    #         if len(count_hor_start_lines_j) != 0 and len(count_hor_end_lines_j) != 0:
    #             min_start_hor_lines = min(count_hor_start_lines_j)
    #             max_end_hor_lines = max(count_hor_end_lines_j)
    #             for p in count_hor_start_end_lines_i:
    #                 for j in range(min_start_hor_lines, max_end_hor_lines + 1):
    #                     horizontal_lines[p][j] = 255
    #             count_hor_start_end_lines_i = []
    #             count_hor_start_lines_j = []
    #             count_hor_end_lines_j = []

    # Поиск крайней левой и крайней правой точки (j)
    count_hor_start_end_lines_i = []
    count_hor_start_lines_j = []
    count_hor_end_lines_i = []
    count_hor_end_lines_j = []
    all_hor_start_end_lines = []
    for i in range(horizontal_lines.shape[0]):
        count = 0
        for j in range(horizontal_lines.shape[1]):
            if horizontal_lines[i][j] == 255:
                count += 1
        if count >= 10:
            for j in range(horizontal_lines.shape[1]):
                if horizontal_lines[i][j] == 255:
                    count_hor_start_end_lines_i.append(i)
                    count_hor_start_lines_j.append(j)
                    break
            for k in range(horizontal_lines.shape[1] - 1, 0, -1):
                if horizontal_lines[i][k] == 255:
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
    for i in range(horizontal_lines.shape[0] - 1, 0, -1):
        if horizontal_lines[i][min_start_hor_elem] == 255 or horizontal_lines[i][min_start_hor_elem + 1] == 255 or \
                horizontal_lines[i][min_start_hor_elem + 2] == 255 or horizontal_lines[i][min_start_hor_elem + 3] == 255:
            max_start_ver_elem = i
            break

    for i in range(max_start_ver_elem, horizontal_lines.shape[0]):
        count = 0
        for j in range(min_start_hor_elem, max_start_hor_elem):
            if horizontal_lines[i][j] == 255:
                count += 1
        if count != 0:
            max_start_ver_elem_1 = i
        else:
            break

    # print("max_start_ver_elem: ", max_start_ver_elem)
    # print("max_start_ver_elem_1: ", max_start_ver_elem_1)

    # Дорисовка линий, которые находятся снизу
    one_hor_line = []
    for i in range(horizontal_lines.shape[0] - 1, max_start_ver_elem_1 + 1, -1):
        count = 0
        for j in range(min_start_hor_elem, max_start_hor_elem):
            if horizontal_lines[i][j] == 255:
                count += 1
        if count != 0:
            one_hor_line.append(i)
        if count == 0 and len(one_hor_line) != 0:
            max_one_hor_line = max(one_hor_line)
            for k in range(min_start_hor_elem, max_start_hor_elem):
                horizontal_lines[max_one_hor_line][k] = 255
            one_hor_line = []

        # Поиск самой верхней линии, которая начинается с минимального индекса по горизонтали (i)
    for i in range(horizontal_lines.shape[0]):
        if horizontal_lines[i][min_start_hor_elem] == 255 or horizontal_lines[i][min_start_hor_elem + 1] == 255 or \
                horizontal_lines[i][min_start_hor_elem + 2] == 255 or horizontal_lines[i][min_start_hor_elem + 3] == 255:
            max_start_ver_elem_up = i
            break

    for i in range(max_start_ver_elem_up, 0, -1):
        count = 0
        for j in range(min_start_hor_elem, max_start_hor_elem):
            if horizontal_lines[i][j] == 255:
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
            if horizontal_lines[i][j] == 255:
                count += 1
        if count != 0:
            one_hor_line.append(i)
        if count == 0 and len(one_hor_line) != 0:
            max_one_hor_line = max(one_hor_line)
            for k in range(min_start_hor_elem, max_start_hor_elem):
                horizontal_lines[max_one_hor_line][k] = 255
            one_hor_line = []

    # Plot the generated image
    # cv2.imshow("erosion_hor", erosion_hor)
    # cv2.waitKey(0)

    cv2.imshow("horizontal_lines", horizontal_lines)
    cv2.waitKey(0)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    cv2.imshow("img_vh", img_vh)
    cv2.waitKey(0)

    # # Eroding and thesholding the image
    # img_vh = cv2.dilate(img_vh, kernel_ver, iterations=3)
    img_vh = cv2.dilate(img_vh, kernel_hor, iterations=3)
    cv2.imshow("img_vh", img_vh)
    cv2.waitKey(0)

    thresh, img_vh = (cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY))

    # print("last")
    cv2.imshow("img_vh+", img_vh)
    cv2.waitKey(0)

    img_vh_black = cv2.bitwise_not(img_vh)
    cv2.imshow("img_vh_black", img_vh_black)
    cv2.waitKey(0)





    # print("img_bin")
    cv2.imshow("img_bin", img_bin)
    cv2.waitKey(0)

    # print("bitor")
    bitor = cv2.bitwise_or(img_bin, img_vh)
    cv2.imshow("bitor", bitor)
    cv2.waitKey(0)

    img_median = bitor #cv2.medianBlur(bitor, 3)
    cv2.imshow("img_median", img_median)
    cv2.waitKey(0)


    # ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, img_height * 2))  # shape (kernel_len, 1) inverted! xD
    # vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)
    # cv2.imshow("vertical_lines_1", vertical_lines)
    # cv2.waitKey(0)

    img_median = cv2.bitwise_not(img_median)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 4))
    closed = cv2.morphologyEx(img_median, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closed", closed)
    cv2.waitKey(0)
    img_median = cv2.bitwise_not(closed)


    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width * 2, 3))  # shape (kernel_len, 1) inverted! xD
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)
    cv2.imshow("horizontal_lines_1", horizontal_lines)
    cv2.waitKey(0)

    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # print(kernel)
    # print(kernel.shape)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    cv2.imshow("img_vh", img_vh)
    cv2.waitKey(0)

    cv2.imshow("~img_vh", ~img_vh)
    cv2.waitKey(0)

    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    cv2.imshow("img_vh", img_vh)
    cv2.waitKey(0)

    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("/Users/marius/Desktop/img_vh.jpg", img_vh)
    cv2.imshow("img_vh", img_vh)
    cv2.waitKey(0)

    # print("result")
    # bitor but then 5->4 or 3
    bitxor = cv2.bitwise_xor(img_bin, img_vh)
    cv2.imshow("bitxor", bitxor)
    cv2.waitKey(0)

    bitnot = cv2.bitwise_not(bitxor)
    # Plotting the generated image
    cv2.imshow("bitnot", bitnot)
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
        if (w < 0.9 * img_width and h < 0.9 * img_height):
            # image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    box.sort(key=lambda x: (x[1], x[0]))

    for i in box:
        # pass
        # textRecognation(img[i[1]:(i[1] + i[3]), i[0]: (i[0] + i[2])])
        cv2.imshow(f"box{i}", img[i[1]:(i[1] + i[3]), i[0] : (i[0] + i[2])])
        cv2.waitKey(0)

    # # Creating two lists to define row and column in which cell is located
    # row = []
    # column = []
    # j = 0
    #
    # # print("len box", len(box))
    # # Sorting the boxes to their respective row and column
    # for i in range(len(box)):
    #     if (i == 0):
    #         column.append(box[i])
    #         previous = box[i]
    #
    #     else:
    #         if (box[i][1] <= previous[1] + mean / 2):
    #             column.append(box[i])
    #             previous = box[i]
    #
    #             if (i == len(box) - 1):
    #                 row.append(column)
    #
    #         else:
    #             row.append(column)
    #             column = []
    #             previous = box[i]
    #             column.append(box[i])
    #
    # # print(column)
    # # print(row)
    #
    # # calculating maximum number of cells
    # countcol = 0
    # index = 0
    # for i in range(len(row)):
    #     current = len(row[i])
    #     # print("len",len(row[i]))
    #     if current > countcol:
    #         countcol = current
    #         index = i
    #
    # # print("countcol", countcol)
    #
    # # Retrieving the center of each column
    # # center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    # center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]
    # # print("center",center)
    #
    # center = np.array(center)
    # center.sort()
    # # print("center.sort()", center)
    # # Regarding the distance to the columns center, the boxes are arranged in respective order
    #
    # finalboxes = []
    # for i in range(len(row)):
    #     lis = []
    #     for k in range(countcol):
    #         lis.append([])
    #     for j in range(len(row[i])):
    #         diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
    #         minimum = min(diff)
    #         indexing = list(diff).index(minimum)
    #         lis[indexing].append(row[i][j])
    #     finalboxes.append(lis)
    #
    # return finalboxes, img_bin