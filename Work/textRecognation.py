import cv2
import numpy as np
import pytesseract

recognation_text = []
table_for_df = []


def textRecognation(img, box, offset):

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # buf_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # hor_i = [0, 1, 2]
    # ver_j = [0, 1, 2]
    #
    #
    # for i in range(buf_img.shape[0]):
    #     for j in range(len(ver_j) - 1):
    #         buf_img[i][j] = 255
    #
    # for i in range(len(hor_i) - 1):
    #     for j in range(buf_img.shape[1]):
    #         buf_img[i][j] = 255



    # _, buf_img = cv2.threshold(buf_img, 170, 255, cv2.THRESH_BINARY)

    gray_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # gray_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # gray_box = cv2.resize(gray_box, (0, 0), fx=2, fy=2)
    # cv2.imshow("big", gray_box)
    # cv2.waitKey(0)







    config = r'--oem 3 --psm 6'

    img_text = pytesseract.image_to_string(gray_box, lang='rus+eng', config=config)
    print(img_text)
    cv2.imshow("big", gray_box)
    cv2.waitKey(0)

    # recognation_text.append(img_text)
    # return recognation_text

    # print("Первая попытка: ", img_text)
    # img_text_1 = pytesseract.image_to_boxes(big, lang='rus+eng', config=config)
    # # print("Вторая попытка: ", img_text_1)

    # # Detect
    # himg, wimg, _ = big.shape
    # boxes = pytesseract.image_to_boxes(big)
    #
    # for b in boxes.splitlines():
    #     b = b.split(' ')
    #     # print(b)
    #     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    #     cv2.rectangle(big, (x, himg - y), (x + w, himg - h), (0, 0, 255), 1)
    #     cv2.putText(big, b[0], (x, himg - y + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
    #
    # cv2.imshow("big_result", big)
    # cv2.waitKey(0)

    # # img = cv2.resize(img, (0, 0), fx=4, fy=4)
    # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grayImageBlur = cv2.blur(grayImage, (3, 3))
    # edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)
    #
    # cv2.imshow("edgedImage", edgedImage)
    # cv2.waitKey(0)

def textRecognationV1Row1(img, row):
    config = r'--oem 3 --psm 6'
    row_for_df = []
    for i in row:
        img_1 = img[i[1]:(i[1] + i[3]), i[0]: (i[0] + i[2])]
        gray_box = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_text = pytesseract.image_to_string(gray_box, lang='rus+eng', config=config)
        row_for_df.append(img_text)

    return row_for_df


def textRecognationV1(img, row, columns):
    config = r'--oem 3 --psm 6'
    row_for_df = [None] * len(columns)
    sorted(row, key=lambda x: x[0])
    for i in row:
        for j in columns:
            if abs(i[0] - j) < 8:
                place = columns.index(j)
                img_1 = img[i[1]:(i[1] + i[3]), i[0]: (i[0] + i[2])]
                gray_box = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
                img_text = pytesseract.image_to_string(gray_box, lang='rus+eng', config=config)
                row_for_df[place] = img_text
                # cv2.imshow("img_1", img_1)
                # cv2.waitKey(0)
    # table_for_df.append(row_for_df)

    return row_for_df




