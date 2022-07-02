import cv2
import numpy as np

# Степень конверсии
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res

# Рассчитать угол с помощью преобразования Хафа
def CalcDegree(srcImage):
    midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    # new_img = cv2.threshold(midImage, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Imagelines", new_img)
    # cv2.waitKey(0)

    dstImage = cv2.Canny(midImage, 50, 200, 3)
    # cv2.imshow("dstImage", dstImage)
    # cv2.waitKey(0)
    lineimage = srcImage.copy()

    # Обнаружение прямых линий по преобразованию Хафа
    # Четвертый параметр - это порог, чем больше порог, тем выше точность обнаружения
    # lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 140, srn=0, stn=0,  min_theta=1.39626, max_theta=1.74533) #1.562
    # lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 140, srn=0, stn=0,  min_theta=1.5708, max_theta=1.65806) #1.562 право
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 140, srn=0, stn=0,  min_theta=1.5708, max_theta=1.74533) #1.562 право


    # lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 200) #1.562

    # Из-за разных изображений порог установить нелегко, так как он установлен слишком высоко, поэтому линия не может быть обнаружена, порог слишком низкий, линия слишком большая, скорость очень низкая
    sum = 0
    # Нарисуйте каждый отрезок по очереди
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
            # В качестве угла поворота выберите только наименьший угол
            sum += theta
            cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.imshow("Imagelines", lineimage)
            # cv2.waitKey(0)

    # Усредняя все углы, эффект вращения будет лучше
    average = sum / len(lines)
    angle = DegreeTrans(average) - 90
    print(angle)
    # Центр вращения является центром изображения
    h, w = srcImage.shape[:2]
    # Рассчитать 2D повернутую матрицу аффинного преобразования
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle / 2, 1)
    # print(RotateMatrix)
    # Аффинное преобразование, цвет фона заполнен белым
    rotate = cv2.warpAffine(srcImage, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate
