import cv2
from pdf2image import convert_from_path
# https://github.com/Belval/pdf2image
from Work.tableDetection import tableDetect

from Work.tableStructerRecognationV4 import recognizeStructerV4
from Work.rotateDocumentV4 import rotateDocumentV4

file_pdf = 'pdf/..'

pages = convert_from_path(file_pdf, 166)  # 96
print(pages)
for i, page in enumerate(pages):
    page.save(f'pictures/img{i}.jpg', 'JPEG')

file_img = 'pictures/img0.jpg'

img = cv2.imread(file_img)
cv2.imshow("img", img)
cv2.waitKey(0)

rotate_img = rotateDocumentV4(img)
# cv2.imshow("rotate", rotate_img)
# cv2.waitKey(0)

rotate_img, table_list = tableDetect(rotate_img)
cv2.imshow("tableDetect", rotate_img)
cv2.waitKey(0)

for table in table_list:
    # pass
    # recognizeStructer(table)
    recognizeStructerV4(table)

# for table in range(2, len(table_list)):
#     # recognizeStructer(table)
#     recognizeStructerV4(table_list[table])

# print(len(table_list))
