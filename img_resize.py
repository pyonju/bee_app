import cv2
import os

path_wasp = os.listdir('./images/wasp_origin/')
path_paper_wasp = os.listdir('./images/paper_wasp_origin/')

for i in range(len(path_wasp)):
    if path_wasp[i].split('.')[1] == 'jpg':
        img = cv2.imread('./images/wasp_origin/' + path_wasp[i])
        img = cv2.resize(img, (100,100))
        cv2.imwrite('./images/wasp/' + path_wasp[i], img)

for i in range(len(path_paper_wasp)):
    if path_paper_wasp[i].split('.')[1] == 'jpg':
        img = cv2.imread('./images/paper_wasp_origin/' + path_paper_wasp[i])
        img = cv2.resize(img, (100,100))
        cv2.imwrite('./images/paper_wasp/' + path_paper_wasp[i], img)
