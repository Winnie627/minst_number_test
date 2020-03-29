import imageio
import glob
import cv2
image = cv2.imread("D:/Pycharmproject/minst_number/mydata_1.png",0)
image_a = cv2.resize(image, (28, 28))
cv2.imshow("img", image_a)