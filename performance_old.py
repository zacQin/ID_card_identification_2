import cv2
import pandas as pd
from face_alignment_1 import face_alignment
from face_base import find_face
from face_base import license_detection_Rough
from face_base import license_detection_Detailed
from smooth_sharpen import smooth
from smooth_sharpen import sharpen
from face_base import divide_image
from face_base import face_wipeoff
from info_divide import info_divide
from PIL import Image
import pytesseract
import numpy as np
import imutils

img = cv2.imread('image/64B44403DAE21C578A2C0B986213BC2F.jpg')

B,G,R = cv2.split(img)

width,height,layer = img.shape

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face,face_plus,img,img_gray = find_face(img)
print('secenon')
lincese ,lincese_gray = license_detection_Rough(img,img_gray,face_plus)

cv2.imshow('license3',lincese)
cv2.waitKey(0)

lincese,lincese_gray = face_alignment(lincese,lincese_gray)
cv2.imshow('license3',lincese)
cv2.waitKey(0)

face,face_plus,img,img_gray = find_face(lincese)
lincese ,lincese_gray = license_detection_Detailed(lincese,lincese_gray,face_plus)

cv2.imshow('license3',lincese)
cv2.waitKey(0)

# cv2.imshow('license3',lincese_gray)
# cv2.waitKey(0)

def lll(img):
    width,height,layer = img.shape
    for i in range(width):
        for ii in range(height):
            a = abs(int(img[i][ii][0]) - int(img[i][ii][1])) < 5
            b = abs(int(img[i][ii][0]) - int(img[i][ii][2])) < 5
            c = abs(int(img[i][ii][1]) - int(img[i][ii][2])) < 5


            aa = img[i][ii][0] < 160
            bb = img[i][ii][1] < 160
            cc = img[i][ii][2] < 160

            if a and b and c and aa and bb and cc:
                img[i][ii][0] = 25
                img[i][ii][1] = 25
                img[i][ii][2] = 25
    return img
def ll(img):
    width,height = img.shape
    for i in range(width):
        for ii in range(height):
            # a = abs(int(img[i][ii]) - int(img[i][ii])) < 5
            # b = abs(int(img[i][ii]) - int(img[i][ii])) < 5
            # c = abs(int(img[i][ii]) - int(img[i][ii])) < 5


            aa = img[i][ii] < 70

            if  aa :
                img[i][ii] = int(img[i][ii]/2)

    return img



# lincese = smooth(lincese)
# lincese = sharpen(lincese)

# lincese_gray_noface = face_wipeoff(lincese_gray,face_plus)

# cv2.imshow('noface',lincese_gray_noface)
# cv2.waitKey(0)

upper,lower = divide_image(lincese,face_plus)


# ret, binary = cv2.threshold(lincese_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

# lincese_gray_noface = face_wipeoff(lincese_gray,face_plus)

# index_ = info_divide(lincese_gray,face_plus)

cv2.imshow('upper',upper)
cv2.waitKey(0)
cv2.imshow('lower',lower)
cv2.waitKey(0)
cv2.imwrite('upper.png',upper)
cv2.imwrite('lower.png',lower)
# ss= pytesseract.image_to_data(upper,lang='chi_sim',output_type = 'dict')

# cv2.waitKey(0)


# text2 = pytesseract.image_to_string(lower,lang='chi_sim')
# print(text1)


#
# cv2.imshow('license',lincese_gray)
# cv2.waitKey(0)



print()