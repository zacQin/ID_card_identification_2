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
from PIL import Image
import pytesseract
import numpy as np
from dfg import rotate_image
import os
import ocr
import shutil
import numpy as np
from PIL import Image
from glob import glob
import imutils


image_files = glob('test_images2/*.*')
result_dir = 'test_result2'
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.mkdir(result_dir)

for image_file in sorted(image_files):

    print(image_file)

    img = np.array(Image.open(image_file).convert('RGB'))

    width,height,layer = img.shape

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face,face_plus,img,img_gray = find_face(img)

    if face_plus == 'No':
        continue


    print('secenon')
    lincese ,lincese_gray = license_detection_Rough(img,img_gray,face_plus)

    # cv2.imshow('license_ori', lincese)
    # cv2.waitKey(0)


    face,face_plus,img,img_gray = find_face(lincese)
    upper,lower = divide_image(lincese,face_plus)
    result_lower, lower = ocr.model(lower)

    lincese,lincese_gray,tag = rotate_image(lower,lincese,lincese_gray)

    if tag == 0:
        continue

    # cv2.imshow('upper', upper)
    # cv2.waitKey(0)
    # cv2.imshow('lower', lower)
    # cv2.waitKey(0)
    # cv2.imshow('license', lincese)
    # cv2.waitKey(0)


    # lincese_gray = cv2.resize(lincese_gray, (400,247), interpolation=cv2.INTER_CUBIC)
    # lincese = cv2.resize(lincese, (400,247), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow('resize',lincese_gray)
    # cv2.waitKey(0)

    # cv2.imshow('resize', lincese)
    # cv2.waitKey(0)

    face,face_plus,img,img_gray = find_face(lincese)

    print('22')

    lincese ,lincese_gray = license_detection_Detailed(lincese,lincese_gray,face_plus)

    print('3')

    lincese_gray_noface = face_wipeoff(lincese_gray,face_plus)

    print('4')

    face, face_plus, img, img_gray = find_face(lincese)

    # cv2.imshow('bb',bb)
    # cv2.waitKey(0)
    print('1')

    upper,lower = divide_image(lincese,face_plus)


    # cv2.imshow('upper',upper)
    # cv2.waitKey(0)
    # cv2.imshow('lower',lower)
    # cv2.waitKey(0)
    # cv2.imwrite('upper.png',upper)
    # cv2.imwrite('lower.png',lower)

    output_dir = os.path.join(result_dir, os.path.splitext(os.path.split(image_file)[-1])[0])
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    result_upper, image_result_upper = ocr.model(upper)
    output_file = os.path.join(output_dir, 'result_upper.png')
    cv2.imwrite(output_file, image_result_upper)

    result_lower, image_result_lower = ocr.model(lower)
    output_file = os.path.join(output_dir, 'result_lower.png')
    cv2.imwrite(output_file, image_result_lower)

    print('1')

    result, image_framed = ocr.model(lincese)
    # cv2.imshow('img',image_framed)
    # cv2.waitKey(0)
    # output_file = os.path.join(output_dir, 'result.png')
    # cv2.imwrite(output_file, image_framed)

    list = []
    for key in result_lower:
        length=len(result_lower[key][1])
        idnumber=[]
        for i in range(length):
            # print((i == length-1) and (result_lower[key][1][i] == 'X'),i == length-1,result_lower[key][1][i] == 'X')
            if result_lower[key][1][i].isdigit() or (i == length-1 and result_lower[key][1][i] == 'X'):
                idnumber.append(result_lower[key][1][i])
        print(idnumber)
        if idnumber!=[] and len(idnumber)==18:
            list.append(idnumber)

        # list.append(result_lower[key][1])

    for key in result_upper:
        list.append(result_upper[key][1])

    # output_dir = os.path.join(result_dir, os.path.splitext(os.path.split(image_file)[-1])[0])
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # os.mkdir(output_dir)

    output_file = os.path.join(output_dir,'info.txt')

    file = open(output_file, 'w')
    for fp in list:
        file.write(str(fp))
        file.write('\n')
    file.close()

    output_file = os.path.join(output_dir, 'image.png')
    cv2.imwrite(output_file, lincese)



print(' ')