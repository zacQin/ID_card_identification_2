import cv2
import pandas as pd
from face_base import find_face
from face_base import license_detection_Rough
from face_base import license_detection_Detailed
from face_base import divide_image
from face_base import face_wipeoff
from optimization_tool import image_enhancement
import numpy as np
from dfg import rotate_image
import os
import ocr
import shutil
from PIL import Image
from glob import glob
from tool import clean_symbol
import imutils

image_files = glob('bigtest_images/*.*')
# image_files = glob('error/*.*')
result_dir = 'bigtest_result1'
error_dir = 'error_img'
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.mkdir(result_dir)

if os.path.exists(error_dir):
    shutil.rmtree(error_dir)
os.mkdir(error_dir)

for image_file in sorted(image_files):
    print(image_file)

    # img = np.array(Image.open(image_file).convert('RGB'))
    img = cv2.cvtColor(np.asarray( Image.open(image_file)), cv2.COLOR_RGB2BGR)

    # cv2.imshow('dd',img)
    # cv2.waitKey(0)

    width,height,layer = img.shape

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face,face_plus,img,img_gray = find_face(img)

    if face_plus == 'No':
        error_file = os.path.join(error_dir,  os.path.splitext(os.path.split(image_file)[-1])[0]+'.png')
        print(error_file)
        cv2.imwrite(error_file, img)
        continue

    print('secenon')
    lincese ,lincese_gray = license_detection_Rough(img,img_gray,face_plus)

    # cv2.imshow('img',lincese)
    # cv2.waitKey(0)

    face,face_plus,img,img_gray = find_face(lincese)

    if face_plus == 'No':
        error_file = os.path.join(error_dir,  os.path.splitext(os.path.split(image_file)[-1])[0]+'.png')
        print(error_file)
        cv2.imwrite(error_file, img)
        continue

    # cv2.imshow('license_ori', lincese)
    # cv2.waitKey(0)

    print('1kan')
    upper,lower = divide_image(lincese,face_plus)
    # cv2.imshow('lower',lower)
    # cv2.waitKey(0)
    # cv2.imshow('upper',upper)
    # cv2.waitKey(0)
    print('2kan')
    result_lower, lower = ocr.model(lower)
    # cv2.imshow('lower',lower)
    # cv2.waitKey(0)
    print('3kan')

    lincese,lincese_gray,tag = rotate_image(lower,lincese,lincese_gray)

    if tag == 0:
        error_file = os.path.join(error_dir, os.path.splitext(os.path.split(image_file)[-1])[0] + '.png')
        cv2.imwrite(error_file, img)
        continue

    # cv2.imshow('upper', upper)
    # cv2.waitKey(0)
    # cv2.imshow('lower', lower)
    # cv2.waitKey(0)
    # cv2.imshow('license', lincese)
    # cv2.waitKey(0)

    # cv2.imshow('resize',lincese_gray)
    # cv2.waitKey(0)

    # cv2.imshow('resize', lincese)
    # cv2.waitKey(0)

    # cv2.imshow('imglincens',lincese)
    # cv2.waitKey(0)

    face,face_plus,img,img_gray = find_face(lincese)

    print('22')

    lincese ,lincese_gray = license_detection_Detailed(lincese,lincese_gray,face_plus)
    print(lincese.shape)

    lower_change = 0
    if lincese.shape[1]> 800:
        r = 650 / lincese.shape[1]
        dim = (650, int(lincese.shape[0] * r))
        lincese = cv2.resize(lincese, dim, interpolation=cv2.INTER_AREA)
        lincese_gray = cv2.resize(lincese_gray, dim, interpolation=cv2.INTER_AREA)
        lower_change =1
    # elif lincese.shape[0] < 580:
    #     r = int(lincese.shape[1]*1.1) / lincese.shape[1]
    #     dim = (int(lincese.shape[1]*1.1), int(lincese.shape[0] * r))
    #     lincese = cv2.resize(lincese, dim, interpolation=cv2.INTER_CUBIC)


    # r = 640 / lincese.shape[1]
    # dim = (640, int(lincese.shape[0] * r))
    # #
    # # 执行图片缩放，并显示
    # lincese = cv2.resize(lincese, dim, interpolation=cv2.INTER_AREA)

    print(lincese.shape)

    print('3')

    # lincese_gray_noface = face_wipeoff(lincese_gray,face_plus)

    print('4')

    face, face_plus, img, img_gray = find_face(lincese)

    # cv2.imshow('bb',lincese)
    # cv2.waitKey(0)

    if face_plus == 'No':
        error_file = os.path.join(error_dir,  os.path.splitext(os.path.split(image_file)[-1])[0]+'.png')
        print(error_file)
        cv2.imwrite(error_file, img)
        continue

    # cv2.imshow('bb',bb)

    # cv2.waitKey(0)
    print('1')

    upper,lower = divide_image(lincese,face_plus)

    # upper = smooth(upper)

    upper = image_enhancement(upper)
    # lower = image_enhancement(lower)
        # if lower_change ==  1:
        #     r = 600 / lower.shape[1]
        #     dim = (600, int(lower.shape[0] * r))
        #     #
        #     # 执行图片缩放，并显示
        #     lower = cv2.resize(lower, dim, interpolation=cv2.INTER_AREA)
    # upper = sharpen(upper)
    # upper = smooth(upper)

    # upper = image_enhancement(upper)

    # # lower = image_enhancement(lower)

    # cv2.imshow('upper',upper)
    # cv2.waitKey(0)
    # cv2.imshow('lower',lower)
    # cv2.waitKey(0)
    # cv2.imwrite('upper.png',upper)
    # cv2.imwrite('lower.png',lower)

    # output_dir = os.path.join(result_dir, os.path.splitext(os.path.split(image_file)[-1])[0])
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # os.mkdir(output_dir)

    result_upper, image_result_upper = ocr.model(upper)
    # output_file = os.path.join(output_dir, 'result_upper.png')
    # cv2.imwrite(output_file, image_result_upper)

    result_lower, image_result_lower = ocr.model(lower)
    # output_file = os.path.join(output_dir, 'result_lower.png')
    # cv2.imwrite(output_file, image_result_lower)

    if  ((result_upper.__len__() < 1) or (result_lower.__len__() < 1 )):
        error_file = os.path.join(error_dir, os.path.splitext(os.path.split(image_file)[-1])[0] + '.png')
        cv2.imwrite(error_file, img)
        continue

    print('1')

    # cv2.imshow('img',image_framed)
    # cv2.waitKey(0)
    # output_file = os.path.join(output_dir, 'result.png')
    # cv2.imwrite(output_file, image_framed)

    list_lower = []
    for key in result_lower:
        length=len(result_lower[key][1])
        idnumber=[]
        for i in range(length):
            if result_lower[key][1][i].isdigit() or (i == length-1 and result_lower[key][1][i] == 'X'):
                idnumber.append(result_lower[key][1][i])
        print(idnumber)
        if idnumber!=[] and len(idnumber)==18:
            list_lower.append(idnumber)

    coordinate = []
    info = []

    for key in result_upper:
        if np.array( [clean_symbol(result_upper[key][1])] ) != '':
            if ((coordinate == []) and (info == [])):
                coordinate = np.array([ result_upper[key][0] ])
                info = np.array( [clean_symbol(result_upper[key][1])] )

            else:
                coordinate_add = np.array([ result_upper[key][0] ])
                info_add = np.array( [clean_symbol(result_upper[key][1])] )
                coordinate  = np.vstack((coordinate, coordinate_add))
                info = np.vstack((info,info_add))

    list_upper = np.hstack((coordinate,info))

    paper_cup_big = []
    while len(list_upper) != 0:

        min_position = np.where( np.min (list_upper[:,5].astype(int)) == list_upper[:,5].astype(int) )
        min_position = min_position[0][0]

        ceiling = int(list_upper[min_position,1])
        floor = int(list_upper[min_position,5])
        layer_height = np.abs(ceiling - floor)
        needed_height = int ( layer_height / 5 )
        up_limit = (floor - needed_height)
        down_limit = (floor + needed_height)

        paper_cup_medium = []
        times=list_upper.shape[0]
        i = 0
        cursor = 0
        while i <= times-1:
            if ( int(list_upper[cursor,5]) > up_limit ) and ( int(list_upper[cursor,5]) < down_limit):
                if paper_cup_medium == []:
                     paper_cup_medium = list_upper[cursor,:]
                     list_upper = np.delete(list_upper, cursor, 0)
                     i+=1
                else:
                     paper_cup_medium = np.vstack(( paper_cup_medium,list_upper[cursor,:] ))
                     list_upper = np.delete(list_upper, cursor, 0)
                     i+=1
            else:
                i +=1
                cursor += 1

        if paper_cup_medium.shape.__len__() > 1:
            paper_cup_medium = paper_cup_medium[paper_cup_medium[:, 0].astype(int).argsort()]
            paper_cup_small = [x for x in paper_cup_medium[:, -1]]
        else:
            paper_cup_small=[]
            paper_cup_small .append(paper_cup_medium[-1])

        paper_cup_big.append(paper_cup_small)

    list_upper = paper_cup_big

    list = []



    list = list_lower+list_upper

    if list.__len__() == 0:
        error_file = os.path.join(error_dir, os.path.splitext(os.path.split(image_file)[-1])[0] + '.png')
        cv2.imwrite(error_file, img)
        continue

    # output_dir = os.path.join(result_dir, os.path.splitext(os.path.split(image_file)[-1])[0])
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # os.mkdir(output_dir)

    output_dir = os.path.join(result_dir, os.path.splitext(os.path.split(image_file)[-1])[0])
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    output_file = os.path.join(output_dir, 'result_upper.png')
    cv2.imwrite(output_file, image_result_upper)
    output_file = os.path.join(output_dir, 'result_lower.png')
    cv2.imwrite(output_file, image_result_lower)

    output_file = os.path.join(output_dir,'info.txt')

    file = open(output_file, 'w')
    for lines in list:
        # print(fp)
        fp = ''.join(lines)
        file.write(str(fp))
        file.write('\n')
    file.close()

    output_file = os.path.join(output_dir, 'image.png')
    cv2.imwrite(output_file, lincese)

print(' ')