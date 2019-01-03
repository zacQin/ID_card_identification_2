import cv2
import numpy as np
import imutils
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def find_face(img):
    #输入彩色图像
    tag = 1
    img_c=img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_gray
    size = img.shape
    # cv2.imshow('imgfind',img)
    # cv2.waitKey(0)
    face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    eye_detector = cv2.CascadeClassifier('models/haarcascade_eye.xml')
    i = 0
    judge = 'No'
    while judge == 'No':
        # cv2.imshow('shangmian',img)
        # cv2.waitKey(0)
        faces = face_detector.detectMultiScale(img, 1.1, 6)
        while faces.__len__() == 0:

            if i > 4:
                print('eee')
                face_plus = "No"
                return faces, face_plus, img_c, img
                break

            i+=1
            print(i,'1')
            img = imutils.rotate_bound(img, 270)
            img_c = imutils.rotate_bound(img_c, 270)
            size = img.shape
            # cv2.imshow('zhongjian', img)
            # cv2.waitKey(0)
            faces = face_detector.detectMultiScale(img, 1.1, 6)

        for (x, y, w, h) in faces:
            candidate = img[y:y+h,x:x+w]
            # cv2.imshow('dd',candidate)
            # cv2.waitKey(0)
            eyes = eye_detector.detectMultiScale(candidate, 1.2, 3)

            if eyes.__len__() !=0 :
                if np.mean(eyes[:,1]) < h/2:
                    upper = y - int(h/2)
                    lower = y + int(h/2*3)
                    left = x - int(w/10*2)
                    right = x + int(w/5*6)

                    if left < 0:
                        left = 0

                    if right > size[1]:
                        right = size[1]

                    if upper < 0:
                        upper = 0

                    if lower > size[0]:
                        lower = size[0]

                    face_plus=[upper,lower,left,right]

                    if np.abs(upper-lower)/size[0] > .8 or np.abs(left-right)/size[1] >.5:
                        print('比例比例不对')
                        face_plus = "No"
                        return faces, face_plus, img_c, img
                        break

                    judge = 'Ok'
                    #返回面部信息
                    return faces,face_plus,img_c,img

        img = imutils.rotate_bound(img, 270)
        img_c = imutils.rotate_bound(img_c, 270)
        size = img.shape
        i += 1
        print(i,'2')



        if i > 4:
            print('eee')
            face_plus = "No"
            return faces, face_plus, img_c, img
            break

    # print()



def wipeoff_onface(img_gray,facepoint,reservation=0):
    #擦出面部，输入灰度图（彩图也可以，但是希望是灰度的）
    img = img_gray
    img_face_wipeoff = img
    # cv2.imshow('df',img)
    # cv2.waitKey(0)
    faceimg = img_face_wipeoff[facepoint[0]:facepoint[1], facepoint[2]:facepoint[3]]

    # cv2.imshow('df',faceimg)
    # cv2.waitKey(0)
    img_face_wipeoff[facepoint[0]:facepoint[1], facepoint[2]:facepoint[3]] = 255

    if reservation == 0:
        return img_face_wipeoff
    else:
        return img,img_face_wipeoff

def license_area_onface_Rough(img,img_gray,face_plus):
    #检测出证件区域，输入彩图
    upper = face_plus[0]
    lower = face_plus[1]
    left = face_plus[2]
    right = face_plus[3]

    w = abs(left - right)
    h = abs(upper - lower)
    size = img.shape

    # area_left = left - int(1.8*w)
    # area_right = right
    # area_upper = upper - int(h/10)
    # area_lower = lower + int(h/3)

    area_left = left - int(2.5 * w)
    area_right = right + int(.5 * w)
    area_upper = upper - int(h / 5)
    area_lower = lower + int(h / 2)

    if area_left < 0:
        area_left = 0

    if area_right > size[1]:
        area_right = size[1]

    if area_upper < 0:
        area_upper = 0

    if area_lower > size[0]:
        area_lower = size[0]

    license_area = img[area_upper:area_lower,area_left:area_right]
    license_area_gray = img_gray[area_upper:area_lower,area_left:area_right]
    # cv2.imshow('area', img)
    # cv2.waitKey(0)
    # cv2.imshow('area', license_area)
    # cv2.waitKey(0)
    return license_area,license_area_gray

def license_area_onface_Detailed(img,img_gray,face_plus):
    #检测出证件区域，输入彩图
    upper = face_plus[0]
    lower = face_plus[1]
    left = face_plus[2]
    right = face_plus[3]

    w = abs(left - right)
    h = abs(upper - lower)
    size = img.shape

    area_left = left - int(2.3*w)
    area_right = right
    area_upper = upper - int(h/10)
    area_lower = lower + int(h/3)

    if area_left < 0:
        area_left = 0

    if area_right > size[1]:
        area_right = size[1]

    if area_upper < 0:
        area_upper = 0

    if area_lower > size[0]:
        area_lower = size[0]

    license_area = img[area_upper:area_lower,area_left:area_right]
    license_area_gray = img_gray[area_upper:area_lower,area_left:area_right]
    # cv2.imshow('area', img)
    # cv2.waitKey(0)
    # cv2.imshow('area', license_area)
    # cv2.waitKey(0)
    return license_area,license_area_gray

def license_detection_Rough(img,img_gray,face_plus):
    if face_plus :
        license_area,license_area_gray = license_area_onface_Rough(img,img_gray,face_plus)
        return license_area,license_area_gray
    else:
        license_area,license_area_gray =np.zeros([50,50])
        print('NO FACE')
        return license_area,license_area_gray

def license_detection_Detailed(img,img_gray,face_plus):
    if face_plus :
        license_area,license_area_gray = license_area_onface_Detailed(img,img_gray,face_plus)
        return license_area,license_area_gray
    else:
        license_area,license_area_gray =np.zeros([50,50])
        print('NO FACE')
        return license_area,license_area_gray


def face_wipeoff(img_gray,face_plus):
    img_gray = wipeoff_onface(img_gray,face_plus)
    return img_gray

def divide_image(img,face_plus):
    upper = img[0:face_plus[1], 0:face_plus[2]]
    lower = img[face_plus[1]:,:]
    return upper,lower
