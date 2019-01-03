import cv2
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import numpy as np
from PIL import Image
from PIL import ImageEnhance

def smooth (img,show=0):
    img_smooth = cv2.GaussianBlur(img,(3,3),0)
    if show == 1:
        cv2.imshow(img_smooth)
        cv2.waitKey(0)
    return img_smooth

def sharpen (img,show=0,kernel=0):
    if kernel == 0 :
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
    img_sharpen  = cv2.filter2D(img,-1,kernel = kernel)
    if show == 1:
        cv2.imshow(img_sharpen)
        cv2.waitKey(0)
    return img_sharpen

def defind_kernel(UDLR,center):
    kernel = np.array([[0, UDLR, 0], [UDLR, center, UDLR], [0, UDLR, 0]], np.float32)
    return kernel

def uniform_illumination(img):
    cv2.imwrite('image.jpg',img,[cv2.IMWRITE_JPEG_QUALITY, 50])
    b,g,r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image = cv2.merge([b,g,r])
    return image

def image_enhancement(img):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    # img.show()

    enh_con = ImageEnhance.Contrast(img)
    contrast = 1.5
    img_contrasted = enh_con.enhance(contrast)
    # img_contrasted.show()

    img = cv2.cvtColor(np.array(img_contrasted), cv2.COLOR_RGB2BGR)
    # cv2.imshow("OpenCV", img)
    # cv2.waitKey()
    return img



