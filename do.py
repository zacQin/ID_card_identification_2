#-*- coding:utf-8 -*-
import os
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob

# image_files = glob('./test_images/*.*')
#
#
# if __name__ == '__main__':
#     result_dir = './test_result'
#     if os.path.exists(result_dir):
#         shutil.rmtree(result_dir)
#     os.mkdir(result_dir)
#
#     for image_file in sorted(image_files):
#         print(image_file)
#         image = np.array(Image.open(image_file).convert('RGB'))
#         result, image_framed = ocr.model(image)
#
#         output_file = os.path.join(result_dir, image_file.split('/')[-1])
#         a,b = os.path.split(output_file)



    #
    #     Image.fromarray(image_framed).save(output_file)
    #     print("\nRecognition Result:\n")
    #     list = []
    #     for key in result:
    #         list.append(result[key][1])
    #         # print(result[key][1])
    #         # # print(key)
    #
    # file = open(output_file, 'w')
    # for fp in list:
    #     file.write(str(fp))
    #     file.write('\n')
    #
    # file.close()
    # print()

image_files = glob('test_images/*.*')
for image_file in sorted(image_files):
        print(image_file)
        image = np.array(Image.open(image_file).convert('RGB'))
        # result, image_framed = ocr.model(image)

        result_dir = './test_result'
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        # print(output_file)
        # if os.path.exists(result_dir):
        #     shutil.rmtree(result_dir)
        # os.mkdir(result_dir)
        # output_file = os.path.join(result_dir, image_file.split('/')[-1])
        output_file = os.path.join(result_dir, os.path.splitext(os.path.split(image_file)[-1])[0])
        if os.path.exists(result_dir):
                shutil.rmtree(result_dir)
                os.mkdir(result_dir)
        print(output_file)
        #         a,b = os.path.split(output_file)