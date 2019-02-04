import glob
import cv2
import numpy as np

#src_dir = glob.glob('/home/kuma/something_test/dataset_izunuma/images/20180808/kinect_color_depth_sd_01.bag/qhd_color/*.png')
src_dir = glob.glob('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/asaza_gagabuta/*.png')

for file in src_dir:
    print file + "trimming"
    img = cv2.imread(file, -1)
    if len(img.shape) == 3:
        height, width, channels = img.shape[:3]
    else:
        height, width = img.shape[:2]
        channels = 1

    trimming_img = img[0:height/2 ,0:width]
    cv2.imwrite(file, trimming_img)
