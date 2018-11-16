import glob
import cv2
import numpy as np

#dst_rgb_dir = glob.glob('/home/kuma/something_test/dataset_izunuma/images/20180808/kinect_color_depth_hd_03.bag/color/*.png')
dst_rgb_dir = glob.glob('/home/kuma/something_test/kawaguchi/*.jpg')

for file in dst_rgb_dir:
    print file + "resize"

    img = cv2.imread(file, -1)
    if len(img.shape) == 3:
        height, width, channels = img.shape[:3]
    else:
        height, width = img.shape[:2]
        channels = 1

    resize_img = cv2.resize(img, (width/2, height/2))
    #resize_img = cv2.resize(img, (960,544))
    cv2.imwrite(file, resize_img)
