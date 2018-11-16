import glob
import cv2
import numpy as np

dst_rgb_dir = glob.glob('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_rgb/*.png')
dst_label_dir = glob.glob('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_label/*.png')
dst_rgb_val_dir = glob.glob('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_rgb_val/*.png')
dst_label_val_dir = glob.glob('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_label_val/*.png')

for file in dst_rgb_dir:
    print file + "resize"
    img = cv2.imread(file, -1)
    resize_img = cv2.resize(img, (960,544))
    cv2.imwrite(file, resize_img)

for file in dst_label_dir:
    print file + "resize"
    img = cv2.imread(file, -1)
    resize_img = cv2.resize(img, (960,544), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(file, resize_img)

for file in dst_rgb_val_dir:
    print file + "resize"
    img = cv2.imread(file, -1)
    resize_img = cv2.resize(img, (960,544))
    cv2.imwrite(file, resize_img)

for file in dst_label_val_dir:
    print file + "resize"
    img = cv2.imread(file, -1)
    resize_img = cv2.resize(img, (960,544), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(file, resize_img)
