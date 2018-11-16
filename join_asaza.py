import cv2
import glob
import os
import sys
import numpy as np

src_dir_up = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/asaza_gagabuta_half'
src_dir_bottom ='/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/label_asaza'
dst_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_all'

origin_filelist = glob.glob(os.path.join(src_dir_up, '*.png'))

for orig_file in origin_filelist:
    #get file name
    orig_filename = os.path.basename(orig_file)

    #get only watershed file name
    if len(orig_filename) == 30:
        up_filename = orig_filename
        bottom_filename = up_filename.rstrip('_watershed_mask.png') + '.png'
        up_file_path = os.path.join(src_dir_up, up_filename)
        bottom_file_path = os.path.join(src_dir_bottom, bottom_filename)
        full_file_path = os.path.join(dst_dir, up_filename)
        #read image
        up_img_24bit = cv2.imread(up_file_path, 0)
        up_img_int8 = up_img_24bit.astype(np.int8)
        cv2.imwrite(full_file_path, up_img_int8 )

        up_img = cv2.imread(full_file_path, 0)
        bottom_img = cv2.imread(bottom_file_path, 0)

        img_full = cv2.vconcat([up_img, bottom_img])
        cv2.imwrite(full_file_path, img_full )

        print up_filename
