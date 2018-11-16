import os
import glob
import re
import cv2
import shutil
import numpy as np
import random

src_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_all'
dst_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_extract'

origin_filelist = glob.glob(os.path.join(src_dir, '*.png'))

for orig_file in origin_filelist:
    #get file name
    orig_filename = os.path.basename(orig_file)

    #get only rgb file name
    if len(orig_filename) == 15:
        #split filename extension as tuple
        split_filename = os.path.splitext(orig_filename)

        #create watershed_mask filename
        watershed_mask_filename = split_filename[0] + '_watershed_mask.png'

        if os.path.exists(os.path.join(src_dir, watershed_mask_filename)):
            #copy to dst_dir from src_dir
            shutil.copy(os.path.join(src_dir, orig_filename), os.path.join(dst_dir, orig_filename))
            shutil.copy(os.path.join(src_dir, watershed_mask_filename), os.path.join(dst_dir, watershed_mask_filename))
            print orig_filename
            print watershed_mask_filename

        else:
            pass
