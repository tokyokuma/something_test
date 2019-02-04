import os
import glob
import re
import cv2
import shutil
import numpy as np

def main():
    Create_train_txt()
    Create_val_txt()

def Create_train_txt():
    filelist_rgb = glob.glob(os.path.join(train_rgb_dir, '*.png'))

    if os.path.exists(train_txt):
        print 'train.txt already exists'

    else:
        with open(train_txt, mode='w') as f:
            for file_path_rgb in filelist_rgb:
                filename = os.path.basename(file_path_rgb)
                file_path_label = os.path.join(train_label_dir, filename)
                s = file_path_rgb + ',' + file_path_label + '\n'
                f.write(s)

def Create_val_txt():
    filelist_rgb = glob.glob(os.path.join(val_rgb_dir, '*.png'))

    if os.path.exists(val_txt):
        print 'val.txt already exists'

    else:
        with open(val_txt, mode='w') as f:
            for file_path_rgb in filelist_rgb:
                filename = os.path.basename(file_path)
                file_path_label = os.path.join(train_label_dir, filename)
                s = file_path_rgb + ',' + file_path_label + '\n'
                f.write(s)


if __name__ == '__main__':
    train_rgb_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_rgb'
    train_label_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_label'
    val_rgb_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/'
    val_label_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/'
    train_txt = '/home/kuma/something_test/dataset_izunuma/train.txt'
    val_txt = '/home/kuma/something_test/dataset_izunuma/val.txt'

    main()
