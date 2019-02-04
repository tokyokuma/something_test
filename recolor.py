#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import os
import sys
import cv2
import shutil
import numpy as np
from argparse import ArgumentParser

pallete = [[  0,  0,  0],
           [170,234,150],
           [220,220,  0],
           [107,142, 35],
           [152,251,152],
           [ 70,130,180],
           [220, 20, 60],
           [  0, 60,100],
           [150,250,250],
           [  0,  0,  0],
           [  0,  0,  0]]

def reconstruct_label(label_img):
    default_labelmap = [3, 11, 16, 17, 18, 20, 21, 22, 23, 24, 28, 34, 35]

    reconstruct_labelmap = [255, 255, 255,  10, 255, 255, 255, 255, 255, 255,
                            255,  10, 255, 255, 255, 255,   0,   0,   1, 255,
                              2,   3,   4,   5,   6, 255, 255, 255,   7, 255,
                            255, 255, 255, 255,   8,   9]


    num_of_class = len(default_labelmap)

    for i in range (0, num_of_class):
        label_img[np.where(label_img == default_labelmap[i])] = reconstruct_labelmap[default_labelmap[i]]

    return label_img

def recolor(img):
    bamboo_pole = [90, 120, 150]
    h, w, c = img.shape

    for y in range(0, h):
        for x in range(0, w):
            orig_B = img.item(y, x, 0)
            orig_G = img.item(y, x, 1)
            orig_R = img.item(y, x, 2)
            orig_BGR = [orig_B, orig_G, orig_R]

            if orig_BGR == bamboo_pole:
                img[y, x] = pallete[0]
                print img[y, x]
            else:
                pass
    return img

def coloring(img):
    #coloring_img = np.array(palette, dtype=np.uint8)[out]
    coloring_img = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
    for idx in range(len(pallete)):
        [r, g, b] = pallete[idx]
        coloring_img[img == idx] = [b, g, r]

    return coloring_img

def mode1():
    img_src = '/home/kuma/something_test/dataset_izunuma/test/frame018303_color_mask.png'
    img = cv2.imread(img_src)

    water_sur = [128, 128, 128]

    h, w, c = img.shape

    for y in range(0, h):
        for x in range(0, w):
            orig_B = img.item(y, x, 0)
            orig_G = img.item(y, x, 1)
            orig_R = img.item(y, x, 2)
            orig_BGR = [orig_B, orig_G, orig_R]

            if orig_BGR == water_sur:
                img[y, x] = [0, 0, 0]
                print img[y, x]
            else:
                pass

def mode2():
    filepath = args.src_dir + '/frame015727_color_mask.png'
    recolor_filepath = args.src_dir + '/frame015727_recolor_mask.png'
    img = cv2.imread(filepath)
    recolor_img = recolor(img)
    cv2.imwrite(recolor_filepath, recolor_img)

def mode3():
    filepath = '/home/kuma/Documents/修士論文/画像/train_data/frame017245_watershed_mask.png'
    coloring_filepath = '/home/kuma/Documents/修士論文/画像/train_data/frame017245_coloring.png'
    img = cv2.imread(filepath, -1)
    print type(img)
    img = reconstruct_label(img)
    coloring_img = coloring(img)
    cv2.imwrite(coloring_filepath, coloring_img)
    #cv2.imwrite(coloring_filepath, img)

def main(args):
    if args.mode == 1:
        mode1()
    elif args.mode == 2:
        mode2()
    elif args.mode == 3:
        mode3()
    else:
        pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_dir', default="", help='Src directory')
    parser.add_argument('--dst_dir', default="", help='Dst directory')
    parser.add_argument('--ref_dir', default="", help='Reference directory')
    parser.add_argument('--img_extn', default="*png", help='RGB Image format')
    parser.add_argument('--mode', type=int, default=3, help='Switch resize mode')
    args = parser.parse_args()
    main(args)
