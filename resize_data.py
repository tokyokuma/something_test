import glob
import os
import sys
import cv2
import numpy as np
from argparse import ArgumentParser

def Reconstruct_label(label_img):
    default_labelmap = [3, 11, 16, 17, 18, 20, 21, 22, 23, 24, 28, 34, 35]

    reconstruct_labelmap = [255, 255, 255,  10, 255, 255, 255, 255, 255, 255,
                            255,  10, 255, 255, 255, 255,   0,   0,   1, 255,
                              2,   3,   4,   5,   6, 255, 255, 255,   7, 255,
                            255, 255, 255, 255,   8,   9]


    num_of_class = len(default_labelmap)

    for i in range (0, num_of_class):
        label_img[np.where(label_img == default_labelmap[i])] = reconstruct_labelmap[default_labelmap[i]]

    return label_img

def mode1():
    filelist = glob.glob(os.path.join(args.src_dir, args.img_extn))
    for i, file in enumerate(filelist):
        filename = os.path.basename(file)
        img_rgb = cv2.imread(file, -1)
        resize_img = cv2.resize(img_rgb, (args.width,args.height))
        cv2.imwrite(os.path.join(args.dst_dir, filename), resize_img)

def mode2():
    filelist = glob.glob(os.path.join(args.src_dir, args.img_extn))
    for i, file in enumerate(filelist):
        filename = os.path.basename(file)
        label_img_rgb = cv2.imread(file, 0)
        label_img_int8 = label_img_rgb.astype(np.int8)
        resize_img = cv2.resize(label_img_int8, (args.width,args.height),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.dst_dir, filename), resize_img)


def mode3():
    src_dir = args.src_dir + '/' + args.distance + '/origin'
    dst_dir = args.dst_dir + '/' + args.distance + '/resize'

    filelist = glob.glob(os.path.join(src_dir, args.img_extn))
    for i, file in enumerate(filelist):
        filename = os.path.basename(file)
        split_filename = os.path.splitext(filename)
        filename = split_filename[0].rstrip('_watershed_mask') + '.png'
        label_img_rgb = cv2.imread(file, 0)
        label_img_int8 = label_img_rgb.astype(np.int8)
        resize_img = cv2.resize(label_img_int8, (args.width,args.height),interpolation=cv2.INTER_NEAREST)

        if args.relabel:
            resize_img = Reconstruct_label(resize_img)
        else:
            pass

        cv2.imwrite(os.path.join(dst_dir, filename), resize_img)

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
    parser.add_argument('--img_extn', default="*png", help='RGB Image format')
    parser.add_argument('--width', type=int, default=768, help='Width of image')
    parser.add_argument('--height', type=int, default=432, help='Height of image')
    parser.add_argument('--relabel', default="True", help='Relabel')
    parser.add_argument('--mode', type=int, default=3, help='Switch resize mode')
    parser.add_argument('--distance', default="", help='Distance')

    args = parser.parse_args()
    main(args)
