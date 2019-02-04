import glob
import os
import sys
import cv2
import shutil
import numpy as np
from argparse import ArgumentParser

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
    distance_list = [1000, 1500, 2000, 2500, 3000 , 3500, 4000, 4500]
    scale_list = ['0.5', '1.0', '1.5']

    for n in range(len(distance_list)):
        ref_dir = args.ref_dir + '/' + str(distance_list[n]) + '/resize'
        base_dst_dir = args.dst_dir + '/' + str(distance_list[n]) + '/seg'
        filelist = glob.glob(os.path.join(ref_dir, args.img_extn))

        for j in range (len(scale_list)):
            dst_dir = os.path.join(base_dst_dir, scale_list[j])
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
                os.mkdir(dst_dir)
            else:
                os.mkdir(dst_dir)

        for i, file in enumerate(filelist):
            filename = os.path.basename(file)
            for l in range (len(scale_list)):
                base_src_dir = args.src_dir + '/' + scale_list[l]
                src_file = os.path.join(base_src_dir, filename)
                dst_dir = os.path.join(base_dst_dir, scale_list[l])

                shutil.copy(src_file, dst_dir)
                print 'copy : ' + filename

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
