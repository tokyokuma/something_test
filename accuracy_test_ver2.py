#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import os
import sys
import glob
import cv2
import csv
import copy
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    label_img[np.where(label_img == 255)] = 10
    return label_img

def Create_accuracy_map(ref, seg, height, width):
    accuracy_map = np.zeros([args.classes-1, args.classes-1], dtype=np.float32)
    correct_map = np.zeros([1, args.classes-1], dtype=np.float32)

    if args.mode == 3 or args.mode == 1 :
        for y in range(0, height):
            for x in range(0, width):
                ref_value = ref.item(y, x)
                seg_value = seg.item(y, x)

                if ref_value == 10 or seg_value == 10:
                    pass

                else:
                    correct_map[0, ref_value] += 1
                    if ref_value == seg_value:
                        accuracy_map[ref_value, ref_value] += 1

                    else:
                        accuracy_map[ref_value, seg_value] += 1

    else:
        for y in range(0, args.height):
            for x in range(0, args.width):
                ref_value = ref.item(y, x)
                seg_value = seg.item(y, x)

                if ref_value == 10 or seg_value == 10:
                    pass

                else:
                    correct_map[0, ref_value] += 1
                    if ref_value == seg_value:
                        accuracy_map[ref_value, ref_value] += 1

                    else:
                        accuracy_map[ref_value, seg_value] += 1

    return accuracy_map, correct_map

def Heatmap(accuracy_map, heatmapname):
    with open(args.class_list) as f:
        class_list = [s.strip() for s in f.readlines()]

    df_cm = pd.DataFrame(accuracy_map, index = [name for i, name in enumerate(class_list)],columns = [name for i, name in enumerate(class_list)])
    plt.figure(figsize = (11,9))
    sn.heatmap(df_cm, annot=True, fmt='1.1f', cmap='Reds', square=True, linewidths=.5, linecolor="black")

    plt.savefig(heatmapname)

def mode1():
    src_base_dir = "/home/kuma/something_test/dataset_izunuma/accuracy_test/images/color/ESPNetV2"
    all_accuracy_csv = src_base_dir + '/all_accuracy_raw_' + str(args.scale) + '.csv'
    img_size = ['256✕144','512✕288','768✕432','1024✕576']
    width = [256, 512, 768, 1024]
    height = [144, 288, 432, 576]

    label_dir = args.base_dir + '/all/label'
    label_list = glob.glob(os.path.join(label_dir, args.img_extn))
    num_of_all_file = len(label_list)



    f = open(all_accuracy_csv, 'w')
    writer = csv.writer(f)
    writer.writerow(['all accuracy'])
    header = ['pole', 'wood', 'asaza', 'lotus', 'hishi', 'sky', 'person', 'boat', 'uki', 'water_sur']
    writer.writerow(header)
    f.close()

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)

    for k in range(len(img_size)):
        sum_accuracy_map = np.zeros([args.classes-1, args.classes-1], dtype=np.float32)
        sum_correct_map = np.zeros([1, args.classes-1], dtype=np.float32)

        f = open(all_accuracy_csv, 'a')
        writer = csv.writer(f)
        writer.writerow([width[k], height[k]])
        f.close()


        for i, label_file in enumerate(label_list):
            label = cv2.imread(label_file, 0)
            label = label.astype(np.int8)
            label_filename = os.path.basename(label_file)
            seg_filename = label_filename.strip('_watershed_mask.png') + '.png'

            seg_dir = args.base_dir + '/all/seg_all/' + img_size[k] + '/' + str(args.scale)
            seg_file = os.path.join(seg_dir, seg_filename)

            print seg_file
            seg = cv2.imread(seg_file, 0)

            label = Reconstruct_label(label)
            #seg = Reconstruct_label(seg)

            label = cv2.resize(label, (seg.shape[1], seg.shape[0]), interpolation=cv2.INTER_NEAREST)


            accuracy_map, correct_map = Create_accuracy_map(label, seg, seg.shape[0], seg.shape[1])
            #print accuracy_map
            sum_accuracy_map = sum_accuracy_map + accuracy_map
            sum_correct_map = sum_correct_map + correct_map


        f = open(all_accuracy_csv, 'a')
        writer = csv.writer(f)
        for t in range(len(sum_accuracy_map)):
            writer.writerow(sum_accuracy_map[t])
        f.close()


        for i in range(0, args.classes-1):
            for j in range(0, args.classes-1):
                if sum_accuracy_map[i, j] != 0:
                    sum_accuracy_map[i, j] = sum_accuracy_map[i, j] / sum_correct_map[0, i] * 100


        f = open(all_accuracy_csv, 'a')
        writer = csv.writer(f)
        for p in range(len(sum_accuracy_map)):
            writer.writerow(sum_accuracy_map[p])
        writer.writerow(sum_correct_map[0])
        f.close()


        heatmapname = img_size[k] + '_' + str(args.scale) + '.png'
        print sum_accuracy_map
        Heatmap(sum_accuracy_map, heatmapname)

def mode2():
    accuracy_csv = args.save_dir + '/accuracy_raw.csv'
    header = ['pole','wood', 'npeltatum', 'lotus', 'chestnuta', 'sky', 'person', 'boat', 'marking', 'water_sur']
    distance_list = [1000, 1500, 2000, 2500, 3000 , 3500, 4000, 4500]
    scale_list = ['0.5', '1.0', '1.5']

    for n in range(len(distance_list)):
        base_dir = args.base_dir + '/' + str(distance_list[n])
        ref_dir = base_dir + '/resize'
        seg_dir = base_dir + '/seg/1.5'

        ref_list = glob.glob(os.path.join(ref_dir, args.img_extn))
        num_of_all_file = len(ref_list)
        sum_accuracy_map = np.zeros([args.classes-1, args.classes-1], dtype=np.float32)
        sum_correct_map = np.zeros([1, args.classes-1], dtype=np.float32)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=2)

        for i, ref_file in enumerate(ref_list):
            ref = cv2.imread(ref_file, 0)
            ref_filename = os.path.basename(ref_file)
            seg_file = os.path.join(seg_dir, ref_filename)
            print seg_file
            seg = cv2.imread(seg_file, 0)
            accuracy_map, correct_map = Create_accuracy_map(ref, seg)
            #print accuracy_map
            sum_accuracy_map = sum_accuracy_map + accuracy_map
            sum_correct_map = sum_correct_map + correct_map


        sum_accuracy_map = sum_accuracy_map.tolist()
        sum_correct_map = sum_correct_map.tolist()

        for i in range(0, args.classes-1):
            for j in range(0, args.classes-1):
                if sum_accuracy_map[i, j] != 0:
                    sum_accuracy_map[i, j] = sum_accuracy_map[i, j] / sum_correct_map[0, i] * 100

        print sum_accuracy_map
        sum_accuracy_map = sum_accuracy_map.tolist()
        f = open(accuracy_csv, 'a')
        writer = csv.writer(f)
        writer.writerow(header)
        for n in range(0, args.classes-1):
            writer.writerow(sum_accuracy_map[n])

        f.close()
        #Heatmap(sum_accuracy_map)

def mode3():
    src_base_dir = "/home/kuma/something_test/dataset_izunuma/accuracy_test/images/color/ESPNetV2"
    #distance accuracy
    distance_accuracy_csv = src_base_dir + '/distance/distance_accuracy_raw_' + str(args.scale) + '.csv'
    lux_accuracy_csv = src_base_dir + '/lux/lux_accuracy_raw_' + str(args.scale) + '.csv'
    if args.option == 0:
        f = open(distance_accuracy_csv, 'w')
        writer = csv.writer(f)
        writer.writerow(['distance accuracy'])
        f.close()

    elif args.option == 1:
        f = open(lux_accuracy_csv, 'w')
        writer = csv.writer(f)
        writer.writerow(['lux accuracy'])
        f.close()

    distance_list = [1000, 1500, 2000, 2500, 3000 , 3500, 4000, 4500]
    scale_list = ['0.5', '1.0', '1.5']
    dir_list = ['distance', 'lux', 'asaza']
    img_size = ['256✕144','512✕288','768✕432','1024✕576']
    width = [256, 512, 768, 1024]
    height = [144, 288, 432, 576]

    #distance img lis
    distance_img_list = [['014436', '015914', '016145'],
                         ['014434', '015918', '016141'],
                         ['014431', '015923', '016138'],
                        writer.writerow(['all accuracy'])
     ['014429', '015928', '016135'],
                         ['014427', '015937', '016132'],
                         ['014425', '015950', '016129'],
                         ['014423', '014422', '016126'],
                         ['014409', '015860', '016119']]

    lux_img_list = [['005875', '005876', '001431'],
                    ['005832', '005833', '001348'],
                    ['005777', '005778', '001272'],
                    ['005692', '005693', '001207'],
                    ['005655', '005656', '001147'],
                    ['005602', '005603', '001098'],
                    ['005535', '005536', '001036'],
                    ['005138', '005137', '000877']]

    ref_dir = src_base_dir + '/' + dir_list[args.option] + '/original'

    for j in range(len(img_size)):
        seg_dir = src_base_dir + '/' + dir_list[args.option] + '/seg_all/' + img_size[j] + '/' + str(args.scale)

        if args.option == 0:
            f = open(distance_accuracy_csv, 'a')
        elif args.option == 1:
            f = open(lux_accuracy_csv, 'a')
        writer = csv.writer(f)
        writer.writerow([width[j], height[j]])
        f.close()

        print img_size[j]
        for k in range(len(distance_list)):
            distance_accuracy = [str(distance_list[k]), 0, 0, 0]

            for l in range(0,3):
                sum_accuracy_map = np.zeros([args.classes-1, args.classes-1], dtype=np.float32)
                sum_correct_map = np.zeros([1, args.classes-1], dtype=np.float32)
                np.set_printoptions(suppress=True)
                np.set_printoptions(precision=2)

                if args.option == 0:
                    ref_filename = 'frame' + distance_img_list[k][l] + '_watershed_mask.png'
                    seg_filename = 'frame' + distance_img_list[k][l] + '.png'

                if args.option == 1:
                    ref_filename = 'frame' + lux_img_list[k][l] + '_watershed_mask.png'
                    seg_filename = 'frame' + lux_img_list[k][l] + '.png'

                ref_file = ref_dir + '/' + ref_filename
                seg_file = seg_dir + '/' + seg_filename

                #print seg_file

                ref = cv2.imread(ref_file, 0)
                seg = cv2.imread(seg_file, 0)

                ref = Reconstruct_label(ref)
                seg = Reconstruct_label(seg)

                ref = cv2.resize(ref, (seg.shape[1], seg.shape[0]), interpolation=cv2.INTER_NEAREST)
                accuracy_map, correct_map = Create_accuracy_map(ref, seg, seg.shape[0], seg.shape[1])
                #print accuracy_map

                sum_accuracy_map = sum_accuracy_map + accuracy_map
                sum_correct_map = sum_correct_map + correct_map

                for i in range(0, args.classes-1):
                    for j in range(0, args.classes-1):
                        if sum_accuracy_map[i, j] != 0:
                            sum_accuracy_map[i, j] = sum_accuracy_map[i, j] / sum_correct_map[0, i] * 100

                #print sum_accuracy_map
                sum_accuracy_map = sum_accuracy_map.tolist()
                distance_accuracy[l+1] = sum_accuracy_map[0][0]

            print distance_accuracy

            if args.option == 0:
                f = open(distance_accuracy_csv, 'a')
            elif args.option == 1:
                f = open(lux_accuracy_csv, 'a')

            writer = csv.writer(f)
            writer.writerow(distance_accuracy)
            f.close()

            #Heatmap(sum_accuracy_map)

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
    parser.add_argument('--base_dir', default="/home/kuma/something_test/dataset_izunuma/accuracy_test/images/color/ESPNetV2", help='Label directory')
    parser.add_argument('--img_extn', default="*png", help='RGB Image format')
    parser.add_argument('--scale', default=1.5, help='Training scale')
    parser.add_argument('--width', type=int, default=768, help='Width of image')
    parser.add_argument('--height', type=int, default=432, help='Height of image')
    parser.add_argument('--save_dir', default='../', help='directory to save the results')
    parser.add_argument('--class_list', default="./izunuma_classes_en.txt", help='Classes memo file')
    parser.add_argument('--classes', default=11, type=int, help='Number of classes in the dataset. 20 for Cityscapes')
    parser.add_argument('--relabel', default="True", help='Relabel')
    parser.add_argument('--mode', type=int, default=3, help='Switch resize mode')
    parser.add_argument('--option', type=int, default=0, help='0:distance 1:lux')


    args = parser.parse_args()
    main(args)
