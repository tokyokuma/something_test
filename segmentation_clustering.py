#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
from argparse import ArgumentParser

def main(args):
    filename = 'frame005654.png'
    filepath = args.src_dir + '/' + filename
    thershold_path = args.src_dir + '/thershold.png'
    noise_path = args.src_dir + '/noise.png'
    obstacle_path = args.src_dir + '/obstacle.png'
    #img = cv2.imread(file_path, 0)
    #img = img.astype(np.int8)
    #cv2.imwrite(file_path, img)
    img = cv2.imread(filepath, -1)


    #0.0083sec
    pole_threshold = Pole_Threshold(img)
    cv2.imwrite(thershold_path, pole_threshold)
    #0.0011sec
    pole_remove_noise = Open_Close(pole_threshold)
    cv2.imwrite(noise_path, pole_remove_noise)

    start = time.time()
    #0.0040sec
    color_src = cv2.cvtColor(pole_remove_noise, cv2.COLOR_GRAY2BGR)
    label = cv2.connectedComponentsWithStats(pole_remove_noise)
    elapsed_time = time.time() - start
    print elapsed_time

    n = label[0] - 1

    #remove backgrouond label
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    # オブジェクト情報を利用してラベリング結果を画面に表示
    for i in range(n):
        # 各オブジェクトの外接矩形を赤枠で表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(color_src, (x0, y0), (x1, y1), (0, 0, 255))

        # 各オブジェクトのラベル番号と面積に黄文字で表示
        cv2.putText(color_src, "ID: " +str(i + 1), (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        cv2.putText(color_src, "S: " +str(data[i][4]), (x1 - 20, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        cv2.putText(color_src, "X: " + str(int(center[i][0])), (x1 + 20, y1 - 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        cv2.putText(color_src, "Y: " + str(int(center[i][1])), (x1 + 20, y1 - 115), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
    '''
    obstacle_coordination = np.zeros([n,2], dtype=np.float32)
    for i in range(n):
        obstacle_coordination[i-1][0] = int(center[i][1])
        obstacle_coordination[i-1][1] = int(center[i][0])

    print obstacle_coordination
    '''

    #cv2.imshow("pole_threshold", pole_threshold)
    #cv2.imshow("pole_remove_noise", pole_remove_noise)
    cv2.imwrite(obstacle_path, color_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Pole_Threshold(img):
    recolor_img = img.copy()
    pole = [0]
    black = [0]
    white = [255]
    recolor_img[np.where(recolor_img == pole)] = white
    recolor_img[np.where(recolor_img != white)] = black
    return recolor_img

def Open_Close(img):
    img_tmp = img
    element8 = np.ones((10, 1), np.uint8)
    iteration = 1
    while iteration <= 1:
        img_tmp = cv2.morphologyEx(img_tmp, cv2.MORPH_OPEN, element8)
        img_tmp = cv2.morphologyEx(img_tmp, cv2.MORPH_CLOSE, element8)
        iteration = iteration + 1
    return img_tmp

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_dir', default="", help='Src directory')
    parser.add_argument('--dst_dir', default="", help='Dst directory')
    parser.add_argument('--ref_dir', default="", help='Reference directory')
    parser.add_argument('--img_extn', default="*png", help='RGB Image format')
    parser.add_argument('--mode', type=int, default=3, help='Switch resize mode')
    args = parser.parse_args()
    main(args)
