import os
import sys
import glob
import cv2
import csv
import copy
import numpy as np

orig_img_dir = '/home/kuma/something_test/dataset_izunuma/accuracy_test/per_accuracy/orig_img'
seg_img_dir = '/home/kuma/something_test/dataset_izunuma/accuracy_test/per_accuracy/seg_img'

#color map bamboo_pole iron_pole wood lotus sky person boat water_sur
color_map = [[90, 120, 150],
             [153, 153, 153],
             [153, 234, 170],
             [35, 142, 107],
             [152,251,152],
             [180, 130, 70],
             [60, 20, 220],
             [100, 60, 0],
             [128, 128, 128]]

#turu false judge accuracy true bamboo_pole iron_pole wood lotus sky person boat water_sur

def create_accuracy_map(color_map, accuracy_map, cv_orig_img, cv_seg_img, h, w):

    for y in range(0, h):
        for x in range(0, w):
            orig_B = cv_orig_img.item(y, x, 0)
            orig_G = cv_orig_img.item(y, x, 1)
            orig_R = cv_orig_img.item(y, x, 2)
            orig_BGR = [orig_B, orig_G, orig_R]

            seg_B = cv_seg_img.item(y, x, 0)
            seg_G = cv_seg_img.item(y, x, 1)
            seg_R = cv_seg_img.item(y, x, 2)
            seg_BGR = [seg_B, seg_G, seg_R]

            for i in range(0, 9):
                if orig_BGR == color_map[i]:
                    correct_map[i] = correct_map[i] + 1
                else:
                    pass

            if orig_BGR == seg_BGR:
                for i in range(0, 9):
                    if orig_BGR == color_map[i]:
                        accuracy_map[i][0] = accuracy_map[i][0] + 1
                        accuracy_map[i][i+1] = accuracy_map[i][i+1] + 1
                    else:
                        pass

            elif orig_BGR != seg_BGR:
                for j in range(0, 9):
                    if orig_BGR == color_map[j]:
                        for k in range(0, 9):
                            if seg_BGR == color_map[k]:
                                accuracy_map[j][k+1] = accuracy_map[j][k+1] + 1
                            else:
                                pass
                    else:
                        pass

    #print correct_map
    #print accuracy_map
    correct_map_new = copy.deepcopy(correct_map)
    correct_map_new.insert(0,0)

    f = open('accuracy_raw.csv', 'a')
    writer = csv.writer(f)
    writer.writerow(correct_map_new)
    for n in range(0, 9):
        writer.writerow(accuracy_map[n])
    f.close()


def calc_accuracy(accuracy_map, correct_map):
    for l in range(0, 9):
        for m in range(0, 10):
            if correct_map[l] != 0:
                accuracy_map[l][m] = float(accuracy_map[l][m]) / float(correct_map[l])* 100
            else :
                pass

    #f = open('accuracy.csv', 'a')
    #writer = csv.writer(f)
    #for n in range(0, 9):
    #    writer.writerow(accuracy_map[n])
    #f.close()
    #print accuracy_map

def main():
    orig_imglist = glob.glob(os.path.join(orig_img_dir, '*png'))

    for orig_img in orig_imglist:

        global accuracy_map
        accuracy_map = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        global correct_map
        correct_map = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        seg_img = os.path.join(seg_img_dir, os.path.basename(orig_img))
        print os.path.basename(orig_img)
        #read ad cv imamge
        cv_orig_img = cv2.imread(orig_img, -1)
        cv_seg_img = cv2.imread(seg_img, -1)

        h, w, c = cv_orig_img.shape

        create_accuracy_map(color_map, accuracy_map, cv_orig_img, cv_seg_img, h, w)
        calc_accuracy(accuracy_map, correct_map)

if __name__ == "__main__":
    header = ['accyracy','bamboo_pole', 'iron_pole', 'wood', 'lotus', 'hishi', 'sky', 'person', 'boat', 'water_sur']
    f = open('accuracy_raw.csv', 'a')
    writer = csv.writer(f)
    writer.writerow(header)
    f.close()

    main()
