import cv2
import glob
import os
import numpy as np

def main():
    filelist = glob.glob(os.path.join(train_label_dir, '*.png'))
    num_of_files = len(filelist)
    count = 1
    for file in filelist:
        label_img = cv2.imread(file, 0)
        reconstruct_label_img = Reconstruct_label(label_img)
        cv2.imwrite(file, reconstruct_label_img)

        filename = os.path.basename(file)
        print str(count) +  '/' + str(num_of_files) + ' : ' + filename + ' was converted'

        count = count  + 1
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


if __name__ == '__main__':
    train_label_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_label'
    main()
