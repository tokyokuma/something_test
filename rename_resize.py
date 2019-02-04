import glob
import os
import cv2
import numpy as np

def main():
    num = 0
    rgb_file_list = glob.glob(os.path.join(src_rgb_dir, '*.png'))

    if not os.path.isdir(dst_rgb_dir):
        os.mkdir(dst_rgb_dir)
        os.mkdir(dst_label_dir)

    for rgb_file in rgb_file_list:

        rgb_filename = os.path.basename(rgb_file)
        print 'resize : ' + rgb_filename
        #resize rgb
        rgb_img = cv2.imread(rgb_file, 1)
        rgb_img = cv2.resize(rgb_img, (width, height))
        cv2.imwrite(os.path.join(dst_rgb_dir, '%06.f.png'%num), rgb_img)

        #resize label
        split_filename = os.path.splitext(rgb_filename)
        label_filename = split_filename[0] + '_watershed_mask.png'
        label_file = os.path.join(src_label_dir, label_filename)
        print label_filename
        label_img = cv2.imread(label_file, 0)
        label_img = Reconstruct_label(label_img)
        label_img = cv2.resize(label_img, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(dst_label_dir, '%06.f.png'%num), label_img)

        num += 1

def Reconstruct_label(label_img):
    default_labelmap = [3, 11, 16, 17, 18, 20, 21, 22, 23, 24, 28, 34, 35]

    reconstruct_labelmap = [255, 255, 255,  10, 255, 255, 255, 255, 255, 255,
                            255,  10, 255, 255, 255, 255,   0,   0,   1, 255,
                              2,   3,   4,   5,   6, 255, 255, 255,   7, 255,
                            255, 255, 255, 255,   8,   9]

    num_of_class = len(default_labelmap)
    label_img[label_img == 255] = 10
    for i in range (0, num_of_class):
        label_img[np.where(label_img == default_labelmap[i])] = reconstruct_labelmap[default_labelmap[i]]

    return label_img

if __name__ == '__main__':
    width = 512
    height = 288
    src_rgb_dir = '/home/kuma/something_test/dataset_izunuma/accuracy_test/accuracy_test_ESPNetv2/original/rgb'
    src_label_dir = '/home/kuma/something_test/dataset_izunuma/accuracy_test/accuracy_test_ESPNetv2/original/label'
    dst_rgb_dir = '/home/kuma/something_test/dataset_izunuma/accuracy_test/accuracy_test_ESPNetv2/' + str(width) + ':' + str(height) + '/rgb'
    dst_label_dir = '/home/kuma/something_test/dataset_izunuma/accuracy_test/accuracy_test_ESPNetv2/' + str(width) + ':' + str(height) + '/label'

    main()
