import glob
import cv2
import numpy as np
import lmdb
import caffe
import random
import csv
import os
import random

rgb_dir_file = glob.glob('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_rgb/*.png')
rgb_val_dir_file = glob.glob('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_rgb_val/*.png')
label_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_label'
label_val_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_label_val'

rgb_lmdb_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_lmdb/rgb_train')
label_lmdb_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_lmdb/label_train'
rgb_val_lmdb_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_lmdb/img_val')
label_val_lmdb_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_lmdb/label_val'

def read_img(path):
    img = cv2.imread(path, -1)
    if len(img.shape) != 3:
        print("image error 1")
        print(len(img.shape))

    if len(img.shape) != 3 or img.dtype != np.uint8:
        print("image error 2")

    trans_img = img.transpose((2, 0, 1))
    return trans_img

def read_label(path):
    label = cv2.imread(path, -1)
    if len(label.shape) != 2:
        print("label error 1")

    palette = [255, 255, 255 ,255, 255, 255, 255, 255, 255, 255,
               255, 255, 255, 255, 255, 255, 0, 1, 2, 255,
               255, 3, 255, 4, 5, 255, 255, 255, 6, 255,
               255, 255, 255, 255, 255, 7
               ]

    re_label = np.array(palette, dtype=np.uint8)[label]
    if len(re_label.shape) != 2 or output.dtype != np.uint8:
        print("label error 2")

    return re_label

def distribute_data():
    map_size =1e13
    rgb_train_lmdb = lmdb.open(rgb_lmdb_dir, map_size=map_size)
    label_train_lmdb = lmdb.open(label_lmdb_dir, map_size=map_size)
    rgb_val_lmdb = lmdb.open(rgb_val_lmdb_dir, map_size=map_size)
    label_val_lmdb = lmdb.open(label_val_lmdb_dir, map_size=map_size)

    for rgb_file in rgb_dir_file:
        rgb = read_img(rgb_file)
        filename = os.path.basename(rgb_file)
        label = read_label(os.path.join(label_dir,filename))

        with rgb_train_lmdb.begin(write=True) as txn:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 3
            datum.height = height
            datum.width = width
            datum.data = rgb.tobytes()
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

        with label_train_lmdb.begin(write=True) as txn:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = height
            datum.width = width
            datum.data = label.tobytes()
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

        print filename * 'finish'

    for rgb_val_file in rgb_val_dir_file:
        rgb_val = read_img(rgb_val_file)
        filename_val = os.path.basename(rgb_val_file)
        label_val = read_label(os.path.join(label_val_dir,filename_val))

        with rgb_val_lmdb.begin(write=True) as txn:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 3
            datum.height = height
            datum.width = width
            datum.data = rgb_val.tobytes()
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

        with label_val_lmdb.begin(write=True) as txn:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = height
            datum.width = width
            datum.data = label_val.tobytes()
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

        print filename_val * 'finish'

if __name__ == "__main__":
    distribute_data()
