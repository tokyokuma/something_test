import glob
import cv2
import numpy as np
import scipy.io
import lmdb
import caffe
import random
import csv
import os
import random
 

original_height = 1024
original_width = 2048
height = 360
width = 640
size = (width, height)

path_to_coarse_train = "./label/gtCoarse/train/*"
path_to_coarse_train_extra = "./label/gtCoarse/train_extra/*"
path_to_coarse_val = "./label/gtCoarse/val/*"
path_to_fine_train = "./label/gtFine/train/*"
path_to_fine_val = "./label/gtFine/val/*"
path_to_fine_test = "./label/gtFine/test/*"

def resize_data(img, size, is_label=True):
    if is_label == True:
        img_resized = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)
    else:
        img_resized = cv2.resize(img, size, interpolation = cv2.INTER_LANCZOS4)

    return img_resized


def read_label(path):
    label = cv2.imread(path, -1)
    if len(label.shape) != 2:
        print("label error 1")

    if len(label.shape) != 2 or label.dtype != np.uint8:
        print("label error 2")
        
    return label
    
def read_img(path):
    img = cv2.imread(path, -1)
    if len(img.shape) != 3:
        print("image error 1")
        print(len(img.shape))

    if len(img.shape) != 3 or img.dtype != np.uint8:
        print("image error 2")
    
    output = img.transpose((2, 0, 1))
    return output
    
def read_depth(path):   
    depth_mat = scipy.io.loadmat(path)
    depth = depth_mat["depth_map"]
    
    if len(depth.shape ) != 2:
        print("depth error 1")
        print(len(depth.shape))

    output = resize_data(depth, size, is_label=False)
    if len(output.shape) != 2 or output.dtype != np.float64:
        print("depth error 2")
        print(len(depth.shape))
    output =  (output*1000).astype(np.uint16)
#    for i in range(640):
#        for j in range(360):
#            if output[j][i] >= 12 * 1000:
#              output[j][i] = 0
    
    return output
    

def read_data(label_path):
    s = label_path.split("/")
    n = s[5].split("_")[0] + "_" + s[5].split("_")[1] + "_" + s[5].split("_")[2]
    
    img_path = "./image/" + s[3] + "/" + s[4] + "/" + n + "_leftImg8bit.png"
    stereo_path = "./depth/" + s[3] + "/" + s[4] + "/" + n + "_depth_stereoscopic.mat"
    
    img = read_img(img_path)
    if os.path.exists(stereo_path) == True:
      depth = read_depth(stereo_path)
    else: 
      depth = np.full(size, 0, dtype=np.uint8)
      print(stereo_path)
    
    label = read_label(label_path) 
   
    return img, depth, label

def gamma_transport(img, gamma):
    if gamma == 1.:
      return img
    table = np.zeros(256, np.uint8)
    for i in range(256):
      table[i] = int(255. * (i/255.)**gamma)

    return np.array(table, dtype=np.uint8)[img]
    
    


def distribute_data():
    i = 0
    map_size =1e13
    img_train_lmdb = lmdb.open('img_train', map_size=map_size)
    label_train_lmdb = lmdb.open('label_train', map_size=map_size)
    img_val_lmdb = lmdb.open('img_val', map_size=map_size)
    label_val_lmdb = lmdb.open('label_val', map_size=map_size)

    for i in range(20):
      img_path = "/home/nouki/dataset/cityscapes/izunuma_dataset1_rgb/%06.f.png"%(i+1)
      label_path = "/home/nouki/dataset/cityscapes/izunuma_dataset1_label/%06.f.png"%(i+1)
      label = read_label(label_path)
      img = read_img(img_path)
      
      with img_val_lmdb.begin(write=True) as txn:
              datum = caffe.proto.caffe_pb2.Datum()
              datum.channels = 3
              datum.height = height
              datum.width = width
              datum.data = img.tobytes() 
              str_id = '{:08}'.format(i)
              txn.put(str_id.encode('ascii'), datum.SerializeToString())

      with label_val_lmdb.begin(write=True) as txn:
              datum = caffe.proto.caffe_pb2.Datum()
              datum.channels = 1
              datum.height = height
              datum.width = width
              datum.data = label.tobytes() 
              str_id = '{:08}'.format(i)
              txn.put(str_id.encode('ascii'), datum.SerializeToString())
        
      print('{:05}'.format(i) + " images of " + str(216) + " is finished") 

    for i in range(216-20):
      img_path = "/home/nouki/dataset/cityscapes/izunuma_dataset1_rgb/%06.f.png"%(i+21)
      label_path = "/home/nouki/dataset/cityscapes/izunuma_dataset1_label/%06.f.png"%(i+21)
      label = read_label(label_path)
      img = read_img(img_path)
      
      with img_train_lmdb.begin(write=True) as txn:
              datum = caffe.proto.caffe_pb2.Datum()
              datum.channels = 3
              datum.height = height
              datum.width = width
              datum.data = img.tobytes() 
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
        
      print('{:05}'.format(i) + " images of " + str(216) + " is finished") 


if __name__ == "__main__":
    distribute_data() 
    #img = read_img("berlin_000000_000019_leftImg8bit.png")
    #img = gamma_transport(img, 0.5)
    #cv2.imwrite("berlin_000000_000019_leftImg8bit_gamma.png", img.transpose(1,2,0))
    
    #create_inpulse_noise_data(37811)
    #create_gaussian_noise_data(42187)
    #img1= create_gaussian_noise_img()

    #cv2.imwrite("noise_test.png", (img1+img).transpose(1,2,0))

