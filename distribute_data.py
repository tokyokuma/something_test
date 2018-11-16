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

    label = resize_data(label, size, is_label=True)

    palette = [255, 255, 255 ,255, 255, 255, 255, 
               0, 1, 255, 255, 2, 3, 4, 255, 255, 255, 5, 255, 6,
               7, 8, 9, 10, 11, 12, 13, 14, 15, 255,
               255, 16, 17, 18, 255]

    output = np.array(palette, dtype=np.uint8)[label]
    if len(output.shape) != 2 or output.dtype != np.uint8:
        print("label error 2")
        
    return output
    
def read_img(path):
    img = cv2.imread(path, -1)
    if len(img.shape) != 3:
        print("image error 1")
        print(len(img.shape))

    output = resize_data(img, size, is_label=False)
    if len(output.shape) != 3 or output.dtype != np.uint8:
        print("image error 2")
    
    output = output.transpose((2, 0, 1))
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
    img_lmdb = lmdb.open('img_fine_train', map_size=map_size)
    depth_lmdb = lmdb.open('depth_fine_train', map_size=map_size)
    label_lmdb = lmdb.open('label_fine_train', map_size=map_size)
    
    all_path = []
    directory_list = glob.glob(path_to_fine_train)
    #directory_list += glob.glob(path_to_coarse_train_extra)
    
    for path_dir in directory_list:
        path_in_dir = glob.glob(path_dir + "/*_labelIds.png")
        all_path +=path_in_dir
    
    
        
    f = open('fine_train.txt', 'w') 
    
    for j in range(12):
      random.shuffle(all_path)
      for path in all_path:
          gamma = float(1. +random.randint(-5, 5)/10.)
          img, depth, label = read_data(path)
          img = gamma_transport(img, gamma)   
          with img_lmdb.begin(write=True) as txn:
              datum = caffe.proto.caffe_pb2.Datum()
              datum.channels = 3
              datum.height = height
              datum.width = width
              datum.data = img.tobytes() 
              str_id = '{:08}'.format(i)
              txn.put(str_id.encode('ascii'), datum.SerializeToString())
        
          with depth_lmdb.begin(write=True) as txn:
              datum = caffe.proto.caffe_pb2.Datum()
              datum.channels = 1
              datum.height = height
              datum.width = width
              datum.data = depth.tobytes() 
              str_id = '{:08}'.format(i)
              txn.put(str_id.encode('ascii'), datum.SerializeToString())

          with label_lmdb.begin(write=True) as txn:
              datum = caffe.proto.caffe_pb2.Datum()
              datum.channels = 1
              datum.height = height
              datum.width = width
              datum.data = label.tobytes() 
              str_id = '{:08}'.format(i)
              txn.put(str_id.encode('ascii'), datum.SerializeToString())
          f.writelines([str(path) + " ,gamma : " + str(gamma), "\n"])
          i += 1
        
          print('{:05}'.format(i) + " images of " + str(12*len(all_path)) + " is finished") 

    f.close()   
    print(i)

def create_inpulse_noise_data(data_num):
    i = 0
    map_size =1e13
    img_noise_1_lmdb = lmdb.open('img_inpulse_noise_1', map_size=map_size)
    img_noise_2_lmdb = lmdb.open('img_inpulse_noise_2', map_size=map_size)
    for i in range(data_num):
      img1, img2 = create_inpulse_noise_img()
      with img_noise_1_lmdb.begin(write=True) as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 3
        datum.height = height
        datum.width = width
        datum.data = img1.tobytes() 
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

      with img_noise_2_lmdb.begin(write=True) as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 3
        datum.height = height
        datum.width = width
        datum.data = img2.tobytes() 
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
      
      print('{:05}'.format(i) + " noise images of " + str(data_num) + " is created") 

def create_gaussian_noise_data(data_num):
    i = 0
    map_size =1e13
    img_noise_1_lmdb = lmdb.open('img_gaussian_noise', map_size=map_size)
    for i in range(data_num):
      img = create_gaussian_noise_img()
      with img_noise_1_lmdb.begin(write=True) as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 3
        datum.height = height
        datum.width = width
        datum.data = img.tobytes() 
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

      print('{:05}'.format(i) + " noise images of " + str(data_num) + " is created") 


def create_inpulse_noise_img():
    noise_mag = random.randint(10, 100)
    flag_1 = random.randint(0, 1)

    img1 = np.full((3, height, width), 1, dtype=np.uint8)
    img2 = np.full((3, height, width), 0, dtype=np.uint8)
    
    if flag_1 == 0:
      return img1, img2
    
    for i in range(height):
      for j in range(width):
        flag_2 = random.randint(0, noise_mag)
        flag_3 = random.randint(0, 1)
        if flag_2 == 0:
          img1[0][i][j] = img1[1][i][j] = img1[2][i][j] = 0
          if flag_3 == 0:
            img2[0][i][j] = img2[1][i][j] = img2[2][i][j] = 255

    return img1, img2

def create_inpulse_noise_depth():
    noise_mag = random.randint(10, 100)
    flag_1 = random.randint(0, 1)

    img1 = np.full((3, height, width), 1, dtype=np.uint8)
    
    if flag_1 == 0:
      return img1
    
    for i in range(height):
      for j in range(width):
        flag_2 = random.randint(0, noise_mag)
        if flag_2 == 0:
          img1[0][i][j] = img1[1][i][j] = img1[2][i][j] = 0

    return img1

def create_gaussian_noise_img():
    noise_mag = random.randint(20, 40)
    flag_1 = random.randint(0, 1)
    noise = np.random.normal(0, noise_mag, height*width*3)

    img = np.full((3, height, width), 0, dtype=np.int8)
    
    if flag_1 == 0:
      return img
    
    for i in range(height):
      for j in range(width):
        for k in range(3):
          img[k][i][j] = int(noise[i + height*j + width*height*k])

    return img

def create_gaussian_noise_depth():
    noise_mag = float(random.randint(15, 25)/100.)
    flag_1 = random.randint(0, 1)
    
    noise = np.random.normal(0, noise_mag, heigh*width*1)

    img1 = np.full((1, height, width), 1, dtype=np.float32)
    
    if flag_1 == 0:
      return img1
    
    for i in range(height):
      for j in range(width):
        for k in range(1):
          img1[k][i][j] = noise[i + height*j + width*height*k]

    return img1

if __name__ == "__main__":
    #distribute_data() 
    #img = read_img("berlin_000000_000019_leftImg8bit.png")
    #img = gamma_transport(img, 0.5)
    #cv2.imwrite("berlin_000000_000019_leftImg8bit_gamma.png", img.transpose(1,2,0))
    
    #create_inpulse_noise_data(37811)
    create_gaussian_noise_data(42187)
    #img1= create_gaussian_noise_img()

    #cv2.imwrite("noise_test.png", (img1+img).transpose(1,2,0))

