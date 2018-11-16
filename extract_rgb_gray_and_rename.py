import os
import glob
import re
import cv2
import shutil
import numpy as np
import random
from scipy.ndimage.interpolation import rotate

def Refresh_dir(dst_rgb_dir, dst_label_dir, dst_rgb_val_dir, dst_label_val_dir):
    shutil.rmtree(dst_rgb_dir)
    shutil.rmtree(dst_label_dir)
    #shutil.rmtree(dst_rgb_val_dir)
    #shutil.rmtree(dst_label_val_dir)
    os.mkdir(dst_rgb_dir)
    os.mkdir(dst_label_dir)
    #os.mkdir(dst_rgb_val_dir)
    #os.mkdir(dst_label_val_dir)
    print 'refresh directlies'

def All_to_Extract():
    src_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_all'
    dst_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_extract'
    shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)

    origin_filelist = glob.glob(os.path.join(src_dir, '*.png'))

    for orig_file in origin_filelist:
        #get file name
        orig_filename = os.path.basename(orig_file)

        #get only rgb file name
        if len(orig_filename) == 15:
            #split filename extension as tuple
            split_filename = os.path.splitext(orig_filename)

            #create watershed_mask filename
            watershed_mask_filename = split_filename[0] + '_watershed_mask.png'

            if os.path.exists(os.path.join(src_dir, watershed_mask_filename)):
                #copy to dst_dir from src_dir
                shutil.copy(os.path.join(src_dir, orig_filename), os.path.join(dst_dir, orig_filename))
                shutil.copy(os.path.join(src_dir, watershed_mask_filename), os.path.join(dst_dir, watershed_mask_filename))
                print orig_filename
                print watershed_mask_filename

            else:
                pass

def Write_Copy(dst_rgb_dir, label_file_path, dst_label_dir, img, num):
    cv2.imwrite(os.path.join(dst_rgb_dir, '%06.f.png'%num), img)
    shutil.copy(label_file_path, os.path.join(dst_label_dir, '%06.f.png'%num))

def Write_Rot_Reverse(dst_rgb_dir, dst_label_dir, rgb_img, label_img, num):
    cv2.imwrite(os.path.join(dst_rgb_dir, '%06.f.png'%num), rgb_img)
    cv2.imwrite(os.path.join(dst_label_dir, '%06.f.png'%num),label_img)

def Contrast(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, min, max, high_low, a):
    #re_contrast table
    min_table = min
    max_table = max
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )

    # create high-contrast LUT
    for lut in range(0, min_table):
        LUT_HC[lut] = 0
    for lut in range(min_table, max_table):
        LUT_HC[lut] = 255 * (lut - min_table) / diff_table
    for lut in range(max_table, 255):
        LUT_HC[lut] = 255
    # create low-contrast LUT
    for lut in range(256):
        LUT_LC[lut] = min_table + lut * (diff_table) / 255

    #high:1 low:2
    if high_low == 1:
        high_cont_img = cv2.LUT(orig_img, LUT_HC)
        Write_Copy(dst_rgb_dir, label_file_path, dst_label_dir, high_cont_img, a)
    elif high_low == 2:
        low_cont_img = cv2.LUT(orig_img, LUT_LC)
        Write_Copy(dst_rgb_dir, label_file_path, dst_label_dir, low_cont_img, a)

def Average(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, average_param, b):
    average_square = (average_param,average_param)
    average_img = cv2.blur(orig_img, average_square)
    Write_Copy(dst_rgb_dir, label_file_path, dst_label_dir, average_img, b)

def Sharp(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, sharp_param, c):
    #sharpened
    k = 1.0
    op = np.array([[-k, k, -k],
                   [-k, 1 + sharp_param * k, -k],
                   [-k, -k, -k]])

    img_tmp = cv2.filter2D(orig_img, -1 , op)
    sharp_img = cv2.convertScaleAbs(img_tmp)
    Write_Copy(dst_rgb_dir, label_file_path, dst_label_dir, sharp_img, c)

def Salt_Pepper(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, salt_pepper, noise_param, d):
    row,col,ch = orig_img.shape
    s_vs_p = 0.5
    amount = noise_param
    sp_img = orig_img.copy()
    #salt:1 pepper:2
    if salt_pepper == 1:
        num_salt = np.ceil(amount * orig_img.size * s_vs_p)
        coords = [np.random.randint(0, i-1 , int(num_salt)) for i in orig_img.shape]
        sp_img[coords[:-1]] = (255,255,255)
        Write_Copy(dst_rgb_dir, label_file_path, dst_label_dir, sp_img, d)

    elif salt_pepper == 2:
        num_pepper = np.ceil(amount* orig_img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in orig_img.shape]
        sp_img[coords[:-1]] = (0,0,0)
        Write_Copy(dst_rgb_dir, label_file_path, dst_label_dir, sp_img, d)

def Rotation(dst_rgb_dir, dst_label_dir, rgb_img, label_img, angle, e):
    h, w, c = rgb_img.shape
    rgb_rot_img = rotate(rgb_img, angle)
    rgb_rot_img = cv2.resize(rgb_rot_img, (w, h))
    label_rot_img = rotate(label_img, angle)
    label_rot_img = cv2.resize(label_rot_img, (w, h))
    Write_Rot_Reverse(dst_rgb_dir, dst_label_dir, rgb_rot_img, label_rot_img, e)

def Flip(dst_rgb_dir, dst_label_dir, rgb_img, label_img, f):
    rgb_reverse_img = cv2.flip(rgb_img, 1)
    label_reverse_img = cv2.flip(label_img, 1)
    Write_Rot_Reverse(dst_rgb_dir, dst_label_dir, rgb_reverse_img, label_reverse_img, f)

def Resize(orig_file_path, label_file_path):
    #for file in dst_rgb_dir:
    orig_filename = os.path.basename(orig_file_path)
    #print orig_filename + " : resize"
    rgb_img = cv2.imread(orig_file_path, 1)
    resize_rgb_img = cv2.resize(rgb_img, (640,360))
    cv2.imwrite(orig_file_path, resize_rgb_img)

    #for file in dst_label_dir:
    #print file + "resize"
    label_img = cv2.imread(label_file_path, 0)
    resize_label_img = cv2.resize(label_img, (640,360), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(label_file_path, resize_label_img)

def Create_Validation(dst_rgb_dir, dst_label_dir, dst_rgb_val_dir, dst_label_val_dir):
    #create validtion data
    #counting the number of files
    filelist = glob.glob('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_rgb/*.png')
    num_of_files_random = [(m) for m in filelist]
    numbers = range(len(num_of_files_random))

    #pick up one-tenth of the whole image for test image
    test_img = len(num_of_files_random) / 10
    rand = random.sample(numbers, test_img)

    #picked up data move to destination directly
    for i in xrange(len(rand)):
        num = rand[i]
        print num
        shutil.move(os.path.join(dst_rgb_dir,'%06.f.png'%num),dst_rgb_val_dir)
        shutil.move(os.path.join(dst_label_dir,'%06.f.png'%num),dst_label_val_dir)

def main():
    All_to_Extract()
    '''
    src_dir = '/media/nouki/hdd/izunuma/izunuma_dataset1_extract'
    dst_rgb_dir = '/media/nouki/hdd/izunuma/izunuma_dataset1_rgb'
    dst_label_dir = '/media/nouki/hdd/izunuma/izunuma_dataset1_label'
    dst_rgb_val_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_rgb_val'
    dst_label_val_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_label_val'

    Refresh_dir(dst_rgb_dir, dst_label_dir, dst_rgb_val_dir, dst_label_val_dir)

    origin_filelist = glob.glob(os.path.join(src_dir, '*.png'))
    all_num_of_files = [(n) for n in origin_filelist]
    num_of_files = len(all_num_of_files) / 2
    print 'number of files : {}' .format(num_of_files)
    #start file number
    i = 1

    for orig_file in origin_filelist:
        #get file name
        orig_filename = os.path.basename(orig_file)

        #get only rgb file name
        if len(orig_filename) == 15:
            #split filename extension as tuple
            split_filename = os.path.splitext(orig_filename)
            #create watershed_mask filename
            watershed_mask_filename = split_filename[0] + '_watershed_mask.png'
            #copy to dst_dir from src_dir
            shutil.copy(os.path.join(src_dir, orig_filename), os.path.join(dst_rgb_dir, orig_filename))
            shutil.copy(os.path.join(src_dir, watershed_mask_filename), os.path.join(dst_label_dir, watershed_mask_filename))
            #rename at dst_dir
            rename_orig_filename = '%06.f.png'%i
            rename_watershed_mask_filename = '%06.f.png'%i
            #after rename file path orig and label
            orig_file_path = os.path.join(dst_rgb_dir, rename_orig_filename)
            label_file_path = os.path.join(dst_label_dir, rename_watershed_mask_filename)

            os.rename(os.path.join(dst_rgb_dir, orig_filename),orig_file_path)
            os.rename(os.path.join(dst_label_dir, watershed_mask_filename),label_file_path)

            label_img_rgb = cv2.imread(label_file_path, 0)
            #convert rgb to int8 label image and imread
            label_img_int8 = label_img_rgb.astype(np.int8)
            cv2.imwrite(label_file_path, label_img_int8)

            #rgb and label image file_path
            Resize(orig_file_path, label_file_path)
            orig_img = cv2.imread(orig_file_path, 1)
            label_img = cv2.imread(label_file_path, 0)


            file_num = []
            for multiply in range(1,20):
                file_num.append(i + num_of_files * multiply)

            Contrast(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 50, 200, 1, file_num[0])
            Contrast(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 50, 200, 2, file_num[1])
            Contrast(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 70, 180, 1, file_num[2])
            Contrast(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 70, 180, 2, file_num[3])
            Average(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 3, file_num[4])
            Average(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 5, file_num[5])
            Sharp(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 6, file_num[6])
            Sharp(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 7, file_num[7])
            Sharp(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 8, file_num[8])
            Salt_Pepper(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 1, 0.005, file_num[9])
            Salt_Pepper(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 1, 0.005, file_num[10])
            Salt_Pepper(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 2, 0.01, file_num[11])
            Salt_Pepper(dst_rgb_dir, label_file_path, dst_label_dir, orig_img, 2, 0.01, file_num[12])


            print str(i) + '/' + str(num_of_files) + ' : ' + orig_filename

            i = i + 1
        else:
            pass


    #Rotation
    print 'Start Rotation'
    j = 1
    rgb_filelist = glob.glob(os.path.join(dst_rgb_dir, '*.png'))
    num_of_files = [(n) for n in rgb_filelist]
    num_of_files = len(num_of_files)
    print 'number of files : {}' .format(num_of_files)

    for rgb_file in rgb_filelist:
        rgb_filename = os.path.basename(rgb_file)
        label_file = os.path.join(dst_label_dir, rgb_filename)
        rgb_img = cv2.imread(rgb_file, 1)
        label_img = cv2.imread(label_file, 0)

        file_num = []
        for multiply in range(1,20):
            file_num.append(j + num_of_files * multiply)

        for k in range(1, 8):
            Rotation(dst_rgb_dir, dst_label_dir, rgb_img, label_img, 45*k, file_num[k-1])

        print str(j) + '/' + str(num_of_files) + ' : ' + rgb_filename

        j = j + 1

    #Flip
    print 'Start Flip'
    l = 1
    rgb_filelist = glob.glob(os.path.join(dst_rgb_dir, '*.png'))
    num_of_files = [(n) for n in rgb_filelist]
    num_of_files = len(num_of_files)
    print 'number of files : {}' .format(num_of_files)

    for rgb_file in rgb_filelist:
        rgb_filename = os.path.basename(rgb_file)
        label_file = os.path.join(dst_label_dir, rgb_filename)
        rgb_img = cv2.imread(rgb_file, 1)
        label_img = cv2.imread(label_file, 0)

        file_num = []
        for multiply in range(1,3):
            file_num.append(l + num_of_files * multiply)

        Flip(dst_rgb_dir, dst_label_dir, rgb_img, label_img, file_num[0])

        print str(l) + '/' + str(num_of_files) + ' : ' + rgb_filename

        l = l + 1

    #Resize(dst_rgb_dir, dst_label_dir)
    #Create_Validation(dst_rgb_dir, dst_label_dir, dst_rgb_val_dir, dst_label_val_dir)
    '''
if __name__ == '__main__':
    main()
