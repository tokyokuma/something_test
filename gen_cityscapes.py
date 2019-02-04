from __future__ import division
from __future__ import print_function
import os
import time
import glob
import cv2
import torch
import numpy as np
import SegmentationModel as net
import fast_ESPNetv2 as fast
from argparse import ArgumentParser
from torch import nn
from collections import OrderedDict

#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

pallete = [[153,153,153],
           [170,234,150],
           [220,220,  0],
           [107,142, 35],
           [152,251,152],
           [ 70,130,180],
           [220, 20, 60],
           [  0, 60,100],
           [150,250,250],
           [  0,  0,  0],
           [  0,  0,  0]]

def relabel(img):
    img[img == 10] = 10
    img[img == 9] = 9
    img[img == 8] = 8
    img[img == 7] = 7
    img[img == 6] = 6
    img[img == 5] = 5
    img[img == 4] = 4
    img[img == 3] = 3
    img[img == 2] = 2
    img[img == 1] = 1
    img[img == 0] = 0
    return img


def evaluateModel(args, model, image_list):
    # gloabl mean and std values
    '''
    mean = [72.3923111, 82.90893555, 73.15840149]
    std = [45.3192215, 46.15289307, 44.91483307]

    mean = [131.84157, 145.38597, 135.16437]
    std = [76.013596, 67.85283,  70.89791 ]
    '''

    mean = np.array([131.84157, 145.38597, 135.16437])
    std =  np.array([76.013596, 67.85283,  70.89791 ])

    mesure_item = 8
    sum_times = [0] * mesure_item
    num_of_files = len(image_list)

    model.eval()
    for i, imgName in enumerate(image_list):

        start_read = time.time()
        img = cv2.imread(imgName)
        if args.overlay:
            img_orig = np.copy(img)
        elapsed_read = time.time() - start_read
        #print ('read : ', elapsed_read)

        #0.015sec
        img = img.astype(np.float32)
        start_mean_std = time.time()
        fast.mean_std(img, mean, std)
        elapsed_mean_std = time.time() - start_mean_std
        #print ('mean_std : ', elapsed_mean_std)


        #0.0012
        # resize the image to 1024x512x3
        start_resize = time.time()
        img = cv2.resize(img, (args.inWidth, args.inHeight))
        if args.overlay:
            img_orig = cv2.resize(img_orig, (args.inWidth, args.inHeight))
        elapsed_resize = time.time() - start_resize
        #print ('resize : ', elapsed_resize)

        #0.01
        start_numpy_to_tensor = time.time()
        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        elapsed_numpy_to_tensor = time.time() - start_numpy_to_tensor
        #print ('numpy_to_tensor : ', elapsed_numpy_to_tensor)


        #0.0045
        start_set_gpu = time.time()
        if args.gpu:
            img_tensor = img_tensor.cuda()
        elapsed_set_gpu = time.time() - start_set_gpu
        #print ('set_gpu : ', elapsed_set_gpu)


        #0.088
        start_inference = time.time()
        img_out = model(img_tensor)
        elapsed_inference = time.time() - start_inference
        print ('inference : ', elapsed_inference)

        start_tensor_to_numpy = time.time()
        classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()
        elapsed_tensor_to_numpy = time.time() - start_tensor_to_numpy
        #print ('tensor_to_numpy : ', elapsed_tensor_to_numpy)

        # upsample the feature maps to the same size as the input image using Nearest neighbour interpolation
        # upsample the feature map from 1024x512 to 2048x1024
        #classMap_numpy = cv2.resize(classMap_numpy, (args.inWidth*2, args.inHeight*2), interpolation=cv2.INTER_NEAREST)
        if i % 100 == 0 and i > 0:
            print('Processed [{}/{}]'.format(i, len(image_list)))

        name = imgName.split('/')[-1]
        if args.colored:
            start_color = time.time()
            classMap_numpy_color = np.zeros((img.shape[1], img.shape[2], img.shape[0]), dtype=np.uint8)
            for idx in range(len(pallete)):
                [r, g, b] = pallete[idx]
                classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
            elapsed_color = time.time() - start_color
            #print ('color : ', elapsed_color)

            cv2.imwrite(args.savedir + os.sep + 'c_' + name.replace(args.img_extn, 'png'), classMap_numpy_color)
            if args.overlay:
                overlayed = cv2.addWeighted(img_orig, 0.5, classMap_numpy_color, 0.5, 0)
                cv2.imwrite(args.savedir + os.sep + 'over_' + name.replace(args.img_extn, 'jpg'), overlayed)

        #if args.cityFormat:
        #    classMap_numpy = relabel(classMap_numpy.astype(np.uint8))


        cv2.imwrite(args.savedir + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy)

        times = OrderedDict((('read',elapsed_read), ('mean_std',elapsed_mean_std),
                             ('resize',elapsed_resize), ('numpy_to_tensor',elapsed_numpy_to_tensor),
                             ('set_gpu',elapsed_set_gpu), ('inference',elapsed_inference),
                             ('tensor_to_numpy',elapsed_tensor_to_numpy), ('color',elapsed_color)))


        num_of_times = len(times)

        if i >= 3:
            count = 0
            for data_label, data in times.items():
                sum_times[count] += data
                count += 1

    print ('Average times')

    count = 0

    for data_label, data in times.items():
        #(num_of_files - 4) means that discard first ~ forth process
        data = sum_times[count] / (num_of_files - 3)
        print (data, ':', data_label)
        count += 1


def evaluateModel_test(args, model):
    # gloabl mean and std values
    mean = [131.84157, 145.38597, 135.16437]
    std =  [76.013596, 67.85283,  70.89791 ]

    model.eval()
    for i in range(0,100):
        img_tensor = torch.randn(1, 3, args.inHeight, args.inWidth)
        if args.gpu:
            img_tensor = img_tensor.cuda()
        #0.088
        start = time.time()
        img_out = model(img_tensor)

        classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()
        elapsed_time = time.time() - start
        print ('time : ', i)
        print (elapsed_time)

def main(args):
    # read all the images in the folder
    image_list = glob.glob(args.data_dir + os.sep + '*.' + args.img_extn)

    modelA = net.EESPNet_Seg(args.classes, s=args.s)
    if not os.path.isfile(args.pretrained):
        print('Pre-trained model file does not exist. Please check ./pretrained_models folder')
        exit(-1)
    modelA = nn.DataParallel(modelA)
    modelA.load_state_dict(torch.load(args.pretrained))
    if args.gpu:
        modelA = modelA.cuda()

    # set to evaluation mode
    modelA.eval()

    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)


    evaluateModel(args, modelA, image_list)
    #evaluateModel_test(args, modelA)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNetv2", help='Model name')
    parser.add_argument('--data_dir', default="./izunuma/test_image", help='Data directory')
    parser.add_argument('--img_extn', default="png", help='RGB Image format')
    parser.add_argument('--inWidth', type=int, default=768, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=432, help='Height of RGB image')
    parser.add_argument('--savedir', default='./izunuma/test_image/results', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--pretrained', default='../models/izunuma_dataset9_1.5/model_best.pth', help='Pretrained weights directory.')
    parser.add_argument('--s', default=0.5, type=float, help='scale')
    parser.add_argument('--cityFormat', default=True, type=bool, help='If you want to convert to cityscape '
                                                                       'original label ids')
    parser.add_argument('--colored', default=True, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks in color')
    parser.add_argument('--overlay', default=False, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks overlayed on top of RGB image')
    parser.add_argument('--classes', default=11, type=int, help='Number of classes in the dataset. 20 for Cityscapes')

    args = parser.parse_args()
    if args.overlay:
        args.colored = True # This has to be true if you want to overlay
    main(args)
