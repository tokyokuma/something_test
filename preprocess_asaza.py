import cv2
import glob
import os
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt

def Split_rgb(rgb_img):
    #split R G B
    split_rgb = cv2.split(rgb_img)

    return split_rgb

def High_contrast(min, max, img):
    #re_contrast table
    min_table = min
    max_table = max
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )

    # create high-contrast LUT
    for a in range(0, min_table):
        LUT_HC[a] = 0
    for a in range(min_table, max_table):
        LUT_HC[a] = 255 * (a - min_table) / diff_table
    for a in range(max_table, 255):
        LUT_HC[a] = 255

    return cv2.LUT(img, LUT_HC)

def Sharp(high_cont):
    #sharpened
    k = 1.0
    op = np.array([[-k, k,         -k],
                   [-k, 1 + 6 * k, -k],
                   [-k, -k,        -k]])

    img_tmp = cv2.filter2D(high_cont, -1 , op)
    return cv2.convertScaleAbs(img_tmp)

def Canny_edge(high_cont):
    #edges
    gray_img = cv2.cvtColor(high_cont, cv2.COLOR_RGB2GRAY)
    return gray_img, cv2.Canny(gray_img,200,240)

def Recolor_orig(rgb_img):
    recolor_img = rgb_img.copy()
    cond = (recolor_img[..., 2] >= recolor_img[..., 0]) & (recolor_img[..., 2] >= recolor_img[..., 1])
    recolor_img[cond] = [0, 0, 0]

    return recolor_img

def Recolor_high(high_cont, red, green, blue):
    recolor_img = high_cont.copy()
    cond_p = (recolor_img[..., 0] >= red) & (recolor_img[..., 1] >= green) & (recolor_img[..., 2] <= blue)
    cond_f = np.logical_not(cond_p)
    recolor_img[cond_p] = [0, 255, 0]
    recolor_img[cond_f] = [0, 0, 0]

    return recolor_img

def Recolor_hsv(hsv_img, h, s, v):
    recolor_img = hsv_img.copy()
    #cond_p = (hsv_img[..., 2] >= h)
    cond_p = (hsv_img[..., 2] >= h) & (hsv_img[..., 1] >= s) & (hsv_img[..., 0] <= v)
    cond_f = np.logical_not(cond_p)
    recolor_img[cond_p] = [0, 255, 0]
    recolor_img[cond_f] = [0, 0, 0]


    return recolor_img

def Recolor_gray(gray_img, height, width):
    for y in range(0, height):
        for x in range(0, width):
            if gray_img.item(y, x) == 150:
                gray_img.itemset((y, x), 20)
            else :
                gray_img.itemset((y, x), 35)

    return gray_img

def Open_Close(high_cont):
    img_tmp = high_cont
    element8 = np.array([[1,1,1], [1,1,1], [1,1,1]], np.uint8)
    iteration = 1
    while iteration <= 20:
        img_tmp = cv2.morphologyEx(img_tmp, cv2.MORPH_OPEN, element8)
        img_tmp = cv2.morphologyEx(img_tmp, cv2.MORPH_CLOSE, element8)
        iteration = iteration + 1
    return img_tmp

def Gaussian(high_cont):
    return cv2.GaussianBlur(high_cont,(5,5),0)

def Hsv(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

def Threshold(gray_img):
    return cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

def Hough(edge, img_orig):
    lines = cv2.HoughLines(edge,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img_orig,(x1,y1),(x2,y2),(0,0,255),2)

    return img_orig

def Histgram(rgb_roi_bottom):
    histr_r = cv2.calcHist([rgb_roi_bottom],[0],None,[256],[0,256])
    histr_g = cv2.calcHist([rgb_roi_bottom],[1],None,[256],[0,256])
    histr_b = cv2.calcHist([rgb_roi_bottom],[2],None,[256],[0,256])
    histr = [histr_r, histr_g, histr_b]

    return histr

def main():
    src_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/asaza_gagabuta'
    dst_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/label_asaza'
    origin_filelist = glob.glob(os.path.join(src_dir, '*.png'))

    base_img = cv2.imread(os.path.join(src_dir, 'frame016987.png'))

    if len(base_img.shape) == 3:
        height, width, channels = base_img.shape[:3]
    else:
        height, width = base_img.shape[:2]
        channels = 1
    zeros = np.zeros((height, width), base_img.dtype)

    roiWidth = width
    roiHeight= height / 2
    sx = 0
    sy = height - roiHeight
    ex = width
    ey = height

    for orig_file in origin_filelist:
        img_orig = cv2.imread(orig_file)
        orig_filename = os.path.basename(orig_file)

        rgb_img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        rgb_img_cp = rgb_img.copy()
        rgb_img_cp = Recolor_orig(rgb_img_cp)
        hsv_img = Hsv(rgb_img_cp)
        hsv_roi_bottom = hsv_img[sy:ey,sx:ex]
        histgram_hsv = Histgram(hsv_roi_bottom)
        recolor_hsv = Recolor_hsv(hsv_img, 130, 20, 50)

        gray_img = cv2.cvtColor(recolor_hsv, cv2.COLOR_RGB2GRAY)
        gray_img_cp = gray_img.copy()
        recolor_gray_img = Recolor_gray(gray_img_cp, height, width)
        roi_bottom = recolor_gray_img[sy:ey,sx:ex]

        cv2.imwrite(os.path.join(dst_dir, orig_filename), roi_bottom)
        print orig_filename

    print 'finish'

def test1():
    dst_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/asaza_test'
    parser = argparse.ArgumentParser(description="color parameter")
    parser.add_argument("img_num", type=str, default='016987', help="b thershold")

    args = parser.parse_args()
    img_file = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_all/frame' + args.img_num + '.png'

    img_orig = cv2.imread(img_file)

    if len(img_orig.shape) == 3:
        height, width, channels = img_orig.shape[:3]
    else:
        height, width = img_orig.shape[:2]
        channels = 1
    zeros = np.zeros((height, width), img_orig.dtype)

    roiWidth = width
    roiHeight= height / 2
    sx = 0
    sy = height - roiHeight
    ex = width
    ey = height

    color = ['r','g','b']

    rgb_img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    rgb_img_cp = rgb_img.copy()
    rgb_img_cp = Recolor_orig(rgb_img_cp)
    hsv_img = Hsv(rgb_img_cp)
    hsv_roi_bottom = hsv_img[sy:ey,sx:ex]
    histgram_hsv = Histgram(hsv_roi_bottom)
    recolor_hsv = Recolor_hsv(hsv_img, 120,20,50)
    open_close = Open_Close(recolor_hsv)
    gray_img = cv2.cvtColor(open_close, cv2.COLOR_RGB2GRAY)
    gray_img_cp = gray_img.copy()
    recolor_gray_img = Recolor_gray(gray_img_cp, height, width)
    roi_bottom = recolor_gray_img[sy:ey,sx:ex]

    plt.subplot(3,3,1),plt.imshow(rgb_img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,2),plt.imshow(rgb_img_cp)
    plt.title('Recolor orig'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,3),plt.imshow(hsv_img)
    plt.title('Hsv'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,4),plt.imshow(hsv_roi_bottom)
    plt.title('Hsv Roi'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,5),plt.imshow(recolor_hsv)
    plt.title('Recolor hsv'),plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,6),plt.imshow(open_close)
    plt.title('Open Close'),plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,7),plt.imshow(gray_img, cmap='gray')
    plt.title('Gray Image'),plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,8),plt.imshow(recolor_gray_img, cmap='gray')
    plt.title('Recolor Gray Image'),plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,9),plt.imshow(roi_bottom),plt.gray()
    plt.title('Roi bottom'),plt.xticks([]), plt.yticks([])

    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    rgb_img_cp = cv2.cvtColor(rgb_img_cp, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_RGB2BGR)
    hsv_roi_bottom = cv2.cvtColor(hsv_roi_bottom, cv2.COLOR_RGB2BGR)


    cv2.imwrite(os.path.join(dst_dir, 'Rgb_orig.png'), rgb_img)
    cv2.imwrite(os.path.join(dst_dir, 'Reclor_orig.png'), rgb_img_cp)
    cv2.imwrite(os.path.join(dst_dir, 'Hsv.png'), hsv_img)
    cv2.imwrite(os.path.join(dst_dir, 'Hsv_Roi.png'), hsv_roi_bottom)
    cv2.imwrite(os.path.join(dst_dir, 'Recolor_hsv.png'), recolor_hsv)
    cv2.imwrite(os.path.join(dst_dir, 'Open_close.png'), open_close)
    cv2.imwrite(os.path.join(dst_dir, 'Gray_Image.png'), gray_img)
    cv2.imwrite(os.path.join(dst_dir, 'Recolor_Gray_Image.png'), recolor_gray_img)
    cv2.imwrite(os.path.join(dst_dir, 'Roi_bottom.png'), roi_bottom)

    plt.show()

def test2():
    dst_dir = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/asaza_test'
    parser = argparse.ArgumentParser(description="color parameter")
    parser.add_argument("img_num", type=str, default='016987', help="b thershold")

    args = parser.parse_args()

    img_file = '/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_all/frame' + args.img_num + '.png'

    img_orig = cv2.imread(img_file)

    if len(img_orig.shape) == 3:
        height, width, channels = img_orig.shape[:3]
    else:
        height, width = img_orig.shape[:2]
        channels = 1
    zeros = np.zeros((height, width), img_orig.dtype)

    roiWidth = width
    roiHeight= height / 2
    sx = 0
    sy = height - roiHeight
    ex = width
    ey = height

    color = ['r','g','b']

    rgb_img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    high_cont = High_contrast(110, 200, rgb_img)
    recolor_high = Recolor_high(high_cont, 10, 10, 120, )
    high_cont_roi_bottom = recolor_high[sy:ey,sx:ex]
    gray_img = cv2.cvtColor(recolor_high, cv2.COLOR_RGB2GRAY)
    gray_img_cp = gray_img.copy()
    recolor_gray_img = Recolor_gray(gray_img_cp, height, width)
    roi_bottom = recolor_gray_img[sy:ey,sx:ex]

    plt.subplot(3,3,1),plt.imshow(rgb_img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,2),plt.imshow(high_cont)
    plt.title('High Contrast'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,3),plt.imshow(recolor_high)
    plt.title('Recolor High Contrast'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,4),plt.imshow(high_cont_roi_bottom)
    plt.title('High Contrast Roi'), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,5),plt.imshow(gray_img, cmap='gray')
    plt.title('Gray Image'),plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,6),plt.imshow(recolor_gray_img, cmap='gray')
    plt.title('Recolor Gray Image'),plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,7),plt.imshow(roi_bottom),plt.gray()
    plt.title('Roi bottom'),plt.xticks([]), plt.yticks([])

    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    high_cont = cv2.cvtColor(high_cont, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(dst_dir, 'Rgb_orig.png'), rgb_img)
    cv2.imwrite(os.path.join(dst_dir, 'High_contrast.png'), high_cont)
    cv2.imwrite(os.path.join(dst_dir, 'Recolor_high_contrast.png'), recolor_high)
    cv2.imwrite(os.path.join(dst_dir, 'High_contrast_roi.png'), high_cont_roi_bottom)
    cv2.imwrite(os.path.join(dst_dir, 'Gray_Image.png'), gray_img)
    cv2.imwrite(os.path.join(dst_dir, 'Recolor_Gray_Image.png'), recolor_gray_img)
    cv2.imwrite(os.path.join(dst_dir, 'Roi_bottom.png'), roi_bottom)

    plt.show()


if __name__ == '__main__':
    #main()
    test1()
