import cv2
import numpy as np

img_src = '/home/kuma/something_test/dataset_izunuma/test/frame018303_color_mask.png'
img = cv2.imread(img_src)

water_sur = [128, 128, 128]

h, w, c = img.shape

for y in range(0, h):
    for x in range(0, w):
        orig_B = img.item(y, x, 0)
        orig_G = img.item(y, x, 1)
        orig_R = img.item(y, x, 2)
        orig_BGR = [orig_B, orig_G, orig_R]

        if orig_BGR == water_sur:
            img[y, x] = [0, 0, 0]
            print img[y, x]
        else:
            pass
