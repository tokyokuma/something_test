import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_all/frame017246.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#label = cv2.imread('/home/kuma/something_test/dataset_izunuma/izunuma_dataset1/izunuma_dataset1_all/frame017246_watershed_mask.png')

height, width, channels = img.shape

image_zeros = np.zeros((height, width), np.uint8)

for y in range(0, height):
    for x in range(0, width):
        R = img.item(y,x,0)
        G = img.item(y,x,1)
        B = img.item(y,x,2)
        #print 'R : ' + str(R) + 'G : ' + str(G) + 'B : ' + str(B)
        if G + R - B != 0:
            VARI = float((G - R)) / float((G + R -B))
        else:
            VARI = 0
        VARI = VARI * 255
        VARI = int(VARI)
        image_zeros.itemset(y, x, VARI)

cv2.imwrite('/home/kuma/something_test/dataset_izunuma/VARI.png', image_zeros)

plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(image_zeros, cmap='gray')
plt.title('VARI'), plt.xticks([]), plt.yticks([])

plt.show()
