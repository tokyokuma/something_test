import cv2
import numpy as np


label = cv2.imread('/home/kuma/something_test/dataset_izunuma/scripts/004964.png', 0)
print np.where(label > 10) = 255
