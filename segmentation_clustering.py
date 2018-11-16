import cv2
import numpy as np
import time

def main():
    img = cv2.imread(img_path, 0)
    img = img.astype(np.int8)
    cv2.imwrite(img_path, img)
    img = cv2.imread(img_path, 0)


    #0.0083sec
    pole_threshold = Pole_Threshold(img)
    #0.0011sec
    pole_remove_noise = Open_Close(pole_threshold)
    start = time.time()
    #0.0040sec
    label = cv2.connectedComponentsWithStats(pole_remove_noise)
    elapsed_time = time.time() - start
    print elapsed_time

    n = label[0] - 1

    #remove backgrouond label
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)

    obstacle_coordination = np.zeros([n,2], dtype=np.float32)

    for i in range(n):
        obstacle_coordination[i-1][0] = int(center[i][1])
        obstacle_coordination[i-1][1] = int(center[i][0])

    print obstacle_coordination


    #cv2.imshow("pole_threshold", pole_threshold)
    #cv2.imshow("pole_remove_noise", pole_remove_noise)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Pole_Threshold(img):
    recolor_img = img.copy()
    pole = [1]
    black = [0]
    white = [255]
    recolor_img[np.where(recolor_img != pole)] = black
    recolor_img[np.where(recolor_img == pole)] = white

    return recolor_img

def Open_Close(img):
    img_tmp = img
    element8 = np.ones((10, 10), np.uint8)
    iteration = 1
    while iteration <= 1:
        img_tmp = cv2.morphologyEx(img_tmp, cv2.MORPH_OPEN, element8)
        img_tmp = cv2.morphologyEx(img_tmp, cv2.MORPH_CLOSE, element8)
        iteration = iteration + 1
    return img_tmp

if __name__ == '__main__':
    img_path = '/home/kuma/something_test/dataset_izunuma/test/segmentation_clustering/gray.png'
    main()
