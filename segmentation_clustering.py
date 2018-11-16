import cv2
import numpy as np
import time

def main():
    img = cv2.imread(img_path, 0)
    img = img.astype(np.int8)
    cv2.imwrite(img_path, img)
    img = cv2.imread(img_path, 0)

    pole_threshold = Pole_Threshold(img)
    start = time.time()
    pole_remove_noise = Open_Close(pole_threshold)
    elapsed_time = time.time() - start
    print elapsed_time


    '''
    color_src01 = cv2.cvtColor(pole_threshold, cv2.COLOR_GRAY2BGR)
    color_src02 = cv2.cvtColor(pole_threshold, cv2.COLOR_GRAY2BGR)
    label = cv2.connectedComponentsWithStats(pole_threshold)
    '''
    color_src01 = cv2.cvtColor(pole_remove_noise, cv2.COLOR_GRAY2BGR)
    color_src02 = cv2.cvtColor(pole_remove_noise, cv2.COLOR_GRAY2BGR)
    label = cv2.connectedComponentsWithStats(pole_remove_noise)

    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)

    for i in range(n):

        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(color_src01, (x0, y0), (x1, y1), (0, 0, 255))
        cv2.rectangle(color_src02, (x0, y0), (x1, y1), (0, 0, 255))

        cv2.putText(color_src01, "ID: " +str(i + 1), (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        cv2.putText(color_src01, "S: " +str(data[i][4]), (x1 - 20, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        cv2.putText(color_src02, "X: " + str(int(center[i][0])), (x1 - 30, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        cv2.putText(color_src02, "Y: " + str(int(center[i][1])), (x1 - 30, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    cv2.imshow("pole_threshold", pole_threshold)
    cv2.imshow("pole_remove_noise", pole_remove_noise)
    cv2.imshow("color_src01", color_src01)
    cv2.imshow("color_src02", color_src02)


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
